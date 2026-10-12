"""
ood.py — input-novelty (out-of-distribution) scores for commands.

A novelty score says how unusual a command (orientation, v_lin, v_rot) is
compared with a reference set of commands, for example the inputs a model was
trained or validated on. It does NOT say whether the predicted trajectory is
safe: a typical input can still produce a bad trajectory and an unusual input
can produce a good one. Use it to flag inputs where the model's behaviour is
less supported by data.

Scores are turned into percentiles of the reference set's own scores, so the
labels have a frequency meaning: with the default 95th / 99.5th percentiles,
about 5 % of reference inputs are "borderline" or "novel" by construction
(slightly more, because an input is flagged if either detector flags it).
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest

FEATURES = ("orientation", "v_lin", "v_rot")


class MahalanobisOOD:
    """Mahalanobis distance to the reference mean (in units of the reference covariance)."""

    def __init__(self, training_data: np.ndarray):
        training_data = np.asarray(training_data, dtype=float)
        self.cov = EmpiricalCovariance().fit(training_data)
        self.mean = self.cov.location_

    def score_many(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # EmpiricalCovariance.mahalanobis centres X at location_ itself (and
        # returns squared distances); version 1 subtracted the mean a second time.
        return np.sqrt(self.cov.mahalanobis(X))

    def score(self, x: np.ndarray) -> float:
        """Larger score => more unusual."""
        return float(self.score_many(x)[0])


class ISOForestOOD:
    """Isolation Forest anomaly score (positive = more anomalous)."""

    def __init__(self, training_data: np.ndarray, contamination: float = 0.05, seed: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=seed)
        self.model.fit(np.asarray(training_data, dtype=float))

    def score_many(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(np.atleast_2d(np.asarray(X, dtype=float)))

    def score(self, x: np.ndarray) -> float:
        return float(self.score_many(x)[0])


@dataclass
class NoveltyResult:
    label: str                       # "typical" | "borderline" | "novel"
    score: float                     # max of the two reference percentiles, in [0, 1]
    mahalanobis: float
    isolation_forest: float
    mahalanobis_percentile: float    # % of reference inputs with a lower or equal score
    isolation_forest_percentile: float
    outside_reference_range: bool    # a feature lies outside the reference min-max range
    attribution: Dict[str, float]    # Mahalanobis drop if the feature were set to the reference median

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


class NoveltyScorer:
    """Mahalanobis + Isolation Forest with percentile-based labels (see module docstring)."""

    def __init__(self, reference: np.ndarray, contamination: float = 0.05, seed: int = 42,
                 borderline_percentile: float = 95.0, novel_percentile: float = 99.5):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != len(FEATURES) or len(reference) < 20:
            raise ValueError(f"reference data must have shape (n >= 20, {len(FEATURES)})")
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference data contains NaN or infinite values")
        self.reference = reference
        self.maha = MahalanobisOOD(reference)
        self.iso = ISOForestOOD(reference, contamination=contamination, seed=seed)
        self._ref_maha = np.sort(self.maha.score_many(reference))
        self._ref_iso = np.sort(self.iso.score_many(reference))
        self.borderline_percentile = borderline_percentile
        self.novel_percentile = novel_percentile
        self.median = np.median(reference, axis=0)
        self.low, self.high = reference.min(axis=0), reference.max(axis=0)

    @staticmethod
    def _percentile(sorted_ref: np.ndarray, value: float) -> float:
        return float(100.0 * np.searchsorted(sorted_ref, value, side="right") / len(sorted_ref))

    def attribution(self, x: np.ndarray) -> Dict[str, float]:
        """
        Baseline-substitution attribution: how much the Mahalanobis distance
        drops when one feature is replaced by the reference median (others
        unchanged). Positive = this feature makes the input look unusual.
        Not additive; a local, score-based explanation of this one input.
        """
        base = self.maha.score(x)
        out = {}
        for i, name in enumerate(FEATURES):
            x2 = np.array(x, dtype=float)
            x2[i] = self.median[i]
            out[name] = float(base - self.maha.score(x2))
        return out

    def novelty_percentile_many(self, X: np.ndarray) -> np.ndarray:
        """Vectorised novelty score (0-100) for many inputs; same rule as score().label."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        pm = 100.0 * np.searchsorted(self._ref_maha, self.maha.score_many(X), side="right") / len(self._ref_maha)
        pf = 100.0 * np.searchsorted(self._ref_iso, self.iso.score_many(X), side="right") / len(self._ref_iso)
        outside = np.any(X < self.low, axis=1) | np.any(X > self.high, axis=1)
        return np.where(outside, 100.0, np.maximum(pm, pf))

    def score(self, x: Sequence[float]) -> NoveltyResult:
        x = np.asarray(x, dtype=float)
        if x.shape != (len(FEATURES),) or not np.all(np.isfinite(x)):
            raise ValueError(f"command must be {len(FEATURES)} finite numbers {FEATURES}")
        m, f = self.maha.score(x), self.iso.score(x)
        pm, pf = self._percentile(self._ref_maha, m), self._percentile(self._ref_iso, f)
        outside = bool(np.any(x < self.low) or np.any(x > self.high))
        pct = 100.0 if outside else max(pm, pf)
        if pct >= self.novel_percentile:
            label = "novel"
        elif pct >= self.borderline_percentile:
            label = "borderline"
        else:
            label = "typical"
        return NoveltyResult(label=label, score=pct / 100.0, mahalanobis=m, isolation_forest=f,
                             mahalanobis_percentile=pm, isolation_forest_percentile=pf,
                             outside_reference_range=outside, attribution=self.attribution(x))
