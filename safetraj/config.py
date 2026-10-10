from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# Operational command ranges of the smart-wheelchair models studied in the
# MSc thesis (SafeTraj-Experiments): orientation, linear and angular velocity.
OPERATIONAL_RANGES: Dict[str, Tuple[float, float]] = {
    "orientation": (-3.141592653589793, 3.141592653589793),   # rad
    "v_lin": (-1.05, 2.88),                                   # m/s
    "v_rot": (-1.99, 1.99),                                   # rad/s
}


@dataclass
class SafeTrajConfig:
    """
    Configuration for SafeTrajEvaluator. Every value is validated on creation.

    Reference data (for input-novelty scores)
        n_train_samples, seed      size and seed of the synthetic reference set
                                   (uniform over OPERATIONAL_RANGES) used when no
                                   reference data is supplied
        contamination              Isolation Forest contamination parameter
        borderline_percentile      novelty label "borderline" from this percentile
        novel_percentile           novelty label "novel" from this percentile
                                   (percentiles of the reference set's own scores)

    Trajectory prediction (built-in kinematic predictors)
        dt, num_steps              time step (s) and number of predicted steps

    Goal outcome thresholds (as in the thesis)
        strict_threshold           success if final distance to goal < this (m)
        near_threshold             near-success below this (m), failure above

    Rule-based checks (heuristic thresholds; see safetraj/rules.py)
        rule_progress_min, rule_omega_max, rule_domega_max, rule_heading_far
    """

    # reference data / novelty
    n_train_samples: int = 800
    seed: int = 42
    contamination: float = 0.05
    borderline_percentile: float = 95.0
    novel_percentile: float = 99.5

    # trajectory prediction
    dt: float = 0.1
    num_steps: int = 30

    # goal outcome
    strict_threshold: float = 0.30
    near_threshold: float = 0.50
    default_goal: Tuple[float, float] = (1.0, 0.0)

    # rule-based checks
    rule_progress_min: float = 0.10
    rule_omega_max: float = 1.99
    rule_domega_max: float = 1.5
    rule_heading_far: float = 1.0

    # Deprecated (version 1 thresholds on raw scores); accepted but unused.
    maha_thr: Optional[float] = None
    iso_thr: Optional[float] = None
    horizon: Optional[float] = None

    operational_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(OPERATIONAL_RANGES))

    def __post_init__(self):
        errors = []
        if not (isinstance(self.n_train_samples, int) and self.n_train_samples >= 20):
            errors.append("n_train_samples must be an integer >= 20")
        if not 0 < self.contamination <= 0.5:
            errors.append("contamination must be in (0, 0.5]")
        if not 0 < self.borderline_percentile < self.novel_percentile <= 100:
            errors.append("need 0 < borderline_percentile < novel_percentile <= 100")
        if self.dt <= 0:
            errors.append("dt must be positive")
        if not (isinstance(self.num_steps, int) and self.num_steps >= 1):
            errors.append("num_steps must be a positive integer")
        if not 0 < self.strict_threshold < self.near_threshold:
            errors.append("need 0 < strict_threshold < near_threshold")
        for name in ("rule_progress_min", "rule_omega_max", "rule_domega_max", "rule_heading_far"):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if len(tuple(self.default_goal)) != 2:
            errors.append("default_goal must be (x, y)")
        if errors:
            raise ValueError("Invalid SafeTrajConfig: " + "; ".join(errors))
