"""
analysis.py — failure analysis over many commands for any predictor.

The same workflow as the MSc thesis (SafeTraj-Experiments), but for any object
with predict(command, goal) -> Trajectory:

  1. sample commands uniformly over the operational ranges (fixed seed);
  2. predict a trajectory per command and goal; label the strict outcome;
  3. risk map: share of strict failures per (feature, feature) bin;
  4. a shallow decision tree turns the failure regions into if-then rules.

Unlike the thesis tables (training accuracy only), the tree report here also
gives 5-fold cross-validated accuracy and the majority-class baseline, so a
reader can see how much the rules add beyond always predicting the most
common outcome.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from .config import OPERATIONAL_RANGES, SafeTrajConfig
from .metrics import compute_metrics

FEATURE_COLUMNS = ["theta", "v", "omega", "goal_x", "goal_y"]


def sample_commands(n: int, seed: int = 42, ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """n commands uniform over the ranges; shape (n, 3) = [orientation, v_lin, v_rot]."""
    ranges = ranges or OPERATIONAL_RANGES
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(*ranges[k], size=n) for k in ("orientation", "v_lin", "v_rot")]
    return np.column_stack(cols)


def run_sweep(predictor, commands: np.ndarray, goals: Sequence[Sequence[float]],
              config: Optional[SafeTrajConfig] = None) -> pd.DataFrame:
    """Predict and label every (command, goal) pair. The same commands are reused for every goal."""
    config = config or SafeTrajConfig()
    rows = []
    for gx, gy in goals:
        for theta, v, omega in commands:
            traj = predictor.predict((theta, v, omega), (gx, gy))
            m = compute_metrics(traj, config.strict_threshold, config.near_threshold)
            rows.append({"theta": theta, "v": v, "omega": omega, "goal_x": gx, "goal_y": gy,
                         "d_final": m.d_final, "outcome_strict": m.outcome_strict, "outcome_soft": m.outcome_soft,
                         "failure": int(m.outcome_strict == "failure")})
    return pd.DataFrame(rows)


def risk_map(df: pd.DataFrame, x: str = "theta", y: str = "omega", bins: Tuple[int, int] = (8, 8),
             ranges: Optional[Dict[str, Tuple[float, float]]] = None, min_count: int = 3) -> Dict[str, Any]:
    """
    Empirical risk R(bin) = share of strict failures among the samples in the bin.
    Bins with fewer than `min_count` samples are reported as NaN (too sparse).
    """
    key = {"theta": "orientation", "v": "v_lin", "omega": "v_rot"}
    ranges = ranges or OPERATIONAL_RANGES
    xr = ranges.get(key.get(x, x), (df[x].min(), df[x].max()))
    yr = ranges.get(key.get(y, y), (df[y].min(), df[y].max()))
    xe = np.linspace(xr[0], xr[1], bins[0] + 1)
    ye = np.linspace(yr[0], yr[1], bins[1] + 1)
    xi = np.clip(np.digitize(df[x], xe) - 1, 0, bins[0] - 1)
    yi = np.clip(np.digitize(df[y], ye) - 1, 0, bins[1] - 1)
    fails = np.zeros((bins[1], bins[0]))
    counts = np.zeros((bins[1], bins[0]))
    np.add.at(fails, (yi, xi), df["failure"].to_numpy())
    np.add.at(counts, (yi, xi), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(counts >= min_count, fails / counts, np.nan)
    return {"rate": rate, "counts": counts, "x_edges": xe, "y_edges": ye, "x": x, "y": y}


def leaf_rules(tree: DecisionTreeClassifier, feature_names: List[str], target_class: int = 1) -> List[Dict[str, Any]]:
    """
    One readable rule per leaf that predicts `target_class` (1 = failure):
    the conditions on the path, the number of samples and the failure rate.
    """
    t = tree.tree_
    out: List[Dict[str, Any]] = []

    def render(bounds: Dict[str, List[float]]) -> str:
        parts = []
        for name in feature_names:
            if name not in bounds:
                continue
            lo, hi = bounds[name]
            if lo > -np.inf and hi < np.inf:
                parts.append(f"{lo:.2f} < {name} <= {hi:.2f}")
            elif lo > -np.inf:
                parts.append(f"{name} > {lo:.2f}")
            else:
                parts.append(f"{name} <= {hi:.2f}")
        return " and ".join(parts) or "(always)"

    def walk(node: int, bounds: Dict[str, List[float]]) -> None:
        if t.children_left[node] == t.children_right[node]:          # leaf
            counts = t.value[node][0]
            total = float(counts.sum())
            share = counts / total if total > 0 else counts
            if int(np.argmax(share)) == target_class:
                out.append({"conditions": render(bounds), "samples": int(t.n_node_samples[node]),
                            "failure_rate": float(share[1]) if len(share) > 1 else 0.0})
            return
        name, thr = feature_names[t.feature[node]], float(t.threshold[node])
        lo, hi = bounds.get(name, [-np.inf, np.inf])
        walk(t.children_left[node], {**bounds, name: [lo, min(hi, thr)]})
        walk(t.children_right[node], {**bounds, name: [max(lo, thr), hi]})

    walk(0, {})
    return sorted(out, key=lambda r: -r["samples"])


@dataclass
class TreeReport:
    rules_text: str
    train_accuracy: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    majority_baseline: float
    n_samples: int
    n_failures: int
    n_leaves: int
    feature_importances: Dict[str, float]
    failure_rules: List[Dict[str, Any]]
    model: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("model")
        return d


def fit_failure_tree(df: pd.DataFrame, features: Optional[List[str]] = None, max_depth: int = 4,
                     min_samples_leaf: int = 10, seed: int = 42) -> TreeReport:
    """Shallow decision tree on the strict failure label (thesis settings by default)."""
    features = features or [c for c in FEATURE_COLUMNS if df[c].nunique() > 1]
    X, yv = df[features].to_numpy(), df["failure"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
    tree.fit(X, yv)
    n_fail = int(yv.sum())
    if 0 < n_fail < len(yv) and min(n_fail, len(yv) - n_fail) >= 5:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                        random_state=seed), X, yv, cv=cv)
        cv_mean, cv_std = float(scores.mean()), float(scores.std())
    else:
        cv_mean = cv_std = float("nan")
    names = {"theta": "orientation", "v": "v_lin", "omega": "v_rot"}
    readable = [names.get(f, f) for f in features]
    text = export_text(tree, feature_names=readable, decimals=2)
    text = text.replace("class: 0", "class: success").replace("class: 1", "class: failure")
    return TreeReport(
        rules_text=text, train_accuracy=float(tree.score(X, yv)), cv_accuracy_mean=cv_mean, cv_accuracy_std=cv_std,
        majority_baseline=float(max(yv.mean(), 1 - yv.mean())), n_samples=len(yv), n_failures=n_fail,
        n_leaves=int(tree.get_n_leaves()),
        feature_importances={names.get(f, f): float(v) for f, v in zip(features, tree.feature_importances_)},
        failure_rules=leaf_rules(tree, readable),
        model=tree,
    )


def summarize_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Strict success rate, average final distance and soft-outcome shares per goal."""
    g = df.groupby(["goal_x", "goal_y"])
    out = pd.DataFrame({
        "samples": g.size(),
        "strict_success_rate": g["failure"].apply(lambda s: 1 - s.mean()),
        "avg_final_distance": g["d_final"].mean(),
        "near_success_share": g["outcome_soft"].apply(lambda s: (s == "near_success").mean()),
        "failure_share_soft": g["outcome_soft"].apply(lambda s: (s == "failure").mean()),
    }).reset_index()
    return out
