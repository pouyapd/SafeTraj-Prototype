"""
xai.py — explanation helpers.

Explanations in this toolkit come from three places:
  * rule checks: every fired rule reports the measured value and its threshold
    (safetraj/rules.py) — this is a complete account of the check result;
  * novelty attribution: how much the Mahalanobis score drops when one input
    feature is set to the reference median (NoveltyScorer.attribution);
  * failure rules over many inputs: a shallow decision tree fitted to sweep
    results (safetraj/analysis.py).

`simple_feature_importance` is the version 1 helper. It only reports each
input's share of the total magnitude |x_i| / sum |x_j|; it does not look at any
model or score and is kept for backward compatibility only.
"""

from typing import Dict

import numpy as np


def simple_feature_importance(x: np.ndarray) -> Dict[str, float]:
    """Deprecated: |x_i| / sum_j |x_j| (magnitude share, not model-based importance)."""
    names = ["orientation", "v_lin", "v_rot"]
    abs_x = np.abs(np.asarray(x, dtype=float))
    total = float(np.sum(abs_x) + 1e-6)
    return {name: float(val) for name, val in zip(names, abs_x / total)}
