from typing import Dict, Optional, Tuple

import numpy as np

from .config import OPERATIONAL_RANGES


def generate_training_data(
    n_samples: int = 800,
    seed: int = 42,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Synthetic reference commands, uniform over the operational ranges of the
    smart-wheelchair models (orientation, v_lin, v_rot). Replace them with the
    inputs your own model was trained on via SafeTrajEvaluator(reference_data=...).

    Returns:
        Array of shape (n_samples, 3): [orientation, v_lin, v_rot] per row.
    """
    ranges = ranges or OPERATIONAL_RANGES
    rng = np.random.default_rng(seed)
    low = np.array([ranges["orientation"][0], ranges["v_lin"][0], ranges["v_rot"][0]])
    high = np.array([ranges["orientation"][1], ranges["v_lin"][1], ranges["v_rot"][1]])
    return rng.uniform(low=low, high=high, size=(n_samples, 3))
