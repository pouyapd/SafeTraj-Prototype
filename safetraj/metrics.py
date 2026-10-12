"""
metrics.py — model-agnostic trajectory metrics.

All metrics use only the predicted trajectory and its goal, so they work for
any predictor (neural network, physics model, recorded data).

Goal outcome (same definitions as the MSc thesis, SafeTraj-Experiments):
    strict success    final distance to goal d_T <  strict_threshold (0.30 m)
    soft outcome      success (d_T < 0.30) / near_success (0.30 <= d_T < 0.50) / failure (d_T >= 0.50)
Thresholds are configurable (SafeTrajConfig).
"""

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .trajectory import Trajectory


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class TrajectoryMetrics:
    n_points: int
    duration: float              # t_end - t_start (units of t)
    d_start: float               # distance to goal at the first point (m)
    d_final: float               # distance to goal at the last point (m)
    d_min: float                 # closest approach to the goal (m)
    progress: float              # d_start - d_final (m); positive = moved closer
    path_length: float           # metres travelled
    path_efficiency: float       # progress / path_length (1 = straight toward goal; <= 0 = no progress)
    heading_error_final: float   # |heading - bearing to goal| at the last point (rad)
    total_turning: float         # sum of |heading changes| (rad)
    max_curvature: float         # max |dtheta / ds| over segments longer than 1 mm (1/m)
    v_max_abs: float             # max |v| (m/s)
    reverse_fraction: float      # share of samples with v < 0
    omega_max_abs: float         # max |omega| (rad/s)
    omega_mean_abs: float        # mean |omega| (rad/s)
    domega_max: float            # max |omega_t - omega_{t-1}| per sample (rad/s per sample)
    outcome_strict: str          # "success" | "failure"
    outcome_soft: str            # "success" | "near_success" | "failure"

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_metrics(traj: Trajectory, strict_threshold: float = 0.30, near_threshold: float = 0.50) -> TrajectoryMetrics:
    """Compute all trajectory metrics (see TrajectoryMetrics for definitions)."""
    if not 0 < strict_threshold < near_threshold:
        raise ValueError("need 0 < strict_threshold < near_threshold")
    gx, gy = traj.goal
    d = np.hypot(traj.x - gx, traj.y - gy)
    seg = np.hypot(np.diff(traj.x), np.diff(traj.y))
    path = float(seg.sum())
    progress = float(d[0] - d[-1])

    bearing = np.arctan2(gy - traj.y[-1], gx - traj.x[-1])
    heading_err = float(abs(wrap_angle(traj.theta[-1] - bearing))) if d[-1] > 1e-9 else 0.0
    dtheta = np.abs(wrap_angle(np.diff(traj.theta)))
    moving = seg > 1e-3
    max_curv = float(np.max(dtheta[moving] / seg[moving])) if np.any(moving) else 0.0
    domega = np.abs(np.diff(traj.omega))

    d_final = float(d[-1])
    if d_final < strict_threshold:
        soft = "success"
    elif d_final < near_threshold:
        soft = "near_success"
    else:
        soft = "failure"

    return TrajectoryMetrics(
        n_points=len(traj),
        duration=float(traj.t[-1] - traj.t[0]),
        d_start=float(d[0]),
        d_final=d_final,
        d_min=float(d.min()),
        progress=progress,
        path_length=path,
        path_efficiency=float(progress / path) if path > 1e-9 else 0.0,
        heading_error_final=heading_err,
        total_turning=float(dtheta.sum()),
        max_curvature=max_curv,
        v_max_abs=float(np.max(np.abs(traj.v))),
        reverse_fraction=float(np.mean(traj.v < 0)),
        omega_max_abs=float(np.max(np.abs(traj.omega))),
        omega_mean_abs=float(np.mean(np.abs(traj.omega))),
        domega_max=float(domega.max()) if domega.size else 0.0,
        outcome_strict="success" if d_final < strict_threshold else "failure",
        outcome_soft=soft,
    )
