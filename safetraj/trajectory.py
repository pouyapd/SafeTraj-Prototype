"""
trajectory.py — the Trajectory data structure, CSV input/output and validation.

A trajectory is a time series of planar poses and commands toward a goal:
    t, x, y, theta, v, omega    (seconds, metres, radians, m/s, rad/s)
    goal = (goal_x, goal_y)

CSV layout (one row per time step, several trajectories per file):
    traj_id, t, x, y, theta, v, omega, goal_x, goal_y
`t` may be in seconds or in step indices; it only has to increase.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("traj_id", "t", "x", "y", "theta", "v", "omega", "goal_x", "goal_y")


class TrajectoryError(ValueError):
    """Raised for malformed trajectory input (missing columns, NaN, non-increasing time, ...)."""


@dataclass
class Trajectory:
    """One trajectory: arrays of equal length plus a goal."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    goal: Tuple[float, float]
    traj_id: Union[int, str] = 0
    command: Optional[Tuple[float, float, float]] = None   # (theta0, v0, omega0) if known
    source: str = ""
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        arrays = {k: np.asarray(getattr(self, k), dtype=float) for k in ("t", "x", "y", "theta", "v", "omega")}
        n = len(arrays["t"])
        if n < 2:
            raise TrajectoryError(f"trajectory {self.traj_id!r}: needs at least 2 points, got {n}")
        for name, arr in arrays.items():
            if arr.ndim != 1 or len(arr) != n:
                raise TrajectoryError(f"trajectory {self.traj_id!r}: column '{name}' must have {n} values")
            if not np.all(np.isfinite(arr)):
                raise TrajectoryError(f"trajectory {self.traj_id!r}: column '{name}' contains NaN or infinite values")
            setattr(self, name, arr)
        if np.any(np.diff(arrays["t"]) <= 0):
            raise TrajectoryError(f"trajectory {self.traj_id!r}: time 't' must be strictly increasing")
        goal = tuple(float(g) for g in self.goal)
        if len(goal) != 2 or not all(np.isfinite(goal)):
            raise TrajectoryError(f"trajectory {self.traj_id!r}: goal must be two finite numbers")
        self.goal = goal

    def __len__(self) -> int:
        return len(self.t)

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "traj_id": self.traj_id, "t": self.t, "x": self.x, "y": self.y, "theta": self.theta,
            "v": self.v, "omega": self.omega, "goal_x": self.goal[0], "goal_y": self.goal[1],
        })

    def to_dict(self) -> Dict:
        return {
            "traj_id": self.traj_id, "goal": list(self.goal),
            "command": list(self.command) if self.command is not None else None,
            "points": [
                {"t": float(t), "x": float(x), "y": float(y), "theta": float(th), "v": float(v), "omega": float(w)}
                for t, x, y, th, v, w in zip(self.t, self.x, self.y, self.theta, self.v, self.omega)
            ],
        }

    @classmethod
    def from_points(cls, points: Sequence[Dict], goal: Sequence[float], traj_id: Union[int, str] = 0,
                    command: Optional[Sequence[float]] = None) -> "Trajectory":
        """Build from a list of {t, x, y, theta, v, omega} dicts (the API's input format)."""
        cols = {k: [p[k] for p in points] for k in ("t", "x", "y", "theta", "v", "omega")}
        return cls(goal=tuple(goal), traj_id=traj_id, command=tuple(command) if command else None, **cols)


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns and types of a trajectory table; returns a cleaned copy sorted by (traj_id, t)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise TrajectoryError(f"missing column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLUMNS)}")
    out = df.copy()
    for col in REQUIRED_COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        bad = out[col].isna()
        if bad.any():
            rows = (np.flatnonzero(bad.to_numpy())[:5] + 2).tolist()   # +2: header line, 1-based
            raise TrajectoryError(f"column '{col}' has missing or non-numeric values (file lines {rows})")
    if out.empty:
        raise TrajectoryError("the trajectory table is empty")
    return out.sort_values(["traj_id", "t"], kind="stable").reset_index(drop=True)


def frame_to_trajectories(df: pd.DataFrame, source: str = "") -> List[Trajectory]:
    """Split a validated table into Trajectory objects (one per traj_id)."""
    df = validate_frame(df)
    trajs = []
    for traj_id, g in df.groupby("traj_id", sort=False):
        goals = g[["goal_x", "goal_y"]].drop_duplicates()
        if len(goals) > 1:
            raise TrajectoryError(f"trajectory {traj_id!r}: goal_x/goal_y must be constant within a trajectory")
        trajs.append(Trajectory(
            t=g["t"].to_numpy(), x=g["x"].to_numpy(), y=g["y"].to_numpy(), theta=g["theta"].to_numpy(),
            v=g["v"].to_numpy(), omega=g["omega"].to_numpy(),
            goal=(float(goals.iloc[0, 0]), float(goals.iloc[0, 1])),
            traj_id=traj_id.item() if hasattr(traj_id, "item") else traj_id, source=source,
        ))
    return trajs


def load_trajectories(path: Union[str, Path]) -> List[Trajectory]:
    """Read a trajectory CSV (see module docstring for the layout)."""
    path = Path(path)
    if not path.is_file():
        raise TrajectoryError(f"file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error messages vary
        raise TrajectoryError(f"could not read {path}: {exc}") from exc
    return frame_to_trajectories(df, source=str(path.name))


def trajectories_to_frame(trajs: Iterable[Trajectory]) -> pd.DataFrame:
    return pd.concat([t.to_frame() for t in trajs], ignore_index=True)
