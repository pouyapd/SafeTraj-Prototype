import numpy as np
import pandas as pd


def angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def compute_features(traj: pd.DataFrame) -> dict:
    """
    Compute compact features from one trajectory.
    Expected columns: x,y,theta,v,omega,goal_x,goal_y
    """
    x = traj["x"].to_numpy(dtype=float)
    y = traj["y"].to_numpy(dtype=float)
    theta = traj["theta"].to_numpy(dtype=float)
    v = traj["v"].to_numpy(dtype=float)
    omega = traj["omega"].to_numpy(dtype=float)

    gx = float(traj["goal_x"].iloc[0])
    gy = float(traj["goal_y"].iloc[0])

    # Distance to goal over time
    d = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    d0 = float(d[0])
    dT = float(d[-1])
    dmin = float(d.min())

    # Progress (positive means closer to goal)
    progress = float(d0 - dT)

    # Final heading error relative to direction-to-goal (at final position)
    dx = gx - float(x[-1])
    dy = gy - float(y[-1])
    desired = float(np.arctan2(dy, dx))
    heading_err = angle_wrap(float(theta[-1]) - desired)
    heading_err_abs = float(abs(heading_err))

    # Dynamics stats
    v_max = float(np.max(np.abs(v)))
    omega_max = float(np.max(np.abs(omega)))
    omega_mean = float(np.mean(np.abs(omega)))

    # Rotation jump proxy
    domega = np.diff(omega) if len(omega) > 1 else np.array([0.0])
    domega_max = float(np.max(np.abs(domega))) if domega.size else 0.0

    return {
        "goal_x": gx,
        "goal_y": gy,
        "d0": d0,
        "dT": dT,
        "dmin": dmin,
        "progress": progress,
        "heading_err_abs": heading_err_abs,
        "v_max": v_max,
        "omega_max": omega_max,
        "omega_mean": omega_mean,
        "domega_max": domega_max,
        "T": int(len(traj)),
    }
