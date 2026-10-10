"""
predictor.py — built-in reference trajectory predictors.

They map a command (orientation theta0, linear velocity v0, angular velocity
omega0) and a goal to a Trajectory, like the neural predictors this toolkit is
meant to analyse. They are transparent physics models for demonstrations and
tests, NOT learned models: patterns found with them come from their equations.

ConstantCommandPredictor  exact unicycle motion with the command held constant
                          (open loop; the goal does not influence the path).
GoalSeekingPredictor      starts with the commanded (v0, omega0) and steers
                          toward the goal with bounded acceleration and turn rate.

Any callable with the same `predict(command, goal) -> Trajectory` signature can
be analysed with SafeTrajEvaluator and safetraj.analysis.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .metrics import wrap_angle
from .trajectory import Trajectory


def unicycle_arc(theta0: float, v: float, omega: float, t: np.ndarray, x0: float = 0.0, y0: float = 0.0):
    """Exact unicycle position and heading for constant (v, omega)."""
    theta = theta0 + omega * t
    if abs(omega) < 1e-9:
        x = x0 + v * np.cos(theta0) * t
        y = y0 + v * np.sin(theta0) * t
    else:
        x = x0 + (v / omega) * (np.sin(theta) - np.sin(theta0))
        y = y0 - (v / omega) * (np.cos(theta) - np.cos(theta0))
    return x, y, theta


@dataclass
class ConstantCommandPredictor:
    """Open-loop unicycle: the commanded (v, omega) is held for num_steps * dt seconds."""

    dt: float = 0.1
    num_steps: int = 30
    name: str = "constant_command"

    def predict(self, command: Sequence[float], goal: Optional[Sequence[float]] = None) -> Trajectory:
        theta0, v, omega = (float(c) for c in command)
        t = np.arange(self.num_steps + 1) * self.dt
        x, y, theta = unicycle_arc(theta0, v, omega, t)
        n = len(t)
        return Trajectory(t=t, x=x, y=y, theta=wrap_angle(theta), v=np.full(n, v), omega=np.full(n, omega),
                          goal=tuple(goal) if goal is not None else (0.0, 0.0), command=(theta0, v, omega),
                          source=self.name)


@dataclass
class GoalSeekingPredictor:
    """
    Unicycle that starts with the commanded speeds and then steers toward the
    goal: heading controller omega* = k * heading_error, cruise speed scaled by
    cos(heading_error) and slowed near the goal, with acceleration limits
    a_max (m/s^2) and alpha_max (rad/s^2). Speeds stay inside the operational
    ranges of the thesis models (v in [-1.05, 2.88] m/s, |omega| <= 1.99 rad/s).
    """

    dt: float = 0.1
    num_steps: int = 30
    v_cruise: float = 1.0
    k_heading: float = 2.0
    a_max: float = 1.5
    alpha_max: float = 4.0
    omega_max: float = 1.99
    v_limits: Tuple[float, float] = (-1.05, 2.88)
    slow_radius: float = 0.5
    name: str = "goal_seeking"

    def predict(self, command: Sequence[float], goal: Optional[Sequence[float]] = None) -> Trajectory:
        if goal is None:
            raise ValueError("GoalSeekingPredictor needs a goal")
        theta, v, omega = (float(c) for c in command)
        gx, gy = (float(g) for g in goal)
        x = y = 0.0
        rows = [(0.0, x, y, theta, v, omega)]
        for k in range(1, self.num_steps + 1):
            dist = np.hypot(gx - x, gy - y)
            err = float(wrap_angle(np.arctan2(gy - y, gx - x) - theta))
            omega_des = float(np.clip(self.k_heading * err, -self.omega_max, self.omega_max))
            v_des = self.v_cruise * max(0.0, np.cos(err)) * min(1.0, dist / self.slow_radius)
            v += float(np.clip(v_des - v, -self.a_max * self.dt, self.a_max * self.dt))
            v = float(np.clip(v, *self.v_limits))
            omega += float(np.clip(omega_des - omega, -self.alpha_max * self.dt, self.alpha_max * self.dt))
            omega = float(np.clip(omega, -self.omega_max, self.omega_max))
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta = float(wrap_angle(theta + omega * self.dt))
            rows.append((k * self.dt, x, y, theta, v, omega))
        arr = np.asarray(rows)
        return Trajectory(t=arr[:, 0], x=arr[:, 1], y=arr[:, 2], theta=arr[:, 3], v=arr[:, 4], omega=arr[:, 5],
                          goal=(gx, gy), command=tuple(float(c) for c in command), source=self.name)


PREDICTORS = {"goal_seeking": GoalSeekingPredictor, "constant_command": ConstantCommandPredictor}


def make_predictor(name: str, dt: float = 0.1, num_steps: int = 30):
    if name not in PREDICTORS:
        raise ValueError(f"unknown predictor '{name}'. Available: {', '.join(PREDICTORS)}")
    return PREDICTORS[name](dt=dt, num_steps=num_steps)


@dataclass
class DummyTrajectoryPredictor:
    """
    Version 1 interface kept for compatibility: predict([orientation, v, omega])
    returns a (num_steps, 3) array of [x, y, theta] over t in [0, horizon].

    Fixed in version 2: the path now follows the unicycle model (it curves
    with omega). Version 1 drew a straight line along the initial heading and
    only rotated theta, so x and y ignored omega.
    """

    horizon: float = 1.5
    num_steps: int = 20

    def predict(self, x: np.ndarray) -> np.ndarray:
        angle, v_lin, v_rot = (float(c) for c in x)
        t = np.linspace(0.0, self.horizon, self.num_steps)
        xs, ys, theta = unicycle_arc(angle, v_lin, v_rot, t)
        return np.column_stack([xs, ys, theta])
