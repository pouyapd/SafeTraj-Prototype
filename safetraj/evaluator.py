"""
evaluator.py — one entry point used by the Python API, the CLI, the REST API
and the dashboard, so all four give identical results.

For a command (orientation, v_lin, v_rot) and a goal it:
  1. predicts a trajectory with the chosen predictor,
  2. computes trajectory metrics and the goal outcome (strict / soft),
  3. runs the rule-based checks,
  4. scores the input's novelty against the reference commands.
A trajectory supplied directly (for example the output of your own model) goes
through steps 2-4; step 4 runs only if its initial command is known.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SafeTrajConfig
from .data import generate_training_data
from .metrics import compute_metrics
from .ood import NoveltyScorer
from .predictor import make_predictor
from .rules import run_checks
from .trajectory import Trajectory


class SafeTrajEvaluator:
    """
    Args:
        config: thresholds and settings (default SafeTrajConfig()).
        reference_data: (n, 3) array of reference commands for the novelty
            scores; default: synthetic commands uniform over the operational ranges.
        predictor: object with predict(command, goal) -> Trajectory, or the
            name of a built-in predictor ("goal_seeking", "constant_command").
    """

    def __init__(self, config: Optional[SafeTrajConfig] = None, reference_data: Optional[np.ndarray] = None,
                 predictor: Any = "goal_seeking"):
        self.config = config or SafeTrajConfig()
        c = self.config
        self.training_data = (np.asarray(reference_data, dtype=float) if reference_data is not None
                              else generate_training_data(c.n_train_samples, c.seed, c.operational_ranges))
        self.novelty = NoveltyScorer(self.training_data, contamination=c.contamination, seed=c.seed,
                                     borderline_percentile=c.borderline_percentile,
                                     novel_percentile=c.novel_percentile)
        self.predictor = make_predictor(predictor, c.dt, c.num_steps) if isinstance(predictor, str) else predictor

    # Version 1 attribute names (kept for compatibility).
    @property
    def maha(self):
        return self.novelty.maha

    @property
    def iso(self):
        return self.novelty.iso

    # ── single command ───────────────────────────────────────────────────────

    def predict(self, command: Sequence[float], goal: Optional[Sequence[float]] = None) -> Trajectory:
        return self.predictor.predict(command, tuple(goal) if goal is not None else self.config.default_goal)

    def get_trajectory(self, x_list: Sequence[float], goal: Optional[Sequence[float]] = None) -> np.ndarray:
        """Version 1 helper: (T, 3) array of [x, y, theta]."""
        traj = self.predict(x_list, goal)
        return np.column_stack([traj.x, traj.y, traj.theta])

    def evaluate(self, x_list: Sequence[float], return_traj: bool = False,
                 goal: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """
        Evaluate one command [orientation, v_lin, v_rot] toward `goal`
        (default config.default_goal = (1.0, 0.0), the thesis reference goal).
        """
        command = np.asarray(x_list, dtype=float)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("command must be three finite numbers [orientation, v_lin, v_rot]")
        traj = self.predict(command, goal)
        result = self.evaluate_trajectory(traj)
        result["input"] = {"orientation": float(command[0]), "v_lin": float(command[1]), "v_rot": float(command[2])}
        if return_traj:
            result["trajectory"] = traj
        return result

    # ── trajectory supplied by the user ─────────────────────────────────────

    def evaluate_trajectory(self, traj: Trajectory) -> Dict[str, Any]:
        c = self.config
        metrics = compute_metrics(traj, c.strict_threshold, c.near_threshold)
        checks = run_checks(metrics, c)
        result: Dict[str, Any] = {
            "traj_id": traj.traj_id,
            "goal": list(traj.goal),
            "predictor": traj.source or getattr(self.predictor, "name", "user"),
            "metrics": metrics.to_dict(),
            "checks": checks.to_dict(),
            "novelty": None,
        }
        if traj.command is not None:
            nov = self.novelty.score(traj.command)
            result["novelty"] = nov.to_dict()
            # Deprecated version 1 keys (aliases of novelty.*), kept for old callers.
            result.update(mahalanobis_score=nov.mahalanobis, isolation_forest_score=nov.isolation_forest,
                          risk_score=nov.score, risk_label=nov.label)
        return result

    # ── many trajectories / commands ────────────────────────────────────────

    def evaluate_many(self, trajs: Iterable[Trajectory]) -> List[Dict[str, Any]]:
        return [self.evaluate_trajectory(t) for t in trajs]

    def evaluate_commands(self, commands: Sequence[Sequence[float]],
                          goal: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
        return [self.evaluate(cmd, goal=goal) for cmd in commands]

    def compare(self, command_a: Sequence[float], command_b: Sequence[float],
                goal: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """Evaluate two commands toward the same goal and report metric differences (b - a)."""
        a = self.evaluate(command_a, return_traj=True, goal=goal)
        b = self.evaluate(command_b, return_traj=True, goal=goal)
        diff = {k: b["metrics"][k] - a["metrics"][k] for k, v in a["metrics"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return {"a": a, "b": b, "difference_b_minus_a": diff}


def results_to_frame(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten evaluation results to one row per trajectory (for CSV export)."""
    rows = []
    for r in results:
        row = {"traj_id": r["traj_id"], "goal_x": r["goal"][0], "goal_y": r["goal"][1], "predictor": r["predictor"]}
        if "input" in r:
            row.update({f"input_{k}": v for k, v in r["input"].items()})
        row.update(r["metrics"])
        row.update(check_label=r["checks"]["label"], check_score=r["checks"]["score"],
                   rules_triggered=";".join(r["checks"]["triggered"]))
        nov = r.get("novelty")
        if nov:
            row.update(novelty_label=nov["label"], novelty_score=nov["score"], mahalanobis=nov["mahalanobis"],
                       isolation_forest=nov["isolation_forest"],
                       outside_reference_range=nov["outside_reference_range"])
        rows.append(row)
    return pd.DataFrame(rows)
