"""
SafeTraj: behaviour analysis of trajectory predictors.

Main pieces:
    SafeTrajEvaluator    command or trajectory -> metrics, rule checks, input novelty
    safetraj.analysis    sweeps over many commands, risk maps, decision-tree failure rules
    safetraj.report      template reports (optional local-LLM phrasing with fallback)
    safetraj.plotting    figures
"""

from .config import OPERATIONAL_RANGES, SafeTrajConfig
from .evaluator import SafeTrajEvaluator, results_to_frame
from .plotting import plot_attribution, plot_comparison, plot_feature_importance, plot_trajectory
from .predictor import ConstantCommandPredictor, DummyTrajectoryPredictor, GoalSeekingPredictor
from .trajectory import Trajectory, TrajectoryError, load_trajectories

__version__ = "2.0.0"

__all__ = [
    "OPERATIONAL_RANGES",
    "SafeTrajConfig",
    "SafeTrajEvaluator",
    "results_to_frame",
    "plot_attribution",
    "plot_comparison",
    "plot_feature_importance",
    "plot_trajectory",
    "ConstantCommandPredictor",
    "DummyTrajectoryPredictor",
    "GoalSeekingPredictor",
    "Trajectory",
    "TrajectoryError",
    "load_trajectories",
]
