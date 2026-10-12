"""
Minimal example: evaluate one command and show the trajectory.

    python examples/demo_basic.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from safetraj import SafeTrajConfig, SafeTrajEvaluator, plot_trajectory  # noqa: E402
from safetraj.report import template_report  # noqa: E402


def main() -> None:
    evaluator = SafeTrajEvaluator(SafeTrajConfig(seed=42))
    x = [0.5, 1.2, 0.3]  # [orientation (rad), v_lin (m/s), v_rot (rad/s)]
    res = evaluator.evaluate(x, return_traj=True, goal=(1.0, 0.0))

    print("Final distance to goal:", round(res["metrics"]["d_final"], 3), "m")
    print("Goal outcome:", res["metrics"]["outcome_soft"])
    print("Rule checks:", res["checks"]["label"], res["checks"]["triggered"])
    print("Input novelty:", res["novelty"]["label"], f"(score {res['novelty']['score']:.2f})")
    print(template_report(res))

    plot_trajectory(res["trajectory"], result=res)
    plt.show()


if __name__ == "__main__":
    main()
