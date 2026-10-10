"""
Regenerate the README figures in docs/figures/ from the bundled examples:

    python examples/make_figures.py

All inputs are synthetic (built-in physics predictor, fixed seeds); every
figure says so in its title or footnote.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from safetraj import SafeTrajConfig, SafeTrajEvaluator, load_trajectories, results_to_frame  # noqa: E402
from safetraj.analysis import fit_failure_tree, risk_map, run_sweep, sample_commands  # noqa: E402
from safetraj.plotting import (MUTED, OUTCOME, OUTCOME_LABEL, apply_style, draw_goal, draw_trajectory,  # noqa: E402
                               plot_distance_histogram, plot_novelty_map, plot_overlay, plot_risk_map, SERIES)

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
ev = SafeTrajEvaluator(SafeTrajConfig(seed=42), predictor="goal_seeking")
cfg = ev.config
goal = (1.0, 0.0)


def note(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=8, color=MUTED, ha="left", va="top")


# 1. Same goal, four commands (one panel each) ---------------------------------
apply_style()
commands = [((0.0, 0.8, 0.0), "facing the goal"), ((1.6, 1.0, 0.0), "goal to the right"),
            ((2.9, 0.8, 0.0), "facing away, slow"), ((2.9, 2.4, 0.0), "facing away, fast")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
for ax, (cmd, label) in zip(axes, commands):
    res = ev.evaluate(cmd, return_traj=True, goal=goal)
    m = res["metrics"]
    draw_goal(ax, goal, cfg.strict_threshold, cfg.near_threshold)
    draw_trajectory(ax, res["trajectory"], color=SERIES[0], end_color=OUTCOME[m["outcome_soft"]])
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 2.0)
    ax.set_ylim(-1.7, 1.7)
    ax.set_title(f"{label}\nθ={cmd[0]:.1f} rad, v={cmd[1]:.1f} m/s → {OUTCOME_LABEL[m['outcome_soft']]}, "
                 f"{m['d_final']:.2f} m", fontsize=9.5)
    ax.text(0.02, 0.03, f"checks {res['checks']['label']} · novelty {res['novelty']['label']}",
            transform=ax.transAxes, fontsize=8, color="#52514e")
    ax.set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
fig.tight_layout()
note(fig, "Goal-seeking physics predictor (built-in stand-in, not a learned model); start at the origin, "
          "3 s horizon, goal (1, 0). Green ring: strict success (< 0.30 m); dashed: near-success (< 0.50 m).")
fig.savefig(OUT / "four_commands.png")
plt.close(fig)

# 2. Batch overlay + final-distance histogram of the bundled sample ------------
trajs = load_trajectories(ROOT / "examples" / "data" / "sample_trajectories.csv")
results = ev.evaluate_many(trajs)
fig = plot_overlay(trajs, results, title="Batch evaluation of 24 sample trajectories")
note(fig, "examples/data/sample_trajectories.csv: synthetic trajectories (goal-seeking predictor, seed 7), "
          "three goals.")
fig.savefig(OUT / "batch_overlay.png")
plt.close(fig)
table = results_to_frame(results)
fig = plot_distance_histogram(table["d_final"], cfg.strict_threshold, cfg.near_threshold,
                              title="Final distance to goal (24 sample trajectories)")
fig.savefig(OUT / "batch_final_distance.png")
plt.close(fig)

# 3. Risk map from a 400-command sweep ------------------------------------------
df = run_sweep(ev.predictor, sample_commands(1000, seed=42), [goal], cfg)
tree = fit_failure_tree(df, features=["theta", "v", "omega"])
fig = plot_risk_map(risk_map(df, "theta", "omega"),
                    title="Risk map: goal-seeking predictor, goal (1, 0)",
                    note=f"1000 commands uniform over the operational ranges (seed 42); strict success rate "
                         f"{1 - df.failure.mean():.1%}. Hatched: < 3 samples. Synthetic predictor.")
fig.savefig(OUT / "risk_map_goal_seeking.png")
plt.close(fig)
(OUT / "risk_map_goal_seeking_rules.txt").write_text(
    "\n".join(f"IF {r['conditions']} THEN failure ({r['samples']} samples, {r['failure_rate']:.0%} failures)"
              for r in tree.failure_rules)
    + f"\n\ntrain accuracy {tree.train_accuracy:.3f}; 5-fold CV {tree.cv_accuracy_mean:.3f} +/- "
      f"{tree.cv_accuracy_std:.3f}; majority baseline {tree.majority_baseline:.3f}\n", encoding="utf-8")

# 4. What the novelty score measures --------------------------------------------
fig = plot_novelty_map(ev.novelty, v_rot=0.0,
                       title="Input novelty over (θ, v) at ω = 0 — reference: 800 uniform commands")
note(fig, "Percentile of the reference set's own scores (max of Mahalanobis and Isolation Forest); inputs "
          "outside the reference range get 100. Novelty is not a probability of an unsafe trajectory.", y=-0.12)
fig.savefig(OUT / "novelty_map.png")
plt.close(fig)

print("figures written to", OUT)
print(open(OUT / "risk_map_goal_seeking_rules.txt", encoding="utf-8").read())
