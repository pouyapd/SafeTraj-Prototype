"""
Guided end-to-end example (about 30 s):  python examples/end_to_end.py

1. evaluate one command and print the report
2. compare two commands toward the same goal
3. evaluate a CSV of trajectories (the bundled synthetic sample) and export CSV/JSON
4. sweep 400 commands, build a risk map and extract decision-tree failure rules

Outputs go to examples/output/ (not tracked by git).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from safetraj import SafeTrajConfig, SafeTrajEvaluator, load_trajectories, results_to_frame  # noqa: E402
from safetraj.analysis import fit_failure_tree, risk_map, run_sweep, sample_commands  # noqa: E402
from safetraj.plotting import plot_comparison, plot_overlay, plot_risk_map  # noqa: E402
from safetraj.report import template_report  # noqa: E402

OUT = ROOT / "examples" / "output"
OUT.mkdir(parents=True, exist_ok=True)
ev = SafeTrajEvaluator(SafeTrajConfig(seed=42), predictor="goal_seeking")
goal = (1.0, 0.0)

print("1) One command: facing away from the goal at high speed")
res = ev.evaluate([2.9, 2.4, 0.0], goal=goal)
print("  ", template_report(res), "\n")

print("2) Compare: the same speed, facing the goal (A) or away from it (B)")
cmp = ev.compare([0.0, 2.4, 0.0], [2.9, 2.4, 0.0], goal=goal)
for k in ("a", "b"):
    m = cmp[k]["metrics"]
    print(f"   {k.upper()}: final distance {m['d_final']:.2f} m -> {m['outcome_soft']}, checks {cmp[k]['checks']['label']}")
plot_comparison(cmp["a"]["trajectory"], cmp["b"]["trajectory"], labels=("A: facing the goal", "B: facing away"))\
    .savefig(OUT / "comparison.png")

print("\n3) Batch: every trajectory in examples/data/sample_trajectories.csv (synthetic)")
trajs = load_trajectories(ROOT / "examples" / "data" / "sample_trajectories.csv")
results = ev.evaluate_many(trajs)
table = results_to_frame(results)
table.to_csv(OUT / "batch_results.csv", index=False)
(OUT / "batch_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print(f"   {len(table)} trajectories; outcomes {table.outcome_soft.value_counts().to_dict()}; "
      f"checks {table.check_label.value_counts().to_dict()}")
plot_overlay(trajs, results).savefig(OUT / "batch_overlay.png")

print("\n4) Sweep: 400 commands toward the goal, risk map and failure rules")
df = run_sweep(ev.predictor, sample_commands(400, seed=42), [goal], ev.config)
plot_risk_map(risk_map(df, "theta", "omega"), title="Goal-seeking predictor, goal (1, 0)").savefig(OUT / "risk_map.png")
tree = fit_failure_tree(df, features=["theta", "v", "omega"])
print(f"   strict success rate {1 - df.failure.mean():.1%}; tree CV accuracy {tree.cv_accuracy_mean:.2f} "
      f"vs majority baseline {tree.majority_baseline:.2f}")
for r in tree.failure_rules:
    print(f"   IF {r['conditions']} -> failure ({r['samples']} samples, {r['failure_rate']:.0%})")
print(f"\nFigures and tables written to {OUT}")
