"""
cli.py - command-line interface:  python -m safetraj <command> [options]

Commands:
  evaluate     one command (orientation, v, omega) toward a goal
  compare      two commands toward the same goal
  batch        every trajectory in a CSV file -> CSV, JSON, Markdown report, figures
  sweep        many sampled commands -> risk map and decision-tree failure rules
  sample-data  write the synthetic example trajectories used in the README
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional



def _goal(values: Optional[List[float]]):
    return tuple(values) if values else None


def _evaluator(args):
    from .config import SafeTrajConfig
    from .evaluator import SafeTrajEvaluator
    return SafeTrajEvaluator(SafeTrajConfig(), predictor=args.predictor)


def cmd_evaluate(args) -> int:
    from .report import explain
    ev = _evaluator(args)
    res = ev.evaluate([args.orientation, args.v, args.omega], return_traj=True, goal=_goal(args.goal))
    traj = res.pop("trajectory")
    rep = explain(res, use_llm=args.llm is not None, model=args.llm or "mistral:latest")
    if args.json:
        print(json.dumps(dict(res, report=rep), indent=2, default=float))
    else:
        print(rep["text"])
        if rep["llm_note"]:
            print(f"[{rep['llm_note']}]")
    if args.plot:
        from .plotting import plot_trajectory
        plot_trajectory(traj, result=res, strict=ev.config.strict_threshold,
                        near=ev.config.near_threshold).savefig(args.plot)
        print(f"saved {args.plot}")
    return 0


def cmd_compare(args) -> int:
    ev = _evaluator(args)
    out = ev.compare(args.a, args.b, goal=_goal(args.goal))
    for key in ("a", "b"):
        r = out[key]
        print(f"{key.upper()}: {r['input']} -> d_final {r['metrics']['d_final']:.2f} m, "
              f"{r['metrics']['outcome_soft']}, checks {r['checks']['label']}, novelty {r['novelty']['label']}")
    d = out["difference_b_minus_a"]
    print("B - A: " + ", ".join(f"{k} {d[k]:+.3f}" for k in ("d_final", "progress", "path_length", "total_turning")))
    if args.plot:
        from .plotting import plot_comparison
        plot_comparison(out["a"]["trajectory"], out["b"]["trajectory"],
                        labels=(f"A {tuple(args.a)}", f"B {tuple(args.b)}")).savefig(args.plot)
        print(f"saved {args.plot}")
    return 0


def cmd_batch(args) -> int:
    from .evaluator import results_to_frame
    from .report import explain, write_batch_report
    from .trajectory import load_trajectories
    ev = _evaluator(args)
    trajs = load_trajectories(args.input)
    results = ev.evaluate_many(trajs)
    table = results_to_frame(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    reports = [dict(explain(r, use_llm=args.llm is not None, model=args.llm or "mistral:latest"),
                    traj_id=r["traj_id"]) for r in results]
    figures = {}
    if not args.no_figures:
        from .plotting import plot_distance_histogram, plot_overlay
        plot_overlay(trajs, results, strict=ev.config.strict_threshold,
                     near=ev.config.near_threshold).savefig(out / "overlay.png")
        plot_distance_histogram(table["d_final"], ev.config.strict_threshold,
                                ev.config.near_threshold).savefig(out / "final_distance.png")
        figures = {"All trajectories coloured by goal outcome": "overlay.png",
                   "Final distance to goal with the 0.30 m and 0.50 m thresholds": "final_distance.png"}
    report = write_batch_report(table, reports, out, figures=figures, source=Path(args.input).name)
    counts = table["check_label"].value_counts().to_dict()
    print(f"{len(table)} trajectories: checks {counts}; outcomes {table['outcome_soft'].value_counts().to_dict()}")
    print(f"wrote {out / 'results.csv'}, {out / 'results.json'}, {report}")
    return 0


def cmd_sweep(args) -> int:
    from .analysis import fit_failure_tree, risk_map, run_sweep, sample_commands, summarize_sweep
    from .config import SafeTrajConfig
    from .plotting import plot_risk_map
    from .predictor import make_predictor
    cfg = SafeTrajConfig()
    predictor = make_predictor(args.predictor, cfg.dt, cfg.num_steps)
    goals = [tuple(float(v) for v in g.split(",")) for g in args.goals]
    commands = sample_commands(args.n, seed=args.seed)
    df = run_sweep(predictor, commands, goals, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep.csv", index=False)
    summary = summarize_sweep(df)
    summary.to_csv(out / "summary_by_goal.csv", index=False)
    tree = fit_failure_tree(df)
    rules = "\n".join(f"IF {r['conditions']} THEN failure  ({r['samples']} samples, "
                      f"{r['failure_rate']:.0%} failures)" for r in tree.failure_rules)
    rules = rules or "(no leaf predicts failure)"
    (out / "tree_rules.txt").write_text(tree.rules_text + "\nFailure rules:\n" + rules + "\n",
                                        encoding="utf-8")
    with open(out / "tree_report.json", "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2)
    for g in goals:
        sub = df[(df.goal_x == g[0]) & (df.goal_y == g[1])]
        rm = risk_map(sub, "theta", "omega")
        fig = plot_risk_map(rm, title=f"{args.predictor} predictor, goal ({g[0]:g}, {g[1]:g})",
                            note=f"{len(sub)} sampled commands (seed {args.seed}); hatched: fewer than 3 samples.")
        fig.savefig(out / f"risk_map_goal_{g[0]:g}_{g[1]:g}.png")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"decision tree: train accuracy {tree.train_accuracy:.3f}, 5-fold CV {tree.cv_accuracy_mean:.3f} "
          f"(+/- {tree.cv_accuracy_std:.3f}), majority baseline {tree.majority_baseline:.3f}")
    print("failure rules:")
    for r in tree.failure_rules:
        print(f"  IF {r['conditions']} THEN failure ({r['samples']} samples, {r['failure_rate']:.0%} failures)")
    print(f"wrote results to {out}")
    return 0


def cmd_sample_data(args) -> int:
    from .analysis import sample_commands
    from .predictor import make_predictor
    from .trajectory import trajectories_to_frame
    predictor = make_predictor(args.predictor)
    goals = [(0.5, 0.5), (1.0, 0.0), (1.5, -0.5)]
    commands = sample_commands(args.n, seed=args.seed)
    trajs = []
    for i, cmd in enumerate(commands):
        traj = predictor.predict(cmd, goals[i % len(goals)])
        traj.traj_id = i + 1
        trajs.append(traj)
    df = trajectories_to_frame(trajs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(args.out, index=False)
    print(f"wrote {len(trajs)} synthetic trajectories ({args.predictor} predictor, seed {args.seed}) to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m safetraj", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    predictors = ["goal_seeking", "constant_command"]

    e = sub.add_parser("evaluate", help="evaluate one command")
    e.add_argument("--orientation", type=float, required=True, help="initial orientation (rad)")
    e.add_argument("--v", type=float, required=True, help="linear velocity (m/s)")
    e.add_argument("--omega", type=float, required=True, help="angular velocity (rad/s)")
    e.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"), help="goal (default 1.0 0.0)")
    e.add_argument("--predictor", choices=predictors, default="goal_seeking")
    e.add_argument("--json", action="store_true", help="print the full result as JSON")
    e.add_argument("--llm", metavar="MODEL", help="phrase the report with a local Ollama model (optional)")
    e.add_argument("--plot", metavar="PNG", help="save a trajectory figure")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("compare", help="compare two commands")
    c.add_argument("--a", type=float, nargs=3, required=True, metavar=("THETA", "V", "OMEGA"))
    c.add_argument("--b", type=float, nargs=3, required=True, metavar=("THETA", "V", "OMEGA"))
    c.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    c.add_argument("--predictor", choices=predictors, default="goal_seeking")
    c.add_argument("--plot", metavar="PNG")
    c.set_defaults(func=cmd_compare)

    b = sub.add_parser("batch", help="evaluate every trajectory in a CSV file")
    b.add_argument("--input", required=True, help="CSV with traj_id,t,x,y,theta,v,omega,goal_x,goal_y")
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--predictor", choices=predictors, default="goal_seeking", help=argparse.SUPPRESS)
    b.add_argument("--llm", metavar="MODEL", help="optional Ollama model for the explanations")
    b.add_argument("--no-figures", action="store_true")
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("sweep", help="risk map and failure rules over sampled commands")
    s.add_argument("--predictor", choices=predictors, default="goal_seeking")
    s.add_argument("--n", type=int, default=200, help="sampled commands (reused for every goal)")
    s.add_argument("--goals", nargs="+", default=["0.5,0.5", "1.0,0.0", "1.5,-0.5"], metavar="X,Y")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sweep)

    d = sub.add_parser("sample-data", help="write synthetic example trajectories")
    d.add_argument("--out", default="examples/data/sample_trajectories.csv")
    d.add_argument("--n", type=int, default=24)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--predictor", choices=predictors, default="goal_seeking")
    d.set_defaults(func=cmd_sample_data)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from .trajectory import TrajectoryError
    try:
        return args.func(args)
    except (TrajectoryError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
