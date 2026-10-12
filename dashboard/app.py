"""
SafeTraj dashboard:  streamlit run dashboard/app.py

All numbers come from safetraj.SafeTrajEvaluator, the same code used by the CLI
and the REST API.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.charts import overlay_chart, risk_map_chart, trajectory_chart  # noqa: E402
from safetraj import SafeTrajConfig, SafeTrajEvaluator, TrajectoryError, results_to_frame  # noqa: E402
from safetraj.analysis import fit_failure_tree, risk_map, run_sweep, sample_commands, summarize_sweep  # noqa: E402
from safetraj.config import OPERATIONAL_RANGES  # noqa: E402
from safetraj.report import NOVELTY_NOTE, template_report  # noqa: E402
from safetraj.trajectory import frame_to_trajectories  # noqa: E402

SAMPLE_CSV = os.path.join(ROOT, "examples", "data", "sample_trajectories.csv")
OUTCOME_TEXT = {"success": "strict success", "near_success": "near-success", "failure": "failure"}

st.set_page_config(page_title="SafeTraj", layout="wide")


@st.cache_resource
def get_evaluator(predictor: str) -> SafeTrajEvaluator:
    return SafeTrajEvaluator(SafeTrajConfig(seed=42), predictor=predictor)


@st.cache_data
def cached_sweep(predictor: str, n: int, seed: int, goal: tuple):
    ev = get_evaluator(predictor)
    df = run_sweep(ev.predictor, sample_commands(n, seed=seed), [goal], ev.config)
    return df


def traj_frame(traj) -> pd.DataFrame:
    return traj.to_frame()[["t", "x", "y", "theta", "v", "omega"]]


def command_inputs(key: str, defaults=(1.6, 1.0, 0.0), free: bool = False):
    o_lo, o_hi = OPERATIONAL_RANGES["orientation"]
    v_lo, v_hi = OPERATIONAL_RANGES["v_lin"]
    w_lo, w_hi = OPERATIONAL_RANGES["v_rot"]
    if free:
        th = st.number_input("orientation θ (rad)", value=float(defaults[0]), key=f"{key}_th")
        v = st.number_input("linear velocity v (m/s)", value=float(defaults[1]), key=f"{key}_v")
        w = st.number_input("angular velocity ω (rad/s)", value=float(defaults[2]), key=f"{key}_w")
    else:
        th = st.slider("orientation θ (rad)", o_lo, o_hi, float(defaults[0]), 0.05, key=f"{key}_th")
        v = st.slider("linear velocity v (m/s)", v_lo, v_hi, float(defaults[1]), 0.05, key=f"{key}_v")
        w = st.slider("angular velocity ω (rad/s)", w_lo, w_hi, float(defaults[2]), 0.05, key=f"{key}_w")
    return [th, v, w]


def summary_metrics(res):
    m, c, n = res["metrics"], res["checks"], res["novelty"]
    cols = st.columns(4)
    cols[0].metric("final distance to goal", f"{m['d_final']:.2f} m")
    cols[1].metric("goal outcome", {"success": "success", "near_success": "near-success",
                                    "failure": "failure"}[m["outcome_soft"]],
                   help="strict success < 0.30 m, near-success < 0.50 m, failure otherwise")
    cols[2].metric("rule checks", c["label"], help=f"{c['score']} points from fired rules (0-1 PASS, 2-4 REVIEW, 5+ FAIL)")
    cols[3].metric("input novelty", n["label"] if n else "n/a",
                   help="compared with the reference commands; not a probability of an unsafe trajectory")


def rules_table(res) -> pd.DataFrame:
    rows = [{"rule": r["name"], "fired": "yes" if r["triggered"] else "no", "value": round(r["value"], 3),
             "threshold": r["threshold"], "points": r["points"] if r["triggered"] else 0, "detail": r["message"]}
            for r in res["checks"]["rules"]]
    return pd.DataFrame(rows)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("SafeTraj")
predictor = st.sidebar.selectbox(
    "trajectory predictor", ["goal_seeking", "constant_command"],
    help="Built-in physics stand-ins for a learned predictor. goal_seeking starts with the command and steers to "
         "the goal with limited acceleration; constant_command holds the command (open loop).")
gx = st.sidebar.number_input("goal x (m)", value=1.0, step=0.1)
gy = st.sidebar.number_input("goal y (m)", value=0.0, step=0.1)
goal = (gx, gy)
ev = get_evaluator(predictor)
cfg = ev.config
st.sidebar.caption(f"Robot starts at (0, 0). Strict success: final distance < {cfg.strict_threshold:.2f} m; "
                   f"near-success < {cfg.near_threshold:.2f} m. Horizon {cfg.num_steps} × {cfg.dt} s.")
st.sidebar.caption("Predictors here are transparent physics models used for demonstration, not the proprietary "
                   "REXASI-PRO networks.")

st.title("SafeTraj — trajectory behaviour explorer")
tabs = st.tabs(["Evaluate a command", "Compare two commands", "Batch (CSV)", "Risk map", "About the scores"])

# ── 1. single command ────────────────────────────────────────────────────────
with tabs[0]:
    left, right = st.columns([1, 2])
    with left:
        free = st.toggle("free input (allow values outside the operational range)", value=False)
        cmd = command_inputs("single", free=free)
    res = ev.evaluate(cmd, return_traj=True, goal=goal)
    traj = res.pop("trajectory")
    with right:
        st.altair_chart(trajectory_chart({"predicted": traj_frame(traj)}, goal, cfg.strict_threshold,
                                         cfg.near_threshold), width="content")
    summary_metrics(res)
    st.subheader("Report")
    st.write(template_report(res))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rule checks")
        st.dataframe(rules_table(res), hide_index=True, width="stretch")
    with c2:
        st.subheader("Input novelty")
        n = res["novelty"]
        st.write(f"**{n['label']}** — Mahalanobis {n['mahalanobis']:.2f} (higher than "
                 f"{n['mahalanobis_percentile']:.0f}% of reference inputs), Isolation Forest higher than "
                 f"{n['isolation_forest_percentile']:.0f}%." + (" Outside the reference range." if
                                                                  n["outside_reference_range"] else ""))
        attr = pd.DataFrame({"feature": list(n["attribution"]), "Mahalanobis drop if typical": list(n["attribution"].values())})
        st.bar_chart(attr, x="feature", y="Mahalanobis drop if typical", horizontal=True, height=180)
        st.caption(NOVELTY_NOTE)
    st.download_button("download result (JSON)", json.dumps(res, indent=2, default=float), "safetraj_result.json",
                       "application/json")

# ── 2. comparison ────────────────────────────────────────────────────────────
with tabs[1]:
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Command A**")
        cmd_a = command_inputs("A", (0.0, 0.8, 0.0))
    with cb:
        st.markdown("**Command B**")
        cmd_b = command_inputs("B", (2.8, 2.2, 0.0))
    out = ev.compare(cmd_a, cmd_b, goal=goal)
    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        st.altair_chart(trajectory_chart({"A": traj_frame(out["a"]["trajectory"]),
                                          "B": traj_frame(out["b"]["trajectory"])},
                                         goal, cfg.strict_threshold, cfg.near_threshold, size=340), width="content")
    keys = ["d_final", "d_min", "progress", "path_length", "path_efficiency", "heading_error_final", "total_turning",
            "max_curvature", "omega_max_abs", "domega_max"]
    labels = {"d_final": "final distance (m)", "d_min": "closest approach (m)", "progress": "progress (m)",
              "path_length": "path length (m)", "path_efficiency": "path efficiency", "heading_error_final":
              "final heading error (rad)", "total_turning": "total turning (rad)", "max_curvature": "max curvature (1/m)",
              "omega_max_abs": "max |ω| (rad/s)", "domega_max": "max ω jump per step"}
    rows = [{"metric": "goal outcome", "A": out["a"]["metrics"]["outcome_soft"], "B": out["b"]["metrics"]["outcome_soft"],
             "B − A": ""},
            {"metric": "rule checks", "A": out["a"]["checks"]["label"], "B": out["b"]["checks"]["label"], "B − A": ""},
            {"metric": "input novelty", "A": out["a"]["novelty"]["label"], "B": out["b"]["novelty"]["label"],
             "B − A": ""}]
    rows += [{"metric": labels[k], "A": f"{out['a']['metrics'][k]:.3f}", "B": f"{out['b']['metrics'][k]:.3f}",
              "B − A": f"{out['difference_b_minus_a'][k]:+.3f}"} for k in keys]
    with table_col:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 4)

# ── 3. batch ─────────────────────────────────────────────────────────────────
with tabs[2]:
    st.write("Upload trajectories with columns `traj_id, t, x, y, theta, v, omega, goal_x, goal_y` "
             "(for example the outputs of your own predictor), or use the bundled synthetic sample.")
    upload = st.file_uploader("trajectory CSV", type="csv")
    try:
        frame = pd.read_csv(upload) if upload is not None else pd.read_csv(SAMPLE_CSV)
        trajs = frame_to_trajectories(frame, source=upload.name if upload else "sample_trajectories.csv")
    except (TrajectoryError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the file: {exc}")
        trajs = []
    if trajs:
        if upload is None:
            st.caption("Using examples/data/sample_trajectories.csv — synthetic trajectories from the goal-seeking "
                       "physics model (not real or thesis data).")
        results = ev.evaluate_many(trajs)
        table = results_to_frame(results)
        k = st.columns(3)
        k[0].metric("trajectories", len(table))
        k[1].metric("strict successes", int((table.outcome_strict == "success").sum()))
        k[2].metric("checks PASS / REVIEW / FAIL", " / ".join(str(int((table.check_label == lab).sum()))
                                                              for lab in ("PASS", "REVIEW", "FAIL")))
        st.altair_chart(overlay_chart(frame, table), width="stretch")
        show = ["traj_id", "goal_x", "goal_y", "d_final", "d_min", "progress", "path_length", "outcome_soft",
                "check_label", "rules_triggered"]
        st.dataframe(table[show].round(3), hide_index=True, width="stretch")
        st.download_button("download results (CSV)", table.to_csv(index=False), "safetraj_results.csv", "text/csv")
        st.download_button("download results (JSON)", json.dumps(results, indent=2, default=float),
                           "safetraj_results.json", "application/json")

# ── 4. risk map ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.write("Sample commands uniformly over the operational ranges, predict a trajectory for each and map the "
             "share of strict failures. A shallow decision tree then turns the failure regions into rules "
             "(the workflow of the MSc thesis, applied to the selected predictor).")
    c1, c2 = st.columns(2)
    n = c1.slider("sampled commands", 100, 1000, 400, 50)
    seed = c2.number_input("seed", value=42, step=1)
    df = cached_sweep(predictor, int(n), int(seed), goal)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(risk_map_chart(risk_map(df, "theta", "omega")), width="stretch")
        st.caption("Each cell: share of sampled commands whose trajectory ends ≥ 0.30 m from the goal. "
                   "Cells with fewer than 3 samples are left empty.")
    with right:
        s = summarize_sweep(df).iloc[0]
        st.metric("strict success rate", f"{s.strict_success_rate:.1%}")
        tree = fit_failure_tree(df, features=["theta", "v", "omega"])
        st.write(f"Decision tree (depth ≤ 4): training accuracy {tree.train_accuracy:.2f}, 5-fold cross-validated "
                 f"{tree.cv_accuracy_mean:.2f} ± {tree.cv_accuracy_std:.2f}; always predicting the majority "
                 f"outcome scores {tree.majority_baseline:.2f}.")
        if tree.failure_rules:
            st.markdown("**Failure rules**")
            for r in tree.failure_rules:
                st.markdown(f"- IF {r['conditions']} → failure ({r['samples']} samples, {r['failure_rate']:.0%})")
        else:
            st.write("No leaf of the tree predicts failure.")

# ── 5. about ─────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown(f"""
**Trajectory metrics** use only the predicted trajectory and the goal: distance to the goal at the end
and at the closest point, progress, path length and efficiency, heading error, turning, curvature and
angular-velocity jumps. The goal outcome uses the thesis definitions: strict success below
{cfg.strict_threshold:.2f} m, near-success below {cfg.near_threshold:.2f} m, failure otherwise.

**Rule checks** are five fixed-threshold rules (see `safetraj/rules.py`). Each fired rule adds points:
0–1 points = PASS, 2–4 = REVIEW, 5 or more = FAIL. The thresholds are hand-chosen heuristics, not
learned or validated against real collisions. PASS means that no rule fired, not that a trajectory is safe.

**Input novelty** compares the command with reference commands (here: 800 synthetic commands spread
uniformly over the operational ranges; replace them with your model's training inputs). Mahalanobis
distance and an Isolation Forest are converted to percentiles of the reference set's own scores:
below the {cfg.borderline_percentile:g}th percentile = typical, up to the {cfg.novel_percentile:g}th = borderline,
above = novel. {NOVELTY_NOTE}

**Predictors.** The built-in predictors are simple physics models that stand in for a learned
predictor, so the tools can be tried without proprietary networks. Patterns they produce come from
their equations, not from data.
""")
