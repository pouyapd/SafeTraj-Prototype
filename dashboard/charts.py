"""Altair chart builders for the dashboard (interactive: zoom, pan, tooltips)."""

from typing import Dict, List, Sequence

import altair as alt
import numpy as np
import pandas as pd

OUTCOME_COLORS = {"success": "#0ca30c", "near_success": "#eda100", "failure": "#d03b3b"}
SERIES = ["#2a78d6", "#eb6834"]


def _circle(center: Sequence[float], r: float, label: str, n: int = 80) -> pd.DataFrame:
    a = np.linspace(0, 2 * np.pi, n)
    return pd.DataFrame({"x": center[0] + r * np.cos(a), "y": center[1] + r * np.sin(a), "order": range(n),
                         "region": label})


def _square_domain(frames: List[pd.DataFrame], pad: float = 0.4):
    allx = pd.concat([f["x"] for f in frames])
    ally = pd.concat([f["y"] for f in frames])
    cx, cy = (allx.min() + allx.max()) / 2, (ally.min() + ally.max()) / 2
    half = max(allx.max() - allx.min(), ally.max() - ally.min()) / 2 + pad
    return [cx - half, cx + half], [cy - half, cy + half]


def trajectory_chart(trajs: Dict[str, pd.DataFrame], goal: Sequence[float], strict: float, near: float,
                     size: int = 360) -> alt.Chart:
    """trajs: name -> DataFrame(t, x, y, theta, v, omega). Square chart with equal x/y scales."""
    success_label, near_label = f"success region < {strict:.2f} m", f"near-success < {near:.2f} m"
    rings = pd.concat([_circle(goal, strict, success_label), _circle(goal, near, near_label)])
    frames = list(trajs.values()) + [rings]
    xd, yd = _square_domain(frames)
    layers = [
        alt.Chart(rings).mark_line(strokeWidth=1.4).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=xd, nice=False), title="x (m)"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=yd, nice=False), title="y (m)"),
            order="order:O", detail="region:N",
            color=alt.Color("region:N", scale=alt.Scale(domain=[success_label, near_label],
                                                        range=[OUTCOME_COLORS["success"], OUTCOME_COLORS["near_success"]]),
                            legend=alt.Legend(title="goal", orient="bottom")),
            strokeDash=alt.StrokeDash("region:N", scale=alt.Scale(domain=[success_label, near_label],
                                                                  range=[[1, 0], [5, 3]]), legend=None)),
        alt.Chart(pd.DataFrame({"x": [goal[0]], "y": [goal[1]]})).mark_point(
            shape="diamond", size=120, filled=True, color=OUTCOME_COLORS["success"]).encode(x="x:Q", y="y:Q"),
    ]
    data = pd.concat([df.assign(trajectory=name) for name, df in trajs.items()], ignore_index=True)
    names = list(trajs)
    color = alt.Color("trajectory:N", scale=alt.Scale(domain=names, range=SERIES[:len(names)]),
                      legend=alt.Legend(title=None, orient="top-left", fillColor="#fcfcfb", padding=6,
                                         strokeColor="#e1e0d9") if len(names) > 1 else None)
    tooltip = [alt.Tooltip("trajectory:N")] + [alt.Tooltip(c, format=".2f") for c in
                                               ("t", "x", "y", "theta", "v", "omega")]
    layers.append(alt.Chart(data).mark_line(strokeWidth=2.5).encode(x="x:Q", y="y:Q", order="t:Q",
                                                                     detail="trajectory:N", color=color))
    point_color = alt.Color("trajectory:N", scale=alt.Scale(domain=names, range=SERIES[:len(names)]), legend=None)
    layers.append(alt.Chart(data).mark_circle(size=28, opacity=0.8).encode(x="x:Q", y="y:Q", color=point_color,
                                                                           tooltip=tooltip))
    starts = data.groupby("trajectory", sort=False).head(1)
    layers.append(alt.Chart(starts).mark_point(size=90, filled=True, color="#0b0b0b").encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("trajectory:N"), alt.Tooltip("t:Q", title="start t")]))
    # autosize "pad": width/height describe the plot area itself, so equal domains give round goal rings.
    return (alt.layer(*layers).resolve_scale(color="independent", strokeDash="independent")
            .properties(width=size, height=size, autosize=alt.AutoSizeParams(type="pad")).interactive())


def overlay_chart(frame: pd.DataFrame, results: pd.DataFrame, height: int = 420) -> alt.Chart:
    """All trajectories coloured by soft outcome. frame: trajectory rows; results: one row per traj_id."""
    d = frame.merge(results[["traj_id", "outcome_soft", "check_label", "d_final"]], on="traj_id")
    goals = d[["goal_x", "goal_y"]].drop_duplicates().rename(columns={"goal_x": "x", "goal_y": "y"})
    color = alt.Color("outcome_soft:N", title="goal outcome",
                      scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values())))
    lines = alt.Chart(d).mark_line(strokeWidth=1.6, opacity=0.85).encode(
        x=alt.X("x:Q", title="x (m)"), y=alt.Y("y:Q", title="y (m)"), detail="traj_id:N", order="t:Q", color=color,
        tooltip=["traj_id:N", "outcome_soft:N", "check_label:N", alt.Tooltip("d_final:Q", format=".2f")])
    goal_pts = alt.Chart(goals).mark_point(shape="diamond", size=110, filled=True, color="#0b0b0b").encode(
        x="x:Q", y="y:Q")
    return alt.layer(lines, goal_pts).properties(height=height).interactive()


def risk_map_chart(rm: Dict, height: int = 380) -> alt.Chart:
    xe, ye, rate, counts = rm["x_edges"], rm["y_edges"], rm["rate"], rm["counts"]
    rows = []
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            rows.append({"x0": xe[j], "x1": xe[j + 1], "y0": ye[i], "y1": ye[i + 1],
                         "failure_rate": None if np.isnan(rate[i, j]) else float(rate[i, j]),
                         "samples": int(counts[i, j])})
    df = pd.DataFrame(rows)
    names = {"theta": "orientation θ (rad)", "omega": "angular velocity ω (rad/s)", "v": "linear velocity v (m/s)"}
    base = alt.Chart(df).encode(
        x=alt.X("x0:Q", title=names.get(rm["x"], rm["x"])), x2="x1:Q",
        y=alt.Y("y0:Q", title=names.get(rm["y"], rm["y"])), y2="y1:Q",
        tooltip=[alt.Tooltip("x0:Q", format=".2f", title="bin from"), alt.Tooltip("x1:Q", format=".2f", title="bin to"),
                 alt.Tooltip("failure_rate:Q", format=".0%"), "samples:Q"])
    cells = base.mark_rect(stroke="#fcfcfb", strokeWidth=1).encode(
        color=alt.Color("failure_rate:Q", title="strict failure rate", scale=alt.Scale(scheme="oranges", domain=[0, 1])))
    return cells.properties(height=height)
