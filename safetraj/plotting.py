"""
plotting.py — matplotlib figures with one visual language (colour-blind-checked
palette, recessive grid, explicit units). Every function returns a Figure.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from .trajectory import Trajectory

SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots, fixed order
OUTCOME = {"success": "#0ca30c", "near_success": "#eda100", "failure": "#d03b3b"}
OUTCOME_LABEL = {"success": "strict success", "near_success": "near-success", "failure": "failure"}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def _as_xyth(traj: Union[Trajectory, np.ndarray]):
    if isinstance(traj, Trajectory):
        return traj.x, traj.y, traj.theta
    arr = np.asarray(traj)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def draw_goal(ax, goal: Sequence[float], strict: float = 0.30, near: float = 0.50) -> None:
    ax.add_patch(Circle(goal, near, facecolor="none", edgecolor=OUTCOME["near_success"], linewidth=1.0,
                        linestyle=(0, (3, 2)), zorder=1))
    ax.add_patch(Circle(goal, strict, facecolor="#e3f4e3", edgecolor=OUTCOME["success"], linewidth=1.0, zorder=1))
    ax.plot(*goal, marker="*", markersize=12, color=OUTCOME["success"], markeredgecolor=SURFACE, zorder=5)


def draw_trajectory(ax, traj: Union[Trajectory, np.ndarray], color: str = SERIES[0], label: Optional[str] = None,
                    arrows: bool = True, end_color: Optional[str] = None) -> None:
    x, y, th = _as_xyth(traj)
    ax.plot(x, y, color=color, linewidth=2.0, label=label, zorder=3)
    ax.plot(x[0], y[0], marker="o", markersize=7, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=6)
    ax.plot(x[-1], y[-1], marker="o", markersize=7, color=end_color or color, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=6)
    if arrows and len(x) > 2:
        step = max(1, len(x) // 6)
        for i in range(0, len(x) - 1, step):
            ax.annotate("", xy=(x[i] + 0.12 * np.cos(th[i]), y[i] + 0.12 * np.sin(th[i])), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=8), zorder=4)


def _frame(ax, trajs: Iterable, goals: Iterable, pad: float = 0.4) -> None:
    pts = [np.column_stack(_as_xyth(t)[:2]) for t in trajs] + [np.atleast_2d(g) for g in goals]
    allp = np.vstack(pts)
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    span = max(hi - lo)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_trajectory(traj: Union[Trajectory, np.ndarray], goal: Optional[Sequence[float]] = None,
                    result: Optional[Dict] = None, title: Optional[str] = None,
                    strict: float = 0.30, near: float = 0.50):
    """One trajectory with start (black), goal region (strict / near rings) and heading arrows."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    if goal is None and isinstance(traj, Trajectory):
        goal = traj.goal
    end_color = OUTCOME[result["metrics"]["outcome_soft"]] if result else None
    if goal is not None:
        draw_goal(ax, goal, strict, near)
    draw_trajectory(ax, traj, end_color=end_color)
    _frame(ax, [traj], [goal] if goal is not None else [])
    if title is None and result is not None:
        m = result["metrics"]
        title = f"{OUTCOME_LABEL[m['outcome_soft']]}: ends {m['d_final']:.2f} m from the goal"
    ax.set_title(title or "Predicted trajectory")
    handles = [Line2D([], [], marker="o", color=INK, linestyle="", label="start"),
               Line2D([], [], marker="*", color=OUTCOME["success"], linestyle="", markersize=10, label="goal"),
               Patch(facecolor="#e3f4e3", edgecolor=OUTCOME["success"], label=f"success region (< {strict:.2f} m)"),
               Line2D([], [], color=OUTCOME["near_success"], linestyle=(0, (3, 2)), label=f"near-success (< {near:.2f} m)")]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_comparison(traj_a: Trajectory, traj_b: Trajectory, labels: Sequence[str] = ("A", "B"),
                    strict: float = 0.30, near: float = 0.50, title: str = "Two commands, same goal"):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    draw_goal(ax, traj_a.goal, strict, near)
    for traj, lab, col in zip((traj_a, traj_b), labels, SERIES[:2]):
        draw_trajectory(ax, traj, color=col, label=lab)
    _frame(ax, [traj_a, traj_b], [traj_a.goal])
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_overlay(trajs: List[Trajectory], results: List[Dict], title: str = "Trajectories by goal outcome",
                 strict: float = 0.30, near: float = 0.50):
    """All trajectories in one frame, coloured by soft outcome; goals drawn once."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for g in {tuple(t.goal) for t in trajs}:
        draw_goal(ax, g, strict, near)
    for traj, res in zip(trajs, results):
        col = OUTCOME[res["metrics"]["outcome_soft"]]
        ax.plot(traj.x, traj.y, color=col, linewidth=1.4, alpha=0.85, zorder=3)
        ax.plot(traj.x[-1], traj.y[-1], marker="o", markersize=5, color=col, markeredgecolor=SURFACE, zorder=4)
    ax.plot(trajs[0].x[0], trajs[0].y[0], marker="o", markersize=7, color=INK, zorder=6)
    _frame(ax, trajs, [t.goal for t in trajs])
    counts = {k: sum(r["metrics"]["outcome_soft"] == k for r in results) for k in OUTCOME}
    ax.legend(handles=[Line2D([], [], color=OUTCOME[k], label=f"{OUTCOME_LABEL[k]} ({counts[k]})") for k in OUTCOME],
              loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_distance_histogram(d_final: Sequence[float], strict: float = 0.30, near: float = 0.50,
                            title: str = "Final distance to goal"):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    d = np.asarray(d_final, dtype=float)
    bins = np.linspace(0, max(d.max() * 1.05, near * 1.2), 30)
    ax.hist(d, bins=bins, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=1.0)
    for thr, key in ((strict, "success"), (near, "near_success")):
        ax.axvline(thr, color=OUTCOME[key], linewidth=1.2, linestyle=(0, (3, 2)))
        ax.text(thr, ax.get_ylim()[1] * 0.95, f" {thr:.2f} m", color=INK_2, fontsize=8, va="top")
    ax.set_xlabel("final distance to goal (m)")
    ax.set_ylabel("trajectories")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_risk_map(rm: Dict, title: str = "Empirical risk map", note: Optional[str] = None):
    """Heatmap of the strict failure rate per bin (one-hue ramp); sparse bins hatched."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xe, ye, rate = rm["x_edges"], rm["y_edges"], rm["rate"]
    ax.grid(False)
    mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(rate), cmap="Oranges", vmin=0, vmax=1, shading="flat")
    sparse = np.isnan(rate)
    for i, j in zip(*np.nonzero(sparse)):
        ax.add_patch(plt.Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], facecolor=SURFACE,
                                   edgecolor=GRID, hatch="///", linewidth=0))
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            if not np.isnan(rate[i, j]):
                val = rate[i, j]
                ax.text((xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if val > 0.6 else INK)
    names = {"theta": "initial orientation θ (rad)", "omega": "angular velocity ω (rad/s)", "v": "linear velocity v (m/s)"}
    ax.set_xlabel(names.get(rm["x"], rm["x"]))
    ax.set_ylabel(names.get(rm["y"], rm["y"]))
    cb = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("strict failure rate")
    cb.outline.set_visible(False)
    ax.set_title(title)
    if note:
        fig.text(0.01, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    return fig


def plot_novelty_map(scorer, v_rot: float = 0.0, n: int = 160, title: Optional[str] = None):
    """Novelty percentile over (orientation, v_lin) at fixed v_rot, with the label boundaries."""
    apply_style()
    th = np.linspace(-np.pi, np.pi, n)
    v = np.linspace(-1.6, 3.4, n)
    tt, vv = np.meshgrid(th, v)
    pts = np.column_stack([tt.ravel(), vv.ravel(), np.full(tt.size, v_rot)])
    grid = scorer.novelty_percentile_many(pts).reshape(n, n)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.grid(False)
    im = ax.imshow(grid, origin="lower", extent=(th[0], th[-1], v[0], v[-1]), aspect="auto", cmap="Blues",
                   vmin=0, vmax=100)
    ax.contour(th, v, grid, levels=[scorer.borderline_percentile, scorer.novel_percentile],
               colors=["#eda100", "#d03b3b"], linewidths=[1.4, 1.4], linestyles=["--", "-"])
    ax.legend(handles=[Line2D([], [], color="#eda100", linestyle="--",
                              label=f"borderline from the {scorer.borderline_percentile:g}th percentile"),
                       Line2D([], [], color="#d03b3b", label=f"novel from the {scorer.novel_percentile:g}th percentile")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    ax.set_xlabel("orientation θ (rad)")
    ax.set_ylabel("linear velocity v (m/s)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("novelty percentile (reference set)")
    cb.outline.set_visible(False)
    ax.set_title(title or f"Input novelty at ω = {v_rot:.1f} rad/s")
    fig.tight_layout()
    return fig


def plot_attribution(attribution: Dict[str, float], title: str = "Why is this input unusual?"):
    """Bar chart of the novelty attribution (Mahalanobis drop per feature)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    names = list(attribution)
    vals = [attribution[k] for k in names]
    ax.barh(names, vals, color=SERIES[0], height=0.5)
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK_2)
    ax.set_xlabel("drop in Mahalanobis distance if the feature were typical")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


plot_feature_importance = plot_attribution   # version 1 name
