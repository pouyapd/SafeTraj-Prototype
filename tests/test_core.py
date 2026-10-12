"""
test_core.py — trajectory validation, metrics, predictors, novelty scores and rule checks.
"""

import numpy as np
import pytest

from safetraj import (ConstantCommandPredictor, DummyTrajectoryPredictor, GoalSeekingPredictor, SafeTrajConfig,
                      Trajectory, TrajectoryError)
from safetraj.data import generate_training_data
from safetraj.metrics import compute_metrics
from safetraj.ood import MahalanobisOOD, NoveltyScorer
from safetraj.rules import label_from_score, run_checks
from safetraj.trajectory import frame_to_trajectories, load_trajectories, trajectories_to_frame


def line(n=11, goal=(1.0, 0.0), v=1.0, **kw):
    t = np.linspace(0, 1, n)
    return Trajectory(t=t, x=v * t, y=np.zeros(n), theta=np.zeros(n), v=np.full(n, v), omega=np.zeros(n),
                      goal=goal, **kw)


# ── trajectory input validation ─────────────────────────────────────────────

def test_trajectory_rejects_bad_input():
    with pytest.raises(TrajectoryError, match="at least 2"):
        line(n=1)
    t = np.linspace(0, 1, 5)
    with pytest.raises(TrajectoryError, match="NaN"):
        Trajectory(t=t, x=[0, 1, np.nan, 3, 4], y=t, theta=t, v=t, omega=t, goal=(1, 0))
    with pytest.raises(TrajectoryError, match="strictly increasing"):
        Trajectory(t=[0, 1, 1, 2, 3], x=t, y=t, theta=t, v=t, omega=t, goal=(1, 0))
    with pytest.raises(TrajectoryError, match="must have 5 values"):
        Trajectory(t=t, x=t[:4], y=t, theta=t, v=t, omega=t, goal=(1, 0))


def test_csv_validation_and_roundtrip(tmp_path):
    df = trajectories_to_frame([line(traj_id=1), line(goal=(2.0, 1.0), traj_id=2)])
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    back = load_trajectories(path)
    assert [t.traj_id for t in back] == [1, 2] and back[1].goal == (2.0, 1.0)
    with pytest.raises(TrajectoryError, match="missing column"):
        frame_to_trajectories(df.drop(columns=["omega"]))
    bad = df.copy()
    bad["x"] = bad["x"].astype(object)
    bad.loc[3, "x"] = "abc"
    with pytest.raises(TrajectoryError, match="non-numeric"):
        frame_to_trajectories(bad)
    moving_goal = df.copy()
    moving_goal.loc[0, "goal_x"] = 9.0
    with pytest.raises(TrajectoryError, match="constant"):
        frame_to_trajectories(moving_goal)
    with pytest.raises(TrajectoryError, match="not found"):
        load_trajectories(tmp_path / "missing.csv")


# ── metrics ─────────────────────────────────────────────────────────────────

def test_metrics_straight_line_to_goal():
    m = compute_metrics(line(goal=(1.0, 0.0)))
    assert m.d_start == pytest.approx(1.0) and m.d_final == pytest.approx(0.0, abs=1e-12)
    assert m.progress == pytest.approx(1.0) and m.path_length == pytest.approx(1.0)
    assert m.path_efficiency == pytest.approx(1.0)
    assert m.outcome_strict == "success" and m.outcome_soft == "success"


@pytest.mark.parametrize("goal_x,soft,strict", [(1.25, "success", "success"), (1.4, "near_success", "failure"),
                                                (1.6, "failure", "failure")])
def test_soft_outcome_thresholds(goal_x, soft, strict):
    m = compute_metrics(line(goal=(goal_x, 0.0)))
    assert (m.outcome_soft, m.outcome_strict) == (soft, strict)


def test_curvature_of_circle_and_reverse_fraction():
    traj = ConstantCommandPredictor(dt=0.01, num_steps=100).predict((0.0, 0.5, 1.0), (0, 1))
    m = compute_metrics(traj)
    assert m.max_curvature == pytest.approx(1.0 / 0.5, rel=1e-3)        # kappa = omega / v
    rev = line(v=-0.5, goal=(-1, 0))
    assert compute_metrics(rev).reverse_fraction == 1.0


def test_metrics_reject_bad_thresholds():
    with pytest.raises(ValueError):
        compute_metrics(line(), strict_threshold=0.6, near_threshold=0.5)


# ── predictors ──────────────────────────────────────────────────────────────

def test_constant_command_is_exact_unicycle_arc():
    p = ConstantCommandPredictor(dt=0.1, num_steps=63)
    traj = p.predict((0.0, 1.0, 1.0), (0, 0))
    r = np.hypot(traj.x - 0.0, traj.y - 1.0)          # circle of radius v/omega = 1 centred at (0, 1)
    np.testing.assert_allclose(r, 1.0, atol=1e-12)
    straight = p.predict((np.pi / 2, 0.5, 0.0), (0, 0))
    np.testing.assert_allclose(straight.x, 0.0, atol=1e-12)
    assert straight.y[-1] == pytest.approx(0.5 * 6.3)


def test_version1_predictor_now_curves_with_omega():
    """Version 1 drew a straight line for any omega; the path must now depend on omega."""
    p = DummyTrajectoryPredictor()
    a, b = p.predict([0.0, 1.0, 0.0]), p.predict([0.0, 1.0, 1.5])
    assert a.shape == (20, 3)
    np.testing.assert_allclose(a[:, 1], 0.0, atol=1e-12)
    assert b[-1, 1] > 0.3


def test_goal_seeking_predictor_behaviour():
    p = GoalSeekingPredictor()
    facing = compute_metrics(p.predict((0.0, 0.5, 0.0), (1.0, 0.0)))
    assert facing.outcome_strict == "success"
    away = compute_metrics(p.predict((3.0, 2.5, 0.0), (1.0, 0.0)))
    assert away.outcome_strict == "failure"
    traj = p.predict((1.0, 2.8, -1.9), (1.5, -0.5))
    assert np.all(np.abs(traj.omega) <= 1.99 + 1e-9)
    assert np.all(np.abs(np.diff(traj.v)) <= p.a_max * p.dt + 1e-9)
    with pytest.raises(ValueError):
        p.predict((0, 0, 0), None)


# ── novelty ─────────────────────────────────────────────────────────────────

def test_mahalanobis_is_centred_once():
    """Regression test for the version 1 bug (mean subtracted twice)."""
    ref = generate_training_data(800, seed=42)
    maha = MahalanobisOOD(ref)
    mean = ref.mean(axis=0)
    assert maha.score(mean) == pytest.approx(0.0, abs=1e-9)
    x = np.array([0.5, 2.0, -1.0])
    cov = np.cov(ref.T, bias=True)
    expected = np.sqrt((x - mean) @ np.linalg.inv(cov) @ (x - mean))
    assert maha.score(x) == pytest.approx(expected, rel=1e-9)


def test_novelty_labels_and_frequency_meaning():
    ref = generate_training_data(800, seed=42)
    scorer = NoveltyScorer(ref)
    typical = scorer.score(ref.mean(axis=0))
    assert typical.label == "typical" and typical.score < 0.5
    far = scorer.score([3.1, 2.85, 1.98])
    assert far.label in ("borderline", "novel")
    out = scorer.score([0.0, 5.0, 0.0])
    assert out.outside_reference_range and out.label == "novel" and out.score == 1.0
    labels = [scorer.score(x).label for x in ref[:400]]
    flagged = np.mean([lab != "typical" for lab in labels])
    assert 0.03 < flagged < 0.15            # ~5% per detector by construction, either detector can flag
    with pytest.raises(ValueError):
        scorer.score([0.0, np.inf, 0.0])


def test_novelty_attribution_points_at_unusual_feature():
    scorer = NoveltyScorer(generate_training_data(800, seed=1))
    attr = scorer.score([0.0, 2.8, 0.0]).attribution        # only v_lin is extreme
    assert max(attr, key=attr.get) == "v_lin" and attr["v_lin"] > 0


def test_custom_reference_data_changes_novelty():
    from safetraj import SafeTrajEvaluator
    narrow = np.random.default_rng(0).normal([0.0, 0.5, 0.0], [0.2, 0.1, 0.1], size=(500, 3))
    ev = SafeTrajEvaluator(reference_data=narrow)
    assert ev.evaluate([0.0, 0.5, 0.0])["novelty"]["label"] == "typical"
    assert ev.evaluate([2.0, 2.0, 1.0])["novelty"]["label"] == "novel"


# ── rule checks ─────────────────────────────────────────────────────────────

def test_rules_fire_at_thresholds():
    cfg = SafeTrajConfig()
    ok = run_checks(compute_metrics(line()), cfg)
    assert ok.label == "PASS" and ok.triggered == []
    stuck = Trajectory(t=np.arange(5.0), x=np.zeros(5), y=np.zeros(5), theta=np.full(5, np.pi), v=np.zeros(5),
                       omega=[0, 2.5, -0.5, 0, 0], goal=(1.0, 0.0))
    chk = run_checks(compute_metrics(stuck), cfg)
    assert set(chk.triggered) == {"far_from_goal_end", "low_progress", "omega_too_high", "high_omega_jumps",
                                  "bad_heading_far_from_goal"}
    assert chk.score == 8 and chk.label == "FAIL"
    assert [label_from_score(s) for s in (0, 1, 2, 4, 5)] == ["PASS", "PASS", "REVIEW", "REVIEW", "FAIL"]


def test_config_validation():
    with pytest.raises(ValueError, match="strict_threshold"):
        SafeTrajConfig(strict_threshold=0.6, near_threshold=0.5)
    with pytest.raises(ValueError, match="percentile"):
        SafeTrajConfig(borderline_percentile=99.9, novel_percentile=99.0)
    SafeTrajConfig(maha_thr=3.0, iso_thr=0.3)          # version 1 arguments still accepted


def test_vectorised_novelty_matches_single_scores():
    scorer = NoveltyScorer(generate_training_data(300, seed=5))
    X = np.array([[0.0, 0.5, 0.0], [3.0, 2.8, 1.9], [0.0, 4.0, 0.0], [-2.0, -1.0, 1.0]])
    many = scorer.novelty_percentile_many(X)
    single = [scorer.score(x).score * 100 for x in X]
    np.testing.assert_allclose(many, single)
