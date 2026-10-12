"""
test_evaluator.py — the shared evaluator, analysis tools and reports.
"""

import json

import numpy as np
import pandas as pd
import pytest

from safetraj import SafeTrajConfig, SafeTrajEvaluator, results_to_frame
from safetraj.analysis import fit_failure_tree, leaf_rules, risk_map, run_sweep, sample_commands, summarize_sweep
from safetraj.predictor import GoalSeekingPredictor
from safetraj import llm, report


@pytest.fixture(scope="module")
def ev():
    return SafeTrajEvaluator(SafeTrajConfig(seed=0))


def test_evaluate_runs(ev):
    res = ev.evaluate([0.0, 0.5, 0.0])
    assert "risk_score" in res and "risk_label" in res          # version 1 keys still present
    assert isinstance(res["risk_score"], float)
    assert set(res) >= {"metrics", "checks", "novelty", "input", "goal"}
    json.dumps(res)                                             # JSON-serialisable


def test_evaluate_rejects_bad_command(ev):
    with pytest.raises(ValueError):
        ev.evaluate([0.0, float("nan"), 0.0])
    with pytest.raises(ValueError):
        ev.evaluate([0.0, 1.0])


def test_compare_reports_differences(ev):
    out = ev.compare([0.0, 0.5, 0.0], [3.0, 2.5, 0.0])
    d = out["difference_b_minus_a"]
    assert d["d_final"] == pytest.approx(out["b"]["metrics"]["d_final"] - out["a"]["metrics"]["d_final"])


def test_results_to_frame_one_row_per_trajectory(ev):
    rows = results_to_frame(ev.evaluate_commands([[0, 0.5, 0], [1, 1, 1], [3, 2.5, 0]]))
    assert len(rows) == 3 and {"d_final", "check_label", "novelty_label", "rules_triggered"} <= set(rows.columns)


def test_sweep_risk_map_and_tree():
    cmds = sample_commands(120, seed=3)
    df = run_sweep(GoalSeekingPredictor(), cmds, [(1.0, 0.0), (0.5, 0.5)])
    assert len(df) == 240 and set(df.outcome_soft) <= {"success", "near_success", "failure"}
    rm = risk_map(df[df.goal_x == 1.0], "theta", "omega", bins=(4, 4), min_count=1)
    assert rm["counts"].sum() == 120
    assert np.nanmax(rm["rate"]) <= 1.0 and np.nanmin(rm["rate"]) >= 0.0
    tree = fit_failure_tree(df)
    assert 0 <= tree.cv_accuracy_mean <= 1 and tree.majority_baseline >= 0.5
    assert tree.n_failures == int(df.failure.sum())
    summary = summarize_sweep(df)
    assert len(summary) == 2 and summary["samples"].tolist() == [120, 120]


def test_risk_map_rate_is_failure_share():
    df = pd.DataFrame({"theta": [-3.0, -3.0, 3.0, 3.0], "omega": [0.0, 0.0, 0.0, 0.0], "failure": [1, 0, 1, 1]})
    rm = risk_map(df, "theta", "omega", bins=(2, 1), min_count=1)
    np.testing.assert_allclose(rm["rate"], [[0.5, 1.0]])


def test_leaf_rules_merge_bounds():
    from sklearn.tree import DecisionTreeClassifier
    X = np.array([[x] for x in np.linspace(0, 10, 101)])
    y = ((X[:, 0] > 3) & (X[:, 0] <= 6)).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    rules = leaf_rules(tree, ["speed"])
    assert len(rules) == 1 and rules[0]["conditions"].startswith("3.0") and "< speed <=" in rules[0]["conditions"]
    assert rules[0]["failure_rate"] == 1.0


def test_template_report_states_facts(ev):
    res = ev.evaluate([3.0, 2.5, 1.5])
    text = report.template_report(res)
    assert f"{res['metrics']['d_final']:.2f} m" in text
    for name in res["checks"]["triggered"]:
        assert name in text
    assert "not a probability" in text


def test_llm_fallback_when_unavailable(ev, monkeypatch):
    def boom(*a, **k):
        raise llm.LLMUnavailable("connection refused")
    monkeypatch.setattr(llm, "call_ollama", boom)
    res = ev.evaluate([3.0, 2.5, 1.5])
    out = report.explain(res, use_llm=True)
    assert out["source"] == "template" and "unavailable" in out["llm_note"]


def test_llm_answer_accepted_only_if_consistent(ev, monkeypatch):
    res = ev.evaluate([3.0, 2.5, 1.5])
    fired = res["checks"]["triggered"]
    good = json.dumps({"label": res["checks"]["label"], "reasons": [f"Check failed here [{r}]" for r in fired]})
    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: good)
    assert report.explain(res, use_llm=True)["source"] == "llm"
    bad = json.dumps({"label": "PASS", "reasons": ["Ends 1.2 m away [far_from_goal_end]"]})
    monkeypatch.setattr(llm, "call_ollama", lambda *a, **k: bad)
    out = report.explain(res, use_llm=True)
    assert out["source"] == "template" and "label mismatch" in out["llm_note"] and "number" in out["llm_note"]


def test_llm_skipped_for_pass(ev):
    res = ev.evaluate([0.0, 0.5, 0.0])
    assert res["checks"]["label"] == "PASS"
    assert report.explain(res, use_llm=True)["llm_note"].startswith("LLM skipped")
