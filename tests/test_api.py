"""
test_api.py — REST routes, input validation and consistency with the Python API.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import CONFIG, app
from safetraj import SafeTrajEvaluator

client = TestClient(app)


def test_info_routes():
    assert client.get("/health").json() == {"status": "ok"}
    assert "/evaluate" in client.get("/").json()["endpoints"]
    cfg = client.get("/config").json()
    assert cfg["goal_thresholds_m"] == {"strict_success": 0.3, "near_success": 0.5}


def test_evaluate_matches_python_api():
    body = {"orientation": 2.5, "v_lin": 1.8, "v_rot": 0.4, "goal_x": 1.0, "goal_y": 0.0}
    api = client.post("/evaluate", json=body).json()
    py = SafeTrajEvaluator(CONFIG).evaluate([2.5, 1.8, 0.4], goal=(1.0, 0.0))
    assert api["metrics"] == pytest.approx(py["metrics"])
    assert api["checks"]["label"] == py["checks"]["label"]
    assert api["novelty"]["label"] == py["novelty"]["label"]
    assert "trajectory" not in api


def test_evaluate_can_return_trajectory():
    out = client.post("/evaluate", json={"include_trajectory": True}).json()
    assert len(out["trajectory"]) == CONFIG.num_steps + 1
    assert set(out["trajectory"][0]) == {"t", "x", "y", "theta", "v", "omega"}


@pytest.mark.parametrize("body", [
    {"orientation": 4.0}, {"v_lin": 3.5}, {"v_rot": -2.5}, {"predictor": "neural_net"}, {"goal_x": "far"},
])
def test_evaluate_validation_errors(body):
    r = client.post("/evaluate", json=body)
    assert r.status_code == 422


def test_trajectory_endpoint_and_errors():
    pts = [{"t": 0.1 * i, "x": 0.1 * i, "y": 0.0, "theta": 0.0, "v": 1.0, "omega": 0.0} for i in range(11)]
    ok = client.post("/evaluate/trajectory", json={"points": pts, "goal_x": 1.0, "goal_y": 0.0})
    assert ok.status_code == 200 and ok.json()["metrics"]["outcome_soft"] == "success"
    assert ok.json()["novelty"] is None
    with_cmd = client.post("/evaluate/trajectory",
                           json={"points": pts, "goal_x": 1.0, "goal_y": 0.0, "command": [0.0, 1.0, 0.0]})
    assert with_cmd.json()["novelty"]["label"] in ("typical", "borderline", "novel")
    short = client.post("/evaluate/trajectory", json={"points": pts[:1], "goal_x": 1.0, "goal_y": 0.0})
    assert short.status_code == 422
    backwards = pts[:3] + [dict(pts[3], t=0.0)]
    r = client.post("/evaluate/trajectory", json={"points": backwards, "goal_x": 1.0, "goal_y": 0.0})
    assert r.status_code == 422 and "increasing" in r.json()["detail"]


def test_batch_endpoint():
    r = client.post("/evaluate/batch", json={"commands": [[0, 0.5, 0], [3.0, 2.5, 0.0]]})
    body = r.json()
    assert body["n"] == 2 and sum(body["check_labels"].values()) == 2
    too_many = client.post("/evaluate/batch", json={"commands": [[0, 0.5, 0]] * 501})
    assert too_many.status_code == 422
    bad = client.post("/evaluate/batch", json={"commands": [[0, 9.0, 0]]})
    assert bad.status_code == 422 and "v_lin" in str(bad.json())


def test_compare_and_report():
    r = client.post("/compare", json={"a": {"orientation": 0.0}, "b": {"orientation": 3.0, "v_lin": 2.5}})
    assert r.status_code == 200 and "d_final" in r.json()["difference_b_minus_a"]
    mismatch = client.post("/compare", json={"a": {"goal_x": 1.0}, "b": {"goal_x": 2.0}})
    assert mismatch.status_code == 422
    rep = client.post("/report", json={"orientation": 3.0, "v_lin": 2.5}).json()
    assert "Checks:" in rep["report"] and rep["checks"] in ("PASS", "REVIEW", "FAIL")
