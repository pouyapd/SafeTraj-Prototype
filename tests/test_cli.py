"""
test_cli.py — command-line workflows produce the documented files.
"""

import json
from pathlib import Path

import pandas as pd

from safetraj.cli import main

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "data" / "sample_trajectories.csv"


def test_evaluate_json(capsys):
    assert main(["evaluate", "--orientation", "0", "--v", "0.5", "--omega", "0.2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["checks"]["label"] == "PASS" and out["report"]["source"] == "template"


def test_batch_on_sample_data(tmp_path):
    assert main(["batch", "--input", str(SAMPLE), "--out", str(tmp_path)]) == 0
    table = pd.read_csv(tmp_path / "results.csv")
    assert len(table) == pd.read_csv(SAMPLE)["traj_id"].nunique()
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# SafeTraj batch report")
    assert (tmp_path / "overlay.png").is_file() and (tmp_path / "results.json").is_file()


def test_batch_reports_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("traj_id,t,x\n1,0,0\n", encoding="utf-8")
    assert main(["batch", "--input", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert "missing column" in capsys.readouterr().err


def test_sweep_and_compare(tmp_path):
    assert main(["sweep", "--n", "60", "--goals", "1.0,0.0", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "sweep.csv").is_file() and (tmp_path / "risk_map_goal_1_0.png").is_file()
    assert "Failure rules" in (tmp_path / "tree_rules.txt").read_text(encoding="utf-8")
    assert main(["compare", "--a", "0", "0.5", "0", "--b", "3", "2.5", "0", "--plot", str(tmp_path / "c.png")]) == 0
    assert (tmp_path / "c.png").is_file()


def test_sample_data_is_reproducible(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert main(["sample-data", "--out", str(a)]) == 0 and main(["sample-data", "--out", str(b)]) == 0
    assert a.read_text() == b.read_text() == SAMPLE.read_text()
