"""
test_dashboard.py — the Streamlit app runs every tab without errors (headless).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def test_dashboard_runs_without_errors():
    # absolute path: newer Streamlit versions resolve relative paths against this test file
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "final distance to goal" in labels and "strict success rate" in labels
