"""
test_dashboard.py — the Streamlit app runs every tab without errors (headless).
"""

from streamlit.testing.v1 import AppTest


def test_dashboard_runs_without_errors():
    at = AppTest.from_file("dashboard/app.py", default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert "final distance to goal" in labels and "strict success rate" in labels
