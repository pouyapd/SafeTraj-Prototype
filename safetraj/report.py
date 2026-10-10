"""
report.py — human-readable reports.

`template_report` is deterministic: it states the goal outcome, every rule
that fired with its measured value, and the input novelty, and always works.
`explain` adds an optional local-LLM phrasing (Ollama) and falls back to the
template when the LLM is unavailable or its answer fails the consistency check.
`write_batch_report` writes a Markdown report with tables and figures for a
batch of trajectories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import llm

NOVELTY_NOTE = ("Novelty compares the input command with the reference commands; it is not a probability "
                "that the trajectory is unsafe.")


def template_report(result: Dict[str, Any]) -> str:
    m, chk = result["metrics"], result["checks"]
    name = f"Trajectory {result['traj_id']}" if "input" not in result else (
        "Command (orientation {orientation:+.2f} rad, v {v_lin:+.2f} m/s, omega {v_rot:+.2f} rad/s)"
        .format(**result["input"]))
    goal = result["goal"]
    outcome = {"success": "strict success", "near_success": "near-success", "failure": "failure"}[m["outcome_soft"]]
    lines = [f"{name} toward goal ({goal[0]:.2f}, {goal[1]:.2f}): ends {m['d_final']:.2f} m from the goal "
             f"({outcome}); closest approach {m['d_min']:.2f} m; path length {m['path_length']:.2f} m."]
    fired = [r for r in chk["rules"] if r["triggered"]]
    if fired:
        reasons = "; ".join(f"{r['name']}: {r['message']}" for r in fired)
        lines.append(f"Checks: {chk['label']} (score {chk['score']}). Fired: {reasons}.")
    else:
        lines.append(f"Checks: {chk['label']} (score 0). No rule fired.")
    nov = result.get("novelty")
    if nov:
        extra = " One or more inputs lie outside the reference range." if nov["outside_reference_range"] else ""
        lines.append(f"Input novelty: {nov['label']} (Mahalanobis distance {nov['mahalanobis']:.2f}, higher than "
                     f"{nov['mahalanobis_percentile']:.0f}% of reference inputs; Isolation Forest score higher than "
                     f"{nov['isolation_forest_percentile']:.0f}%).{extra} {NOVELTY_NOTE}")
    return " ".join(lines)


def explain(result: Dict[str, Any], use_llm: bool = False, model: str = "mistral:latest",
            url: str = llm.DEFAULT_URL, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Returns {"text": ..., "source": "template" | "llm", "llm_note": ...}.
    The LLM is only asked for non-PASS results, as in the original TrajSafe-LLM design.
    """
    out = {"text": template_report(result), "source": "template", "llm_note": None}
    if not use_llm:
        return out
    if result["checks"]["label"] == "PASS":
        out["llm_note"] = "LLM skipped: no rule fired"
        return out
    try:
        answer = llm.call_ollama(llm.build_prompt(result), model=model, url=url, timeout=timeout)
    except llm.LLMUnavailable as exc:
        out["llm_note"] = f"LLM unavailable, template used ({exc})"
        return out
    parsed = llm.parse_llm_json(answer)
    issues = llm.consistency_issues(parsed, result["checks"])
    if issues:
        out["llm_note"] = "LLM answer rejected, template used: " + "; ".join(issues)
        return out
    out.update(text=" ".join(parsed["reasons"]), source="llm", llm_note=f"phrased by {model}")
    return out


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_batch_report(table: pd.DataFrame, reports: List[Dict[str, Any]], out_dir: Path,
                       figures: Optional[Dict[str, str]] = None, title: str = "SafeTraj batch report",
                       source: str = "") -> Path:
    """Markdown report: overview counts, per-trajectory table, figures and text explanations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if source:
        lines += [f"Input: `{source}` · {len(table)} trajectories", ""]
    lines += ["## Overview", ""]
    overview = pd.DataFrame({
        "goal outcome (soft)": table["outcome_soft"].value_counts().reindex(
            ["success", "near_success", "failure"], fill_value=0),
    })
    lines += [_md_table(overview.reset_index().rename(columns={"index": "outcome"}), "{}"), ""]
    checks = table["check_label"].value_counts().reindex(["PASS", "REVIEW", "FAIL"], fill_value=0)
    lines += [_md_table(checks.rename("trajectories").reset_index().rename(columns={"index": "check label"}), "{}"), ""]
    if "novelty_label" in table:
        nov = table["novelty_label"].value_counts().reindex(["typical", "borderline", "novel"], fill_value=0)
        lines += [_md_table(nov.rename("trajectories").reset_index().rename(columns={"index": "input novelty"}), "{}"),
                  "", f"_{NOVELTY_NOTE}_", ""]
    lines += ["## Per-trajectory results", ""]
    cols = [c for c in ("traj_id", "d_final", "d_min", "progress", "path_length", "outcome_soft", "check_label",
                        "rules_triggered", "novelty_label") if c in table]
    lines += [_md_table(table[cols]), ""]
    if figures:
        lines += ["## Figures", ""]
        for caption, rel in figures.items():
            lines += [f"![{caption}]({rel})", "", f"_{caption}_", ""]
    lines += ["## Explanations", ""]
    for rep in reports:
        note = f" ({rep['llm_note']})" if rep.get("llm_note") else ""
        lines += [f"- **{rep['traj_id']}** [{rep['source']}{note}]: {rep['text']}"]
    lines += ["", "Checks are heuristic rules with fixed thresholds (see `safetraj/rules.py`); "
                  "PASS means that no rule fired, not that a trajectory is safe."]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
