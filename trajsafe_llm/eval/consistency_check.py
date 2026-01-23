import json
import sys
from pathlib import Path


def load_reports(outputs_dir: str):
    reports = []
    for p in Path(outputs_dir).glob("report_traj_*.json"):
        with open(p, "r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports


def check_report(report: dict):
    traj_id = report["traj_id"]
    rule = report["rule_risk"]
    llm = report.get("llm_explanation")

    issues = []

    # Rule-based expectations
    if rule["label"] == "SAFE":
        if llm is not None:
            issues.append("LLM should be skipped for SAFE trajectory.")
    else:
        if llm is None:
            issues.append("LLM explanation missing for non-SAFE trajectory.")
        else:
            # Label consistency
            if llm.get("label") != rule["label"]:
                issues.append(
                    f"Label mismatch: rule={rule['label']} llm={llm.get('label')}"
                )

            # Reasons must reference active rules
            active_rules = set(rule.get("rules", []))
            for r in llm.get("reasons", []):
                if not any(f"[{ar}]" in r for ar in active_rules):
                    issues.append(f"Reason not linked to active rule: '{r}'")

    return traj_id, issues


def main(outputs_dir: str):
    reports = load_reports(outputs_dir)
    if not reports:
        print("No reports found.")
        sys.exit(1)

    total = len(reports)
    total_issues = 0

    print(f"Evaluating {total} trajectory reports...\n")

    for rep in reports:
        traj_id, issues = check_report(rep)
        if issues:
            print(f"[traj_id={traj_id}] ISSUES:")
            for i in issues:
                print(f"  - {i}")
            total_issues += len(issues)
        else:
            print(f"[traj_id={traj_id}] OK")

    print("\nSummary:")
    print(f"  Trajectories checked: {total}")
    print(f"  Total issues found: {total_issues}")

    if total_issues > 0:
        sys.exit(2)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    main(out_dir)
