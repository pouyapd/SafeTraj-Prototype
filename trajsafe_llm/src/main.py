import json
import pandas as pd

from src.features import compute_features
from src.rules import rule_risk_label
from src.report import ensure_outputs_dir, save_json
from src.llm_ollama import load_text, call_ollama_generate, parse_json_or_fallback


def build_llm_prompt(template: str, features: dict, rule_risk: dict) -> str:
    payload = {"features": features, "rule_risk": rule_risk}
    return template + "\n\nINPUT:\n" + json.dumps(payload, indent=2)


def main(model_name: str = "mistral:latest"):
    df = pd.read_csv("data/demo_trajectories.csv")
    df = df.sort_values(["traj_id", "t"]).reset_index(drop=True)

    print(f"Loaded {len(df)} rows across {df['traj_id'].nunique()} trajectories.\n")

    ensure_outputs_dir("outputs")
    template = load_text("prompts/safety_analyst.txt")

    all_reports = []

    for traj_id, g in df.groupby("traj_id"):
        g = g.sort_values("t")

        feats = compute_features(g)
        risk = rule_risk_label(feats)

        llm_json = None

        # Industry optimization: only call LLM if not clearly SAFE
        if risk["label"] in ("UNSAFE", "UNCERTAIN"):
            prompt = build_llm_prompt(template, feats, risk)
            llm_text = call_ollama_generate(model_name, prompt)
            llm_json = parse_json_or_fallback(llm_text)

        report = {
            "traj_id": int(traj_id),
            "features": feats,
            "rule_risk": risk,
            "llm_explanation": llm_json,  # None for SAFE
        }
        all_reports.append(report)
        save_json(report, f"outputs/report_traj_{int(traj_id)}.json")

        print(f"traj_id={int(traj_id)}")
        print(f"  rule_risk: {risk['label']} score={risk['score']} rules={risk['rules']}")
        if llm_json is None:
            print("  llm_explanation: skipped (SAFE)\n")
        else:
            print(f"  llm_label: {llm_json.get('label')} confidence={llm_json.get('confidence')}")
            print(f"  llm_reasons: {llm_json.get('reasons')}\n")

    save_json({"reports": all_reports}, "outputs/summary.json")
    print("Saved JSON reports (LLM only for UNSAFE/UNCERTAIN) to outputs/.")


if __name__ == "__main__":
    main()
