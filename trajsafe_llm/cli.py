import argparse
import json
import pandas as pd

from src.features import compute_features
from src.rules import rule_risk_label
from src.report import ensure_outputs_dir, save_json
from src.llm_ollama import load_text, call_ollama_generate, parse_json_or_fallback


def build_llm_prompt(template: str, features: dict, rule_risk: dict) -> str:
    payload = {"features": features, "rule_risk": rule_risk}
    return template + "\n\nINPUT:\n" + json.dumps(payload, indent=2)


def run_pipeline(input_csv: str, out_dir: str, model_name: str, prompt_path: str):
    df = pd.read_csv(input_csv)
    df = df.sort_values(["traj_id", "t"]).reset_index(drop=True)

    ensure_outputs_dir(out_dir)
    template = load_text(prompt_path)

    all_reports = []
    n_traj = df["traj_id"].nunique()

    print(f"Loaded {len(df)} rows across {n_traj} trajectories.")
    print(f"Output folder: {out_dir}")
    print(f"LLM model: {model_name}\n")

    for traj_id, g in df.groupby("traj_id"):
        g = g.sort_values("t")

        feats = compute_features(g)
        risk = rule_risk_label(feats)

        llm_json = None
        if risk["label"] in ("UNSAFE", "UNCERTAIN"):
            prompt = build_llm_prompt(template, feats, risk)
            llm_text = call_ollama_generate(model_name, prompt)
            llm_json = parse_json_or_fallback(llm_text)

        report = {
            "traj_id": int(traj_id),
            "features": feats,
            "rule_risk": risk,
            "llm_explanation": llm_json,
        }
        all_reports.append(report)

        save_json(report, f"{out_dir}/report_traj_{int(traj_id)}.json")

        # Console summary
        print(f"traj_id={int(traj_id)}")
        print(f"  rule_risk: {risk['label']} score={risk['score']} rules={risk['rules']}")
        if llm_json is None:
            print("  llm_explanation: skipped (SAFE)\n")
        else:
            print(f"  llm_label: {llm_json.get('label')} confidence={llm_json.get('confidence')}")
            print(f"  llm_reasons: {llm_json.get('reasons')}\n")

    save_json({"reports": all_reports}, f"{out_dir}/summary.json")
    print("Done. Saved reports.")


def main():
    parser = argparse.ArgumentParser(description="TrajSafe-LLM: trajectory safety reporting")
    parser.add_argument("--input", required=True, help="Path to input CSV (trajectory time-series).")
    parser.add_argument("--out", default="outputs", help="Output directory for JSON reports.")
    parser.add_argument("--model", default="mistral:latest", help="Ollama model name (e.g., mistral:latest).")
    parser.add_argument("--prompt", default="prompts/safety_analyst.txt", help="Path to prompt file.")
    args = parser.parse_args()

    run_pipeline(args.input, args.out, args.model, args.prompt)


if __name__ == "__main__":
    main()

