# TrajSafe-LLM (SafeTraj-X module)

This module generates **trajectory safety reports** using a practical design:

- A **rule-based** risk label (deterministic and auditable)
- An optional **local LLM explanation layer** via Ollama (only for `UNSAFE/UNCERTAIN`)
- A simple **CLI** entry point
- A small **evaluation script** to check consistency between rules and explanations

The LLM is used only to explain decisions — it is not the decision maker.

---

## What problem does this solve?

In safety-critical navigation, a risk label alone is often not enough for debugging or audit.
This module bridges **deterministic safety rules** with **human-readable explanations**, without delegating safety decisions to an LLM.

---

## Quick start

From this folder:

bash
python cli.py --input data/demo_trajectories.csv --out outputs --model mistral:latest

## Outputs

When the CLI is executed, JSON safety reports are generated under the `outputs/` directory:

- `outputs/report_traj_<id>.json`
- `outputs/summary.json`

These files are generated automatically and are not tracked by git.


## Example output (UNSAFE)

```json
{
  "label": "UNSAFE",
  "confidence": "high",
  "reasons": [
    "Low progress towards the goal [low_progress]",
    "Heading error remains substantial far from goal [bad_heading_far_from_goal]"
  ]
}

