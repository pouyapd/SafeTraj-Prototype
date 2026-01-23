# TrajSafe-LLM (SafeTraj-X module)

This module generates **trajectory safety reports** using a practical design:

- A **rule-based** risk label (deterministic and auditable)
- An optional **local LLM explanation layer** via Ollama (only for `UNSAFE/UNCERTAIN`)
- A simple **CLI** entry point
- A small **evaluation script** to check consistency

The LLM is used only to explain decisions — it is not the decision maker.

## Quick start

From this folder:

```bash
python cli.py --input data/demo_trajectories.csv --out outputs --model mistral:latest
