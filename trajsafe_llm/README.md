# TrajSafe-LLM (now part of `safetraj`)

> Part of [SafeTraj-Prototype](../README.md).

Version 1 of this folder held its own feature extraction, rules and Ollama client.
In version 2 that code lives in the `safetraj` package so the CLI, REST API and
dashboard share one implementation:

| Version 1 file | Version 2 location |
|---|---|
| `src/features.py` | `safetraj/metrics.py` |
| `src/rules.py` | `safetraj/rules.py` (labels PASS / REVIEW / FAIL) |
| `src/llm_ollama.py`, `eval/consistency_check.py` | `safetraj/llm.py` |
| `prompts/safety_analyst.txt` | `safetraj/prompts/safety_analyst.txt` |

The LLM only rephrases the rule result. Its answer is used only if it passes a
consistency check (same label, every reason tied to a fired rule, no numbers);
otherwise, or when Ollama is not running, the deterministic template report is
used. The LLM is never asked for trajectories with label PASS.

```bash
# from the repository root
python trajsafe_llm/cli.py --input trajsafe_llm/data/demo_trajectories.csv --out outputs
python trajsafe_llm/cli.py --input trajsafe_llm/data/demo_trajectories.csv --out outputs --model mistral:latest
```

Outputs: `results.csv`, `results.json`, `report.md` and figures in `--out`.
