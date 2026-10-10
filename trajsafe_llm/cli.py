"""
TrajSafe-LLM command (kept for compatibility with version 1).

The feature extraction, rules and Ollama code moved into the safetraj package;
this wrapper runs `python -m safetraj batch` with an optional local LLM.

Usage (from the repository root):
    python trajsafe_llm/cli.py --input trajsafe_llm/data/demo_trajectories.csv --out outputs
    python trajsafe_llm/cli.py --input ... --out outputs --model mistral:latest   # needs Ollama
Without Ollama the deterministic template report is used.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetraj.cli import main as safetraj_main  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="TrajSafe-LLM: trajectory reports (wrapper around safetraj batch)")
    p.add_argument("--input", required=True, help="trajectory CSV: traj_id,t,x,y,theta,v,omega,goal_x,goal_y")
    p.add_argument("--out", default="outputs", help="output directory")
    p.add_argument("--model", default=None, help="Ollama model, e.g. mistral:latest (optional)")
    args = p.parse_args()
    argv = ["batch", "--input", args.input, "--out", args.out]
    if args.model:
        argv += ["--llm", args.model]
    return safetraj_main(argv)


if __name__ == "__main__":
    sys.exit(main())
