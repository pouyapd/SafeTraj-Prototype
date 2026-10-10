"""
llm.py — optional natural-language explanations from a local LLM (Ollama).

The LLM never decides anything. It only rephrases the rule-based result, and
its answer is accepted only if it passes a consistency check (same label,
every reason tied to a rule that actually fired, no numbers). If Ollama is not
running, times out, or the answer fails the check, the caller falls back to
the deterministic template report (safetraj/report.py).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "safety_analyst.txt"
DEFAULT_URL = "http://localhost:11434"


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable or returned an error."""


def ollama_available(url: str = DEFAULT_URL, timeout: float = 1.5) -> bool:
    try:
        import requests
        return requests.get(f"{url}/api/tags", timeout=timeout).ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str = "mistral:latest", url: str = DEFAULT_URL, timeout: float = 60.0) -> str:
    try:
        import requests
        r = requests.post(f"{url}/api/generate", json={"model": model, "prompt": prompt, "stream": False},
                          timeout=timeout)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as exc:
        raise LLMUnavailable(f"Ollama request failed: {exc}") from exc


def build_prompt(result: Dict[str, Any], template: Optional[str] = None) -> str:
    template = template if template is not None else PROMPT_PATH.read_text(encoding="utf-8")
    payload = {"metrics": result["metrics"], "checks": {k: result["checks"][k] for k in ("label", "score", "triggered")}}
    return template + "\n\nINPUT:\n" + json.dumps(payload, indent=2)


def parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from the model output (None if there is none)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def consistency_issues(explanation: Optional[Dict[str, Any]], checks: Dict[str, Any]) -> List[str]:
    """Problems that make an LLM explanation unusable (empty list = accepted)."""
    if explanation is None:
        return ["no valid JSON object in the answer"]
    issues = []
    if explanation.get("label") != checks["label"]:
        issues.append(f"label mismatch: checks={checks['label']} llm={explanation.get('label')}")
    reasons = explanation.get("reasons")
    if not isinstance(reasons, list) or not reasons:
        return issues + ["no reasons given"]
    active = set(checks["triggered"])
    for reason in reasons:
        if not isinstance(reason, str):
            issues.append("reason is not text")
            continue
        tags = set(re.findall(r"\[([a-z_]+)\]", reason))
        if not tags or not tags <= active:
            issues.append(f"reason not tied to a fired rule: {reason!r}")
        if re.search(r"\d", reason):
            issues.append(f"reason contains a number: {reason!r}")
    return issues
