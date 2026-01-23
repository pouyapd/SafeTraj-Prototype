import json
import requests


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_ollama_generate(model: str, prompt: str, timeout_s: int = 120) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def parse_json_or_fallback(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "label": "UNCERTAIN",
            "confidence": "low",
            "reasons": ["LLM did not return valid JSON."],
            "_raw": text,
        }
