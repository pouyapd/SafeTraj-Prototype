import json
import os


def ensure_outputs_dir(path: str = "outputs") -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
