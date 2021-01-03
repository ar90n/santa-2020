from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pickle


def get_input_dir() -> Path:
    def has_input_dir(p: Path) -> bool:
        return any([child.name == "input" and child.is_dir() for child in p.iterdir()])

    cur = Path.cwd()
    if has_input_dir(cur):
        return cur / "input"
    while cur.parent != cur:
        cur = cur.parent
        if has_input_dir(cur):
            return cur / "input"
    raise EnvironmentError("Kaggle input dir is not found.")


def load_model(name: Optional[str]) -> dict[str, Any]:
    if name is None:
        return {}
    return {
        "model": pickle.loads(
            (get_input_dir() / "my-santa-2020-data" / f"{name}.pickle").read_bytes()
        )
    }

