from typing import Optional
from returns.io import impure_safe, IOResultE
from .agents import Agent, try_to_submit_source
from pathlib import Path


def save_submit(agent: Agent, filename: Optional[str] = None) -> IOResultE[int]:
    if filename is None:
        filename = "submission.py"

    return try_to_submit_source(agent).bind(
        lambda source: impure_safe(Path(filename).write_text)(source)
    )