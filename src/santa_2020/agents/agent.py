from __future__ import annotations

from returns.result import ResultE, safe
from dataclasses import dataclass
from typing import Callable
import inspect

from santa_2020 import agents


@dataclass(frozen=True)
class Agent:
    key: str
    code: str | Callable


@safe
def try_to_str(agent: Agent) -> str:
    if isinstance(agent.code, str):
        return agent.code
    elif callable(agent.code):
        return inspect.getsource(agent.code)

    raise ValueError("failed to convert code to str.")


def try_to_submit_source(agent: Agent) -> ResultE[str]:
    from . import common

    common_source_lines = inspect.getsource(common)
    return try_to_str(agent).map(
        lambda agent_source_lines: f"{common_source_lines}\n\n{agent_source_lines}"
    )


def _init_agent_registry() -> tuple[Callable[[Agent], None], Callable[[str], Agent]]:
    _registry = {}

    def _register_agent(agent: Agent):
        nonlocal _registry
        _registry[agent.key] = agent

    def _get_agent(key: str) -> Agent:
        nonlocal _registry
        return _registry[key]

    return _register_agent, _get_agent


register, get = _init_agent_registry()
