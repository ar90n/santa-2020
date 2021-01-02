from __future__ import annotations

from returns.result import ResultE, safe
from dataclasses import dataclass
from typing import Callable, Optional, Any
import inspect

from santa_2020 import agents


@dataclass(frozen=True)
class Agent:
    key: str
    code: str | Callable
    resource: Optional[dict[Any, Any]]
    comment: Optional[str]


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

    def _register_agent_constructor(key: str, agent: Agent):
        nonlocal _registry
        _registry[key] = agent

    def _construct_agent(
        key: str, resource: Optional[dict[Any, Any]] = None, comment: Optional[str] = None
    ) -> Agent:
        nonlocal _registry
        return _registry[key](resource, comment)

    return _register_agent_constructor, _construct_agent


register, construct = _init_agent_registry()
