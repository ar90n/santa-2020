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


def _comment_out(comment: str) -> str:
    return "\n".join([f"# {l}" for l in comment.split("\n")])


def try_to_submit_source(agent: Agent) -> ResultE[str]:
    from . import common

    def _add_comment(source_lines: str, comment: Optional[str]) -> str:
        if comment is None:
            return source_lines
        comment_lines = _comment_out(comment)
        return f"{source_lines}\n\n{comment_lines}"

    def _add_resource(source_lines: str, resource: Optional[dict[str, str]]) -> str:
        if resource is None:
            return source_lines
        return f"{source_lines}\n\n_RESOURCE={str(resource)}"

    common_source_lines = inspect.getsource(common)
    return (
        try_to_str(agent)
        .map(
            lambda agent_source_lines: f"{common_source_lines}\n\n{agent_source_lines}"
        )
        .map(lambda source_lines: _add_resource(source_lines, agent.resource))
        .map(lambda source_lines: _add_comment(source_lines, agent.comment))
    )


def _init_agent_registry() -> tuple[Callable[[Agent], None], Callable[[str], Agent]]:
    _registry = {}

    def _register_agent_constructor(key: str, agent: Agent):
        nonlocal _registry
        _registry[key] = agent

    def _construct_agent(
        key: str,
        resource: Optional[dict[Any, Any]] = None,
        comment: Optional[str] = None,
    ) -> Agent:
        nonlocal _registry
        return _registry[key](resource, comment)

    return _register_agent_constructor, _construct_agent


register, construct = _init_agent_registry()
