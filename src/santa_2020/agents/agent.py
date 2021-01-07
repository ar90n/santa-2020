from __future__ import annotations

from .common import serialize
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
    name: Optional[str]


@safe
def try_to_str(agent: Agent) -> str:
    if isinstance(agent.code, str):
        return agent.code
    elif callable(agent.code):
        return inspect.getsource(agent.code)

    raise ValueError("failed to convert code to str.")


def try_to_submit_source(agent: Agent) -> ResultE[str]:
    from . import common

    def _comment_out(comment: str) -> str:
        return "\n".join([f"# {l}" for l in comment.split("\n")])

    def _get_comment_source_lines(comment: Optional[str]) -> str:
        if comment is None:
            return ""
        return _comment_out(comment)

    def _get_resource_source_lines(resource: Optional[dict[str, Any]]) -> str:
        if resource is None:
            return ""
        return f"_RESOURCE={serialize(resource)}"

    def _get_submit_source_lines(
        common: str, comment: str, resource: str, agent: str
    ) -> str:
        return f"{common}\n\n{comment}\n\n{resource}\n\n{agent}"

    common_source_lines = inspect.getsource(common)
    comment_source_lines = _get_comment_source_lines(agent.comment)
    resource_source_lines = _get_resource_source_lines(agent.resource)
    return try_to_str(agent).map(
        lambda agent_source_lines: _get_submit_source_lines(
            common_source_lines,
            comment_source_lines,
            resource_source_lines,
            agent_source_lines,
        )
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
        name: Optional[str] = None,
    ) -> Agent:
        nonlocal _registry
        return _registry[key](resource, comment, name)

    return _register_agent_constructor, _construct_agent


register, construct = _init_agent_registry()
