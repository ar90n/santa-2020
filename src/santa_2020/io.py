from __future__ import annotations

from typing import Optional, Any
from returns.io import impure_safe, IOResultE
from .agents import Agent, try_to_submit_source
from pathlib import Path
import yaml
import pickle

from santa_2020 import agents


def load_agent_conf(config_path: Path) -> dict:
    def _parse_resource(org_resource: dict[str, Any]) -> dict[str, Any]:
        def _parse(key: str, value: Any) -> Any:
            if "@" in key:
                k, t = key.split("@")
                if t == "pickle":
                    content = pickle.loads(Path(value).read_bytes())
                    return (k, content)
                raise ValueError(f"Unknow value type:{t}")

            return (key, value)

        return dict(_parse(k, v) for k, v in org_resource.items())

    with config_path.open("r") as fp:
        conf = yaml.load(fp)

    agent = conf["agent"]
    agent["resource"] = _parse_resource(agent.get("resource", {}))
    agent["comment"] = agent.get("comment")
    return agent


def save_submit(agent: Agent, filename: Optional[str] = None) -> IOResultE[int]:
    if filename is None:
        filename = "submission.py"

    return try_to_submit_source(agent).bind(
        lambda source: impure_safe(Path(filename).write_text)(source)
    )
