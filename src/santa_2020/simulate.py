from __future__ import annotations

from itertools import product, chain
from typing import Any
import datetime
from kaggle_environments import make
from dataclasses import dataclass
from typing import Any, Set

from returns.result import Success, safe
from .agents import Agent, try_to_submit_source
from returns.iterables import Fold
from concurrent.futures import ProcessPoolExecutor


@dataclass
class SimulateResult:
    duration: float
    result: dict
    env: Any


def _run_inner(lh_source: str, rh_source: str) -> SimulateResult:
    env = make("mab", debug=True)
    start_time = datetime.datetime.now()
    result = env.run([lh_source, rh_source])
    duration = datetime.datetime.now() - start_time

    return SimulateResult(duration, result, env)


def _swap_agents_if_need(lh: Agent, rh: Agent) -> tuple[Agent, Agent]:
    if rh.key < lh.key:
        return rh, lh
    return lh, rh


def _get_result_key(lh: Agent, rh: Agent) -> tuple[str, str]:
    lh_key = lh.key
    if lh.name is not None:
        lh_key = f"{lh_key}_{lh.name}"

    rh_key = rh.key
    if rh.name is not None:
        rh_key = f"{rh_key}_{rh.name}"

    return (lh_key, rh_key)


@safe
def run(target_agents: Set[Agent], enemy_agents: Set[Agent] = set()):
    agent_key_sources = (
        try_to_submit_source(agent).map(lambda sources: (agent.key, sources))
        for agent in chain(target_agents, enemy_agents)
    )
    source_map = (
        Fold.loop(
            agent_key_sources,
            Success([]),
            lambda key_source: lambda acc: acc + [key_source],
        )
        .map(lambda key_sources: dict(key_sources))
        .unwrap()
    )

    result_futures = {}
    with ProcessPoolExecutor() as executor:
        for lh, rh in product(target_agents, chain(target_agents, enemy_agents)):
            if lh == rh:
                continue

            lh, rh = _swap_agents_if_need(lh, rh)
            result_key = _get_result_key(lh, rh)
            lh_source = source_map[lh.key]
            rh_source = source_map[rh.key]
            result_futures[result_key] = executor.submit(
                _run_inner, lh_source, rh_source
            )

    return {k: f.result() for k, f in result_futures.items()}
