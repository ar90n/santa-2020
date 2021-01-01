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
    return (lh.key, rh.key)


def _should_skip(
    result: dict[tuple[str, str], Any], result_key: tuple[str, str]
) -> bool:
    return result_key in result_key or result_key[0] == result_key[1]


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
            lh, rh = _swap_agents_if_need(lh, rh)
            result_key = _get_result_key(lh, rh)

            if _should_skip(result_futures, result_key):
                continue
            lh_source = source_map[result_key[0]]
            rh_source = source_map[result_key[1]]
            result_futures[result_key] = executor.submit(
                _run_inner, lh_source, rh_source
            )

    return {k: f.result() for k, f in result_futures.items()}
