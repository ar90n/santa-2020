from __future__ import annotations

import pickle
from io import BytesIO
import base64
from functools import wraps
from typing import Callable, Optional, Any
from dataclasses import dataclass
from kaggle_environments.envs.mab.mab import Configuration, Observation


@dataclass
class Bandit:
    n_self_pulls: int
    n_opp_pulls: int
    n_wins: int


def _calc_agent_indice(agent_index: int) -> tuple[int, int]:
    self_idx = agent_index
    opp_idx = (self_idx + 1) % 2
    return (self_idx, opp_idx)


def _get_last_actions(obs: Observation) -> tuple[int, int]:
    return tuple([obs.lastActions[idx] for idx in _calc_agent_indice(obs.agentIndex)])


def serialize(obj: Any) -> str:
    return base64.b64encode(pickle.dumps(obj))


def deserialize(serialized: str) -> Any:
    return pickle.loads(base64.b64decode(serialized))


class Environment:
    _resource: dict[str, Any]
    _store: dict[Any, Any]

    def __init__(self, resource: dict[str, Any]) -> None:
        self._store = {}
        self._resource = resource

    @property
    def resource(self) -> dict[str, Any]:
        return self._resource

    @property
    def store(self) -> dict[Any, Any]:
        return self._store


class BanditStats:
    _bandits: list[Bandit]
    _last_self_action: Optional[int]
    _last_opp_action: Optional[int]
    _last_reward: Optional[float]
    _decay_rate: float
    _reward: float
    _step: int

    def __init__(self, conf: Configuration) -> None:
        self._bandits = [Bandit(0, 0, 0) for _ in range(conf.banditCount)]
        self._last_self_action = None
        self._last_opp_action = None
        self._last_reward = None
        self._decay_rate = conf.decayRate
        self._reward = 0.0
        self._step = 0

    def update(self, obs: Observation) -> None:
        if obs.step == 0:
            return

        self._step = obs.step
        self._last_reward = obs.reward - self.reward
        self._reward = obs.reward
        self._last_self_action, self._last_opp_action = _get_last_actions(obs)
        self._update_by_self_action(self._last_self_action, self._last_reward)
        self._update_by_opp_action(self._last_opp_action)

    def _update_by_self_action(self, action: int, reward: float) -> None:
        bandit = self._bandits[action]
        bandit.n_self_pulls += 1
        bandit.n_wins += int(0 < reward)

    def _update_by_opp_action(self, action: int) -> None:
        bandit = self._bandits[action]
        bandit.n_opp_pulls += 1

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def bandits(self) -> list[Bandit]:
        return [x for x in self._bandits]

    @property
    def last_self_action(self) -> Optional[int]:
        return self._last_self_action

    @property
    def last_opp_action(self) -> Optional[int]:
        return self._last_opp_action

    @property
    def last_reward(self) -> Optional[float]:
        return self._last_reward

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def step(self) -> int:
        return self._step

    def __len__(self) -> int:
        return len(self._bandits)


def context(
    agent_func: Callable[[BanditStats, Environment], int]
) -> Callable[[Observation, Configuration], int]:
    stats: Optional[BanditStats] = None
    env = Environment(deserialize(_RESOURCE))

    @wraps(agent_func)
    def wrapper(obs: Observation, conf: Configuration) -> int:
        nonlocal stats
        if stats is None:
            stats = BanditStats(conf)
        else:
            stats.update(obs)

        return agent_func(stats, env)

    return wrapper

_RESOURCE = "gAN9cQAu"

# Add agent function here
