from __future__ import annotations

from functools import wraps
from typing import Callable, Optional
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


class BanditStats:
    _bandits: list[Bandit]
    _self_actions: list[int]
    _opp_actions: list[int]
    _decay_rate: float
    _reward: float
    _step: int

    def __init__(self, conf: Configuration) -> None:
        self._bandits = [Bandit(0, 0, 0) for _ in range(conf.banditCount)]
        self._decay_rate = conf.decayRate
        self._reward = 0.0
        self._step = 0

    def update(self, obs: Observation) -> None:
        if obs.step == 0:
            return

        self._step = obs.step
        last_reward = obs.reward - self.reward
        self._reward = obs.reward
        self_last_action, opp_last_action = _get_last_actions(obs)
        self._update_by_self_action(self_last_action, last_reward)
        self._update_by_opp_action(opp_last_action)

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
    def reward(self) -> float:
        return self._reward

    @property
    def step(self) -> int:
        return self._step


def bandit_stats(
    agent_func: Callable[[BanditStats], int]
) -> Callable[[Observation, Configuration], int]:
    stats: Optional[BanditStats] = None

    @wraps(agent_func)
    def wrapper(obs: Observation, conf: Configuration) -> int:
        nonlocal stats
        if stats is None:
            stats = BanditStats(conf)
        else:
            stats.update(obs)

        return agent_func(stats)

    return wrapper


# Add agent function here
