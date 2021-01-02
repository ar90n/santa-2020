from __future__ import annotations
from .common import *
from .agent import Agent, register

# from https://www.kaggle.com/ilialar/simple-multi-armed-bandit
@bandit_stats
def agent(stats: BanditStats) -> int:
    import numpy as np

    if stats.step == 0:
        stats.store["state"] = np.ones((2, len(stats.bandits)))
    else:
        assert stats.last_self_action is not None
        assert stats.last_opp_action is not None
        assert stats.last_reward is not None

        state = stats.store["state"]
        idx = int(stats.last_reward == 0)
        state[idx, stats.last_self_action] += 1.0

        state[0, stats.last_self_action] = (
            state[0, stats.last_self_action] - 1.0
        ) * stats.decay_rate + 1
        state[0, stats.last_opp_action] = (
            state[0, stats.last_opp_action] - 1.0
        ) * stats.decay_rate + 1

    if stats.step < len(stats):
        return stats.step
    else:
        state = stats.store["state"]
        proba = np.random.beta(state[0], state[1])
        return int(np.argmax(proba))


register(Agent("multi_armed_bandit", agent))