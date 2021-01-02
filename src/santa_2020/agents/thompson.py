from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "thompson"

#from https://www.kaggle.com/aatiffraz/a-beginner-s-approach-to-performance-analysis
@context
def agent(stats: BanditStats, env: Environment) -> int:
    import math
    import random
    def calc_score(bandit: Bandit) -> float:
        n_selections = bandit.n_self_pulls
        return random.betavariate(bandit.n_wins + 1, n_selections - bandit.n_wins + 1)

    if stats.step < len(stats):
        return stats.step
    else:
        _, best_idx = max([(calc_score(bandit), i) for i, bandit in enumerate(stats.bandits)])
        return best_idx

register(AGENT_KEY, lambda resource, comment: Agent(AGENT_KEY, agent, resource, comment))