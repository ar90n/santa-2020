from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "vegas_pull"

# from https://www.kaggle.com/sirishks/pull-vegas-slot-machines
@context
def agent(stats: BanditStats, env: Environment) -> int:
    import random

    def calc_score(bandit: Bandit) -> float:
        num = (
            2 * bandit.n_wins
            - bandit.n_self_pulls
            + bandit.n_opp_pulls
            - (bandit.n_opp_pulls > 0) * 1.5
        )
        den = bandit.n_self_pulls + bandit.n_opp_pulls
        return num / (den + 1e-12)

    if stats.step < 64:
        return random.randrange(len(stats.bandits) - 1)
    else:
        _, best_idx = max(
            [(calc_score(bandit), i) for i, bandit in enumerate(stats.bandits)]
        )
        return best_idx

register(
    AGENT_KEY, lambda resource, comment: Agent(AGENT_KEY, agent, resource, comment)
)
