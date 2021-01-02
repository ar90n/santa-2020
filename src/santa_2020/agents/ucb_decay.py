from __future__ import annotations
from .common import *
from .agent import Agent, register

#from https://www.kaggle.com/xhlulu/santa-2020-ucb-and-bayesian-ucb-starter
@bandit_stats
def agent(stats: BanditStats) -> int:
    import math
    def calc_score(bandit: Bandit) -> float:
        n_selections = bandit.n_self_pulls
        avg_reward = stats.decay_rate * bandit.n_wins / float(n_selections)
        delta_i =  math.sqrt(2 * math.log(stats.step) / n_selections)
        return avg_reward + delta_i

    if stats.step < len(stats):
        return stats.step
    else:
        _, best_idx = max([(calc_score(bandit), i) for i, bandit in enumerate(stats.bandits)])
        return best_idx

register(Agent("ucb_decay", agent))