from __future__ import annotations
from .common import *
from .agent import Agent, register

#from https://www.kaggle.com/xhlulu/santa-2020-ucb-and-bayesian-ucb-starter
@bandit_stats
def agent(stats: BanditStats) -> int:
    from scipy.stats import beta
    c = 3

    def calc_score(bandit: Bandit) -> float:
        n_selections = bandit.n_self_pulls
        return bandit.n_wins / float(n_selections) + beta.std(bandit.n_wins, n_selections - bandit.n_wins) * c

    if stats.step < len(stats):
        return stats.step
    else:
        _, best_idx = max([(calc_score(bandit), i) for i, bandit in enumerate(stats.bandits)])
        return best_idx

register(Agent("bayesian_ucb", agent))