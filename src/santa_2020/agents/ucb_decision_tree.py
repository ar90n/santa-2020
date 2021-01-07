from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "ucb_decision_tree"

# from https://www.kaggle.com/lebroschar/1000-greedy-decision-tree-model
@context
def agent(stats: BanditStats, env: Environment) -> int:
    import numpy as np

    model = env.resource["model"]
    if stats.step == 0:
        features = np.zeros((len(stats.bandits), 4))
        for i, bandit in enumerate(stats.bandits):
            features[i, 1] = bandit.n_self_pulls
            features[i, 2] = bandit.n_wins
            features[i, 3] = bandit.n_opp_pulls
        env.store["selections"] = np.zeros(len(stats.bandits))
        env.store["predicts"] = model.predict(features)
    else:
        selections = env.store["selections"]
        predicts = env.store["predicts"]
        predicts[[stats.last_opp_action, stats.last_self_action]] = model.predict(
            [
                [
                    stats.step,
                    stats.bandits[stats.last_opp_action].n_self_pulls,
                    stats.bandits[stats.last_opp_action].n_wins,
                    stats.bandits[stats.last_opp_action].n_opp_pulls,
                ],
                [
                    stats.step,
                    stats.bandits[stats.last_self_action].n_self_pulls,
                    stats.bandits[stats.last_self_action].n_wins,
                    stats.bandits[stats.last_self_action].n_opp_pulls,
                ],
            ]
        )
        selections[stats.last_self_action] += 1
        selections[stats.last_opp_action] += 1

    selections = env.store["selections"]
    predicts = env.store["predicts"]
    delta = np.sqrt(np.log(stats.step) / (2 * selections + 1e-12))
    ucb = predicts + delta
    return int(np.argmax(ucb))


register(AGENT_KEY, lambda resource, comment, name: Agent(AGENT_KEY, agent, resource, comment, name))