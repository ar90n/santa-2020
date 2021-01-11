from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "greedy_decision_tree_5"

# from https://www.kaggle.com/lebroschar/1000-greedy-decision-tree-model
@context
def agent(stats: BanditStats, env: Environment) -> int:
    import math
    import numpy as np

    model = env.resource["model"]
    if stats.step == 0:
        features = np.zeros((len(stats.bandits), 14))
        for i, bandit in enumerate(stats.bandits):
            features[i, 1] = bandit.n_self_pulls
            features[i, 2] = bandit.n_wins
            features[i, 3] = bandit.n_opp_pulls
        features[:, 4] = 0 #conf_pulls
        features[:, 5] = 0 #conf_success
        features[:, 6] = 0 #conf_success_rate
        features[:, 7] = features[:, 1] + features[:, 3]
        features[:, 8] = features[:, 1] - features[:, 2]
        features[:, 9] = features[:, 2] / (features[:, 1] + 1e-12)
        features[:, 10] = features[:, 1] / (features[:, 3] + 1e-12)
        features[:, 11] = features[:, 1] / (features[:, 0] + 1e-12)
        features[:, 12] = features[:, 3] / (features[:, 0] + 1e-12)
        features[:, 13] = np.power(0.97, features[:, 7])
        env.store["predicts"] = model.predict(features)
        env.store["conf_pulls"] = [0] * len(stats.bandits)
        env.store["conf_success"] = [0] * len(stats.bandits)
    else:
        predicts = env.store["predicts"]
        conf_pulls = env.store["conf_pulls"]
        conf_success = env.store["conf_success"]
        conf_pulls[stats.last_opp_action] = 0
        conf_success[stats.last_opp_action] = 0
        conf_pulls[stats.last_self_action] += 1
        conf_success[stats.last_self_action] += stats.last_reward
        predicts[[stats.last_opp_action, stats.last_self_action]] += model.predict(
            [
                [
                    stats.step,
                    stats.bandits[stats.last_opp_action].n_self_pulls,
                    stats.bandits[stats.last_opp_action].n_wins,
                    stats.bandits[stats.last_opp_action].n_opp_pulls,
                    0,
                    0,
                    0,
                    stats.bandits[stats.last_opp_action].n_self_pulls + stats.bandits[stats.last_opp_action].n_opp_pulls,
                    stats.bandits[stats.last_opp_action].n_self_pulls - stats.bandits[stats.last_opp_action].n_wins,
                    stats.bandits[stats.last_opp_action].n_wins / (stats.bandits[stats.last_opp_action].n_self_pulls + 1e-12),
                    stats.bandits[stats.last_opp_action].n_self_pulls / (stats.bandits[stats.last_opp_action].n_opp_pulls + 1e-12),
                    stats.bandits[stats.last_opp_action].n_self_pulls / (stats.step + 1e-12),
                    stats.bandits[stats.last_opp_action].n_opp_pulls / (stats.step + 1e-12),
                    math.pow(0.97, stats.bandits[stats.last_opp_action].n_self_pulls + stats.bandits[stats.last_opp_action].n_opp_pulls)
                ],
                [
                    stats.step,
                    stats.bandits[stats.last_self_action].n_self_pulls,
                    stats.bandits[stats.last_self_action].n_wins,
                    stats.bandits[stats.last_self_action].n_opp_pulls,
                    conf_pulls[stats.last_self_action],
                    conf_success[stats.last_self_action],
                    conf_pulls[stats.last_self_action] / (conf_success[stats.last_self_action] + 1e-12),
                    stats.bandits[stats.last_self_action].n_self_pulls + stats.bandits[stats.last_self_action].n_opp_pulls,
                    stats.bandits[stats.last_self_action].n_self_pulls - stats.bandits[stats.last_self_action].n_wins,
                    stats.bandits[stats.last_self_action].n_wins / (stats.bandits[stats.last_self_action].n_self_pulls + 1e-12),
                    stats.bandits[stats.last_self_action].n_self_pulls / (stats.bandits[stats.last_self_action].n_opp_pulls + 1e-12),
                    stats.bandits[stats.last_self_action].n_self_pulls / (stats.step + 1e-12),
                    stats.bandits[stats.last_self_action].n_opp_pulls / (stats.step + 1e-12),
                    math.pow(0.97, stats.bandits[stats.last_self_action].n_self_pulls + stats.bandits[stats.last_self_action].n_opp_pulls)
                ],
            ]
        )
        predicts[[stats.last_opp_action, stats.last_self_action]] /= 2.0

    est_return = env.store["predicts"]
    fudge_factor = env.resource.get("fudge_factor", 0.99)
    thresold = fudge_factor * np.max(est_return)
    return int(np.random.choice(np.where(est_return >= thresold)[0]))


register(AGENT_KEY, lambda resource, comment, name: Agent(AGENT_KEY, agent, resource, comment, name))