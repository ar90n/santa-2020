from __future__ import annotations

from .common import *
from .agent import Agent, register

AGENT_KEY = "greedy_decision_tree_6"

# from https://www.kaggle.com/lebroschar/1000-greedy-decision-tree-model
@context
def agent(stats: BanditStats, env: Environment) -> int:
    def _get_reg_feature(feature):
        return [feature[i] for i in [0, 3, 7, 8, 9, 11, 12, 13, 14]]

    import math
    import numpy as np
    import pandas as pd
    import catboost

    reg_model = env.resource["reg_model"]
    cls_model = env.resource["cls_model"]
    if stats.step == 0:
        features = []
        for i, bandit in enumerate(stats.bandits):
            feature = [0] * 15
            feature[0] = i
            feature[1] = bandit.n_self_pulls
            feature[2] = bandit.n_wins
            feature[3] = bandit.n_opp_pulls
            feature[4] = 0.0  # conf_pulls
            feature[5] = 0.0  # conf_success
            feature[6] = 0.0  # conf_success_rate
            feature[7] = feature[1] + feature[3]
            feature[8] = feature[1] - feature[2]
            feature[9] = feature[2] / (feature[1] + 1e-12)
            feature[10] = feature[1] / (feature[3] + 1e-12)
            feature[11] = feature[1] / (feature[0] + 1e-12)
            feature[12] = feature[3] / (feature[0] + 1e-12)
            feature[13] = np.power(0.97, feature[7])
            feature[14] = np.ravel(cls_model.predict(np.array(feature[:14])))[0]
            features.append(feature)
        columns = [
            "round_num",
            "n_pulls_opp",
            "n_pulls",
            "n_loss_self",
            "success_rate",
            "pull_rate_self",
            "pull_rate_opp",
            "decay",
            "rank_class",
        ]
        df = pd.DataFrame([_get_reg_feature(f) for f in features], columns=columns)
        pool = catboost.Pool(
            data=df,
            cat_features=["rank_class"],
        )
        env.store["predicts"] = reg_model.predict(pool)
        env.store["conf_pulls"] = [0] * len(stats.bandits)
        env.store["conf_success"] = [0] * len(stats.bandits)
    else:
        predicts = env.store["predicts"]
        conf_pulls = env.store["conf_pulls"]
        conf_success = env.store["conf_success"]
        raw_feature = [
            [
                stats.step,
                stats.bandits[stats.last_opp_action].n_self_pulls,
                stats.bandits[stats.last_opp_action].n_wins,
                stats.bandits[stats.last_opp_action].n_opp_pulls,
                0,
                0,
                0,
                stats.bandits[stats.last_opp_action].n_self_pulls
                + stats.bandits[stats.last_opp_action].n_opp_pulls,
                stats.bandits[stats.last_opp_action].n_self_pulls
                - stats.bandits[stats.last_opp_action].n_wins,
                stats.bandits[stats.last_opp_action].n_wins
                / (stats.bandits[stats.last_opp_action].n_self_pulls + 1e-12),
                stats.bandits[stats.last_opp_action].n_self_pulls
                / (stats.bandits[stats.last_opp_action].n_opp_pulls + 1e-12),
                stats.bandits[stats.last_opp_action].n_self_pulls
                / (stats.step + 1e-12),
                stats.bandits[stats.last_opp_action].n_opp_pulls / (stats.step + 1e-12),
                math.pow(
                    0.97,
                    stats.bandits[stats.last_opp_action].n_self_pulls
                    + stats.bandits[stats.last_opp_action].n_opp_pulls,
                ),
            ],
            [
                stats.step,
                stats.bandits[stats.last_self_action].n_self_pulls,
                stats.bandits[stats.last_self_action].n_wins,
                stats.bandits[stats.last_self_action].n_opp_pulls,
                conf_pulls[stats.last_self_action],
                conf_success[stats.last_self_action],
                conf_pulls[stats.last_self_action]
                / (conf_success[stats.last_self_action] + 1e-12),
                stats.bandits[stats.last_self_action].n_self_pulls
                + stats.bandits[stats.last_self_action].n_opp_pulls,
                stats.bandits[stats.last_self_action].n_self_pulls
                - stats.bandits[stats.last_self_action].n_wins,
                stats.bandits[stats.last_self_action].n_wins
                / (stats.bandits[stats.last_self_action].n_self_pulls + 1e-12),
                stats.bandits[stats.last_self_action].n_self_pulls
                / (stats.bandits[stats.last_self_action].n_opp_pulls + 1e-12),
                stats.bandits[stats.last_self_action].n_self_pulls
                / (stats.step + 1e-12),
                stats.bandits[stats.last_self_action].n_opp_pulls
                / (stats.step + 1e-12),
                math.pow(
                    0.97,
                    stats.bandits[stats.last_self_action].n_self_pulls
                    + stats.bandits[stats.last_self_action].n_opp_pulls,
                ),
            ],
        ]
        conf_pulls[stats.last_opp_action] = 0
        conf_success[stats.last_opp_action] = 0
        conf_pulls[stats.last_self_action] += 1
        conf_success[stats.last_self_action] += stats.last_reward

        rank_classes = np.ravel(cls_model.predict(np.array(raw_feature)))
        raw_feature[0].append(rank_classes[0])
        raw_feature[1].append(rank_classes[1])

        columns = [
            "round_num",
            "n_pulls_opp",
            "n_pulls",
            "n_loss_self",
            "success_rate",
            "pull_rate_self",
            "pull_rate_opp",
            "decay",
            "rank_class",
        ]
        df = pd.DataFrame([_get_reg_feature(f) for f in raw_feature], columns=columns)
        pool = catboost.Pool(
            data=df,
            cat_features=["rank_class"],
        )
        predicts[[stats.last_opp_action, stats.last_self_action]] = reg_model.predict(pool)

    est_return = env.store["predicts"]
    fudge_factor = env.resource.get("fudge_factor", 0.99)
    thresold = fudge_factor * np.max(est_return)
    return int(np.random.choice(np.where(est_return >= thresold)[0]))


register(
    AGENT_KEY,
    lambda resource, comment, name: Agent(AGENT_KEY, agent, resource, comment, name),
)

