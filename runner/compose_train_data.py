#%%
import json
from pathlib import Path
import pandas as pd
#%%
SAMPLE_DATA_PATH = Path("/") / "kaggle" / "input" / "sample-training-data" / "training_data_201223.parquet"
TOP_AGENT_DATA_DIR = Path("/") / "kaggle" / "input" / "santa-2020-top-agents-dataset" / "episode"
#%%
sample_data = pd.read_parquet(SAMPLE_DATA_PATH)
# %%
for json_path in TOP_AGENT_DATA_DIR.glob("*.json"):
    data = json.loads(json_path.read_text())
    break
# %%
list(data["steps"][0][1].keys())
# %%
data["steps"][0][1]['status']
                                   # %%
# %%
list(data.kys())
# %%
data["steps"][0][0]
# %%
payout = [p / 100.0 for p in data["steps"][0][0]["observation"]["thresholds"]]
round_num = [0] * len(payout)
machine_id = list(range(len(payout)))
agent_id = [-1] * len(payout)
n_pulls_self = [0] * len(payout)
n_success_self = [0] * len(payout)
n_pulls_opp = [0] * len(payout)
rewards = [[0] * 2 for _ in range(len(payout))]


acc_n_pulls_self = [0] * len(payout)
acc_n_success_self = [0] * len(payout)
acc_n_pulls_opp = [0] * len(payout)

# %%
for log in data["steps"][1:]:
    last_rewards = rewards[-1]
    observation = log[0]["observation"]
    for i, action in enumerate(observation["lastActions"]):
        agent_index = observation["agentIndex"]
        cur_reward = observation["rewards"] - last_rewards[agent_index]

        payout.append(observation["thredholds"][action] / 100.0)
        payout.append(observation["thredholds"][action] / 100.0)
        round_num.append(observation["step"])
        round_num.append(observation["step"])
        machine_id.append(action)
        machine_id.append(action)
        agent_id.append(agent_index)
        agent_id.append(agent_index)

        n_pulls_self.append(acc_n_pulls_)
        n_success_self.append()
        n_pulls_opp.append()

        n_pulls_self.append()
        n_pulls_opp.append()
# %%
sample_data.iloc[90:105]['action']
# %%

sample_data.iloc[0:15]
# %%
data["steps"][1][0]["observation"]
# %%

data["steps"][100][1]
# %%
data
# %%
