# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#%load_ext autoreload
#%autoreload 2

#%%
import os
from santa_2020 import simulate, agents, io

#%%
target_agents = [agents.get(os.environ["AGENT_KEY"])]
enemy_agents = [agents.get("random"), agents.get("round_robin")]
results = simulate.run(target_agents, enemy_agents).unwrap()

#%%
for k, ret in results.items():
    print(f"{k[0]} vs {k[1]}")
    ret.env.render(mode="ipython", width=800, height=500)

# %%
io.save_submit(target_agents[0])
