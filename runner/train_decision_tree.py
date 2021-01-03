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
# https://www.kaggle.com/lebroschar/1000-greedy-decision-tree-model
from pathlib import Path
import pandas as pd
import pickle
import sklearn.tree as skt

from santa_2020.util import get_input_dir

#%%
FUDGE_FACTOR = 0.99
VERBOSE = False
DATA_FILE = get_input_dir() / "sample-training-data" / "training_data_201223.parquet"
TRAIN_FEATS = ["round_num", "n_pulls_self", "n_success_self", "n_pulls_opp"]
TARGET_COL = "payout"

#%%
def make_model():
    """Builds a decision tree model based on stored trainingd data"""
    data = pd.read_parquet(DATA_FILE)
    model = skt.DecisionTreeRegressor(min_samples_leaf=40)
    model.fit(data[TRAIN_FEATS], data[TARGET_COL])
    return model


#%%
model = make_model()
Path("simple_decitoin_tree_regression_model.pickle").write_bytes(pickle.dumps(model))