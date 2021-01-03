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
import sys
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import pickle
import sklearn.tree as skt
from omegaconf import OmegaConf
from dataclasses import MISSING, dataclass, field

#%%
@dataclass
class TableModelConfig:
    data_file: str = MISSING
    train_feats: List[str] = MISSING
    target: str = MISSING
    model_name: str = MISSING
    train_params: Dict[Any, Any] = field(default_factory=dict)


#%%
try:
    yaml_path = sys.argv[1]
except IndexError:
    print(
        "usage: python train_decision_tree.py <path to configguraiton yaml>",
        file=sys.stderr,
    )
    sys.exit(-1)

conf = OmegaConf.create(TableModelConfig(**OmegaConf.load(yaml_path)))

#%%
def make_model():
    """Builds a decision tree model based on stored trainingd data"""
    data = pd.read_parquet(conf.data_file)
    model = skt.DecisionTreeRegressor(**conf.train_params)
    model.fit(data[conf.train_feats], data[conf.target])
    return model


#%%
model = make_model()
Path(f"{conf.model_name}.pickle").write_bytes(pickle.dumps(model))
