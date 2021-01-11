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
import sys
from typing import List, Dict, Any
from pathlib import Path
import pickle
from dataclasses import MISSING, dataclass, field
from santa_2020 import io, util

from omegaconf import OmegaConf
from catboost import CatBoostRegressor, Pool


#%%
@dataclass
class TableModelConfig:
    train_data_folds: List[str] = MISSING
    val_data_folds: List[str] = MISSING
    train_feats: List[str] = MISSING
    target: str = MISSING
    model_name: str = MISSING
    construct_params: Dict[Any, Any] = field(default_factory=dict)
    train_params: Dict[Any, Any] = field(default_factory=dict)


#%%
train_config_path = util.get_my_data_dir() / os.environ["TRAIN_CONFIG_FILENAME"]
conf = OmegaConf.create(TableModelConfig(**OmegaConf.load(train_config_path)))

#%%
def make_sklearn_api_model(train_df, val_df, conf):
    model = CatBoostRegressor(**conf.construct_params)
    model.fit(
        train_df[conf.train_feats],
        train_df[conf.target],
        eval_set=(val_df[conf.train_feats], val_df[conf.target]),
        **conf.train_params,
    )
    return model


#%%
train_df, val_df, = io.load_dataset(conf.train_data_folds, conf.val_data_folds)
model = pickle.loads(Path("/kaggle/input/my-santa-2020-data/top_agents_catboost_regression_model_add_feats_depth_6.pickle").read_bytes())

#%%
Path(f"before.pickle").write_bytes(pickle.dumps(model))

# %%
pool = Pool(val_df[conf.train_feats], val_df[conf.target])
feature_importances = model.get_feature_importance(pool)
feature_names = train_df[conf.train_feats].columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


#%%
Path(f"after.pickle").write_bytes(pickle.dumps(model))

