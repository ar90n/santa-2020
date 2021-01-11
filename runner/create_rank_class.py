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
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from dataclasses import MISSING, dataclass, field
from santa_2020 import io, util

from omegaconf import OmegaConf
import catboost


#%%
@dataclass
class TableModelConfig:
    train_data_folds: List[str] = MISSING
    val_data_folds: List[str] = MISSING
    train_feats: List[str] = MISSING
    target: str = MISSING
    model_name: str = MISSING
    classifier_model: str  = MISSING
    model_type: Optional[str] = "CatBoostRegressor"
    construct_params: Dict[Any, Any] = field(default_factory=dict)
    train_params: Dict[Any, Any] = field(default_factory=dict)


#%%
train_config_path = util.get_my_data_dir() / os.environ["TRAIN_CONFIG_FILENAME"]
conf = OmegaConf.create(TableModelConfig(**OmegaConf.load(train_config_path)))


#%%
train_df, val_df, = io.load_dataset(conf.train_data_folds, conf.val_data_folds)

from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.load_model(conf.classifier_model)
t = [l for l in conf.train_feats if l != "rank_class"]
train_df["rank_class"] = classifier.predict(train_df[t])
val_df["rank_class"] = classifier.predict(val_df[t])

# %%
import pandas as pd
train_data_concat = pd.concat([train_df, val_df], axis=0, ignore_index=True)
#%%
train_data_concat.to_parquet("/kaggle/input/my-santa-2020-data/top_agents_training_data_210109_3.parquet")