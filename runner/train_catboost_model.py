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
import numpy as np

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
    model_type: Optional[str] = "CatBoostRegressor"
    construct_params: Dict[Any, Any] = field(default_factory=dict)
    train_params: Dict[Any, Any] = field(default_factory=dict)


#%%
train_config_path = util.get_my_data_dir() / os.environ["TRAIN_CONFIG_FILENAME"]
conf = OmegaConf.create(TableModelConfig(**OmegaConf.load(train_config_path)))

#%%
def make_sklearn_api_model(train_df, val_df, conf):
    model_type = conf.model_type
    model = getattr(catboost, model_type)(**conf.construct_params)

    categorical_features_indices = np.where(
        (train_df[conf.train_feats].dtypes != np.float)
        & (train_df[conf.train_feats].dtypes != np.int)
    )[0]
    model.fit(
        train_df[conf.train_feats],
        train_df[conf.target],
        cat_features=categorical_features_indices,
        eval_set=(val_df[conf.train_feats], val_df[conf.target]),
        **conf.train_params,
    )
    return model


#%%
train_df, val_df, = io.load_dataset(conf.train_data_folds, conf.val_data_folds)
model = make_sklearn_api_model(train_df, val_df, conf)

#%%
model.save_model(f"{conf.model_name}.cbm")

# %%
categorical_features_indices = np.where(
    (train_df[conf.train_feats].dtypes != np.float)
    & (train_df[conf.train_feats].dtypes != np.int)
)[0]
pool = catboost.Pool(
    val_df[conf.train_feats],
    val_df[conf.target],
    cat_features=categorical_features_indices,
)
feature_importances = model.get_feature_importance(pool)
feature_names = train_df[conf.train_feats].columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print("{}: {}".format(name, score))
