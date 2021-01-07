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
import sys
from typing import List, Dict, Any
from pathlib import Path
import pickle
from omegaconf import OmegaConf
from dataclasses import MISSING, dataclass, field
from santa_2020 import io

#%%
@dataclass
class TableModelConfig:
    train_data_folds: List[str] = MISSING
    val_data_folds: List[str] = MISSING
    train_feats: List[str] = MISSING
    target: str = MISSING
    model_module: str = MISSING
    model_type: str = MISSING
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
def make_sklearn_api_model(train_df, conf):
    from importlib import import_module

    module = import_module(conf.model_module)
    model = getattr(module, conf.model_type)(**conf.train_params)
    model.fit(train_df[conf.train_feats], train_df[conf.target])
    return model


def mse(model, x, y) -> float:
    diff = y - model.predict(x)
    return float(diff @ diff) / len(y)


#%%
train_df, val_df, = io.load_dataset(conf.train_data_folds, conf.val_data_folds)
model = make_sklearn_api_model(train_df, conf)
# %%
loss = mse(model, val_df[conf.train_feats], val_df[conf.target])
print(loss)

#%%
Path(f"{conf.model_name}.pickle").write_bytes(pickle.dumps(model))