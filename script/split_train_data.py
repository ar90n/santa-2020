# %%
import sys
import pandas as pd
from pathlib import Path

# %%
path_to_parquet = Path(sys.argv[1])
n_folds = int(sys.argv[2])
df = pd.read_parquet(path_to_parquet)
# %%
fold_size = int(df["id"].max() / n_folds)
folds = []
for i in range(n_folds):
    beg = i * fold_size
    end = (i + 1) * fold_size - 1
    folds.append(df[df["id"].between(beg, end, inclusive=True)])

# %%
for i, fold in enumerate(folds):
    out_path = path_to_parquet.parent / f"{path_to_parquet.name}_{i}_{n_folds-1}.parquet"
    fold.to_parquet(out_path)