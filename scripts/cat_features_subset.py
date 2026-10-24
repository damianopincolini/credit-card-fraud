###################################################################
#
# --- SCRIPT: cat_featues_subset.py ---
#
# GOAL: Create a specific sub-dataset with categorical features 
#       only
#
# INPUT:
#   1. data_train
#
# OUTPUT:
#   1. data_train_cat: includes target variable and new feature
#      "job risk".
#
###################################################################


import polars as pl
import polars.selectors as cs
from project_config import paths

# define the train dataset with categorical features only.
# Define jos risk feature (for train set only to avoid data leakage
# since the new feature is based on target variable).def

data_train = (
    pl.read_parquet(
        paths["processed_data_dir"] / "data_train_eda.parquet"
    )
)

data_train_cat = (
    data_train
    .select([
        cs.string() | cs.categorical() | cs.enum(),
        cs.by_name("is_fraud")   
    ])
)

