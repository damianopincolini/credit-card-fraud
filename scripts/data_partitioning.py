###################################################################
#
# --- SCRIPT: data_partitioning.py ---
#
# GOAL: split dataset into train and test sets and check the
#       proportion of true/false target values out of the total for
#       overall, train and test dataset.
#
# INPUT: none
#
# OUTPUT:
#   1. data_splitting function.
#
###################################################################


import polars as pl
from sklearn.model_selection import train_test_split


def data_splitting(df_polars: pl.DataFrame):
    # Let's select the single target feature to split the entire dataset
    data_target = df_polars.select("is_fraud").to_series()

    # Let's create stratified index for both train and test set
    train_idx, test_idx = train_test_split(
        range(len(df_polars)), test_size=0.2, stratify=data_target, random_state=42
    )

    # Let's create train and test set
    data_train = df_polars[list(train_idx)]
    data_test = df_polars[list(test_idx)]

    # Check the proportion of the whole dataset target values in train and test sets
    prop_target_data = (
        df_polars.group_by("is_fraud")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("percentage")
        )
    )

    prop_target_data_train = (
        data_train.group_by("is_fraud")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("percentage")
        )
    )

    prop_target_data_test = (
        data_test.group_by("is_fraud")
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("percentage")
        )
    )
    return (
        data_train,
        data_test,
        prop_target_data,
        prop_target_data_train,
        prop_target_data_test,
    )
