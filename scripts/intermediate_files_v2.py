###################################################################
#
# --- SCRIPT: intermediate_files.py ---
#
# GOAL: save intermediate train and test set for both EDA and
#       modeling in parquet format. Datasets for EDA are based
#       on post-feature engineering columns, while datasets for
#       modeling are based on raw data and the serve the end-to-
#       end logic
#
# INPUT: none
#
# OUTPUT:
#   1. save_train_test_eda_dataset and 
#      save_train_test_modeling_dataset
#      functions.
#
###################################################################


from pathlib import Path

import polars as pl


def save_train_test_eda_datasets(
    df_train_polars: pl.DataFrame,
    df_test_polars: pl.DataFrame,
    output_dir: Path
):
    # Defining of paths where output have to be saved
    train_output_path = output_dir / "data_train_eda.parquet"
    test_output_path = output_dir / "data_test_eda.parquet"

    # Saving files
    df_train_polars.write_parquet(train_output_path)
    df_test_polars.write_parquet(test_output_path)

def save_train_test_modeling_datasets(
    df_train_polars: pl.DataFrame,
    df_test_polars: pl.DataFrame,
    output_dir: Path
):
    # Defining of paths where output have to be saved
    train_output_path = output_dir / "data_train_modeling.parquet"
    test_output_path = output_dir / "data_test_modeling.parquet"

    # Saving files
    df_train_polars.write_parquet(train_output_path)
    df_test_polars.write_parquet(test_output_path)
