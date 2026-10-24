###################################################################
#
# --- SCRIPT: data_wrangling.py ---
#
# GOAL: execute all needed transformations and feature creations.
#
# INPUT: none
#
# OUTPUT:
#   1. feature_engineering function which returns a new Polars
#      DataFrame (data) with modified/new feature.
#
# VERSION: 2.0 (Dual-Use Architecture for EDA & Modeling)
# CHANGES: Refactored to be target-agnostic by replacing rigid .select() with
#          .drop(strict=False) and using polars.selectors for conditional target
#          casting when 'is_fraud' is absent.
#
###################################################################


# Feature engineering

import polars as pl
import polars.selectors as cs

# Note that polars.selectors ("cs") is used for target-feature "agnostic"
# transformations.
# Since there is a transformation to do (casting to boolean), cs allows conditional
# operations (only if "is_fraud" if present).
# This is a sefeguard if this script will be re-used during raw data preprocessing
# pipeline creation; at that stage the pipeline (data wrangling/feature engineering +
# column transforer + modeling) will be applied to X_train dataset which will NOT 
# include target variable since scikit need the dataset to be split in X_train and 
# y_train. 
# This code is reusable without throwing errors when operating on target-free feature
# dataset (X).


def feature_engineering(df_polars: pl.DataFrame):
    """
    Runs a routine to adapt existing features and to create new ones.

    Args:
        df_polars (pl.DataFrame): The input data (data_origin) in
        Polars format.
        Note. Added ": pl.DataFrame" is a Type Hint to enable
        Positron's autocompletion and ensure the function strictly
        receives a Polars object, even though it's not mandatory
        for Python execution.

    Returns:
        df_polars: a new Polars DataFrame (data) with modified/new
        feature.
    """

    data = (
        df_polars
        # data type manipulation
        .with_columns(
            # Cast a Boolean ONLY if "is_fraud" is present 
            # (require_all=False)
            cs.by_name("is_fraud", require_all=False).cast(pl.Boolean),
            (pl.col("sn") + 1).alias("id_transaction"),
            pl.col("trans_date_trans_time").str.to_datetime("%d-%m-%Y %H:%M"),
            pl.col("dob").str.to_date("%d-%m-%Y").alias("cardholder_birthday"),
            pl.col("category").cast(pl.Categorical),
            pl.col("gender").cast(pl.Categorical),
            pl.col("merchant").str.slice(6),
        )
        # splitting timestamps into date and time
        .with_columns(
            [
                pl.col("trans_date_trans_time").dt.date().alias("trans_date"),
                pl.col("trans_date_trans_time").dt.time().alias("trans_time"),
            ]
        )
        .with_columns(
            [
                pl.col("trans_date").dt.day().alias("day"),
                pl.col("trans_date").dt.month().alias("month"),
                pl.col("trans_date").dt.weekday().alias("weekday"),
                pl.col("trans_time").dt.hour().alias("hour"),
            ]
        )
        # create bins for age and amount
        .with_columns(
            [
                (
                    (
                        pl.col("trans_date") - pl.col("cardholder_birthday")
                    ).dt.total_days()
                    / 365
                )
                .cast(pl.Int32)
                .alias("age")
            ]
        )
        .with_columns(
            [
                pl.col("age")
                .cut(
                    breaks=[18, 25, 35, 50, 65],
                    labels=["Under 18", "18-25", "26-35", "36-50", "51-65", "Over 65"],
                )
                .cast(pl.Enum(["Under 18", "18-25", "26-35", "36-50", "51-65", "Over 65"]))
                .alias("age_bins"),
                pl.col("amt")
                .cut(
                    breaks=[10, 50, 100, 500],
                    labels=[
                        "Micro (<10)",
                        "Small (10-50)",
                        "Medium (51-100)",
                        "Large (101-500)",
                        "Extra Large (>500)",
                    ],
                )
                .alias("amount_bins"),
            ]
        )
        .with_columns(
            job_count = pl.col("job").count().over("job")
        )
        .with_columns(
            job_aggregate =
            pl.when(pl.col("job_count") < 25)
            .then(pl.lit("other jobs"))
            .otherwise(pl.col("job"))
    )
        .drop(
            [
                "sn",
                "trans_date_trans_time",
                "cc_num",
                "first",
                "last",
                "lat",
                "long",
                "trans_num",
                "merch_lat",
                "merch_long",
                "zip",
                "dob",
                "unix_time",
                "job",
                "job_count"
            ],
            strict=False
        )
        .rename({"amt": "amount"})
    )
    return data  # note that return comand is part and parcel of the function


# Note that Feature Exclusion & Column Ordering Strategy needs to follow
# "adjusted" choices:
#
# 1) .select() is not used and replaced with .drop(..., strict=False) to ensure 
# dual-use flexibility across both EDA and Scikit-Learn pipelines. If .select do not
# find the column returns an error, while .drop can handle that with strict=False.
#
# 2) Column Order: Switching from .select() to .drop() means column ordering is no
# longer explicitly governed or enforced (e.g., forcing target 'is_fraud' to the end).
# This is a deliberate trade-off: machine learning pipelines (and ColumnTransformers)
# select features by name rather than position, making strict visual ordering
# redundant. Using .drop(strict=False) safely removes unwanted raw columns
# while preventing pipeline crashes when processing variable column sets (X vs. X+y).