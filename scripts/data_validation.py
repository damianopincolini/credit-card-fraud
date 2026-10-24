###################################################################
#
# --- SCRIPT: data_validation.py ---
#
# GOAL: create a function that validate data from the business
#       point of view.
#
# Note that validate library requires pandas (pointblank from
# Posit to be test further on)
#
# INPUT: none
#
# OUTPUT:
#   1. validate_data function.
#
###################################################################


import pointblank as pb
import polars as pl
import pandas as pd


def validate_data(df_polars: pl.DataFrame):
    """
    Runs a data quality interrogation on the transactions dataframe.

    Args:
        df_polars (pl.DataFrame): The input data in Polars format.
        Note. Added ": pl.DataFrame" is a Type Hint to enable
        Positron's autocompletion and ensure the function strictly
        receives a Polars object, even though it's not mandatory
        for Python execution.

    Returns:
        pb.Validate: The Pointblank agent with results.
    """

    # 1. Conversion for Pointblank compatibility
    df_pd = df_polars.to_pandas()

    # 2. Initialize the Agent
    # The 'label' will appear as the header of your report
    agent = pb.Validate(data=df_pd, label="Credit Card Fraud Detection - Quality Check")

    # 3. Define the Rules (The core logic)
    agent = (
        agent
        # Check: Amount must be positive
        .col_vals_gt(columns="amount", value=0, na_pass=True)
        # Check: is_fraud must be binary (0 or 1)
        .col_vals_not_null(columns="is_fraud")
        # Check: Coordinates should be within reasonable bounds
        # (lat and long won't be considered)
        # .col_vals_between(columns="lat", left=-90, right=90)
        # .col_vals_between(columns="long", left=-180, right=180)
        # Check: Essential IDs must not be missing
        # .col_vals_not_null(columns=["cc_num", "trans_num", "unix_time"])
        # Check: Population of the city should be a positive number
        .col_vals_ge(columns="city_pop", value=0)
        # Execute the validation
        .interrogate()
    )

    return agent
