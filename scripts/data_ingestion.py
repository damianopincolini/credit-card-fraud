###################################################################
#
# --- SCRIPT: data_ingestion.py ---
#
# GOAL: Create a function to import csv from a specific path
#
# INPUT: none
#
# OUTPUT:
#   1. dataset_import function
#
###################################################################


import polars as pl

"""
Access paths from the 'paths' dictionary defined in config.py to
ensure project-wide consistency.
"""


def dataset_import(target_path):
    # Note that function wants a generic 'target_path'
    return pl.read_csv(target_path)
