###################################################################
#
# --- SCRIPT: project_config.py ---
#
# GOAL: set specific configurations
#
# INPUT: none
#
# OUTPUT:
#   1. root_dir and paths.
#
###################################################################


# Library loading 
"""
The proper approach is loding in the project_config script
only what serves to configuration.
Every other library will be loaded in the notebooks only
when required.
This keeps things lighter and faster.
"""

import os
from pathlib import Path

import polars as pl

# Pointers definition
"""
Storage folder setting.
This notebook is saved into projectFolder/notebooks, then I need
to step back into projectFolder and set this as storage level.
"""
if Path.cwd().name == "notebooks":
    os.chdir("..")


# Paths setting
# The project folder is "25_credit_card_fraud"

root_dir = Path.cwd()

paths = {
    "root_dir": root_dir,
    "raw_data_dir": root_dir / "data" / "raw_data",
    "processed_data_dir": root_dir / "data" / "processed_data",
    "report_dir": root_dir / "outputs" / "reports",
    "plots_dir": root_dir / "outputs" / "plots",
    "scripts_dir": root_dir / "scripts",
    "dataset_path": root_dir / "data" / "raw_data" / "fraudTest.csv",
}


# Polars editing setup

pl.Config.set_tbl_rows(20)  # Sets how many rows are shown in the preview.
pl.Config.set_fmt_str_lengths(100)  # Prevents Polars from cutting long strings with ...
pl.Config.set_float_precision(2)  # Forces floats to show only 2 decimal places.
pl.Config.set_tbl_width_chars(80)  # Limits the table to 80 characters wide.
