##################################################################
#
# --- SCRIPT: eda_numerical_features.py ---
#
# GOAL: as a starting point of EDA, a set of statistics is being
#       calculated (R SmartEda style): mean, IQR, kurtosis, and
#       so on.
#
# INPUT: none
#
# OUTPUT:
#   1. FunEdaNum function.
#
###################################################################


# Eda Numerical features

import polars as pl
import polars.selectors as cs

# create a function that calculate a statistic summary for numerical feature for
# different datasets: the global one, the legitimate transactions only and the frauds
# only

def FunEdaNum(df_input, df_label):
    
    # First of all, I set up statistics' expression to be used further on. Note that I
    # don't calculate any value. I prepare expression to feed next step.
    q1  = pl.all().quantile(0.25)
    q3  = pl.all().quantile(0.75)
    iqr = q3 - q1
    lb  = q1 - 1.5 * iqr
    ub  = q3 + 1.5 * iqr
    
    # Execution
    return(
        df_input
    .select(cs.all().exclude("is_fraud"))
    .select(
        [
            pl.all().count().name.suffix("_count"),
            pl.all().mean().name.suffix("_mean"),
            pl.all().std().name.suffix("_st_dev"),
            (pl.all().std() / pl.all().mean()).name.suffix("_CV"),
            q1.name.suffix("_q1"), # recall expression
            pl.all().median().name.suffix("_median"),
            q3.name.suffix("_q3"), # recall expression
            iqr.name.suffix("_IQR"), # recall expression
            pl.all().min().name.suffix("_min"),
            pl.all().max().name.suffix("_max"),
            (pl.all().max() - pl.all().min()).name.suffix("_range"),
            pl.all().skew().name.suffix("_skewness"),
            pl.all().kurtosis().name.suffix("_kurtosis"),
            lb.name.suffix("_lower_bound"), # recall expression
            ub.name.suffix("_upper_bound"), # recall expression
            (pl.all() < lb).sum().name.suffix("_lower_outlier"), # based on expression
            (pl.all() > ub).sum().name.suffix("_upper_outlier"), # based on expression
            ((pl.all() < lb) | (pl.all() > ub)).sum().name.suffix("_total_outlier")
            ]
    )
    .unpivot()
    .with_columns([
        # Split feature name from statistic name (i.g. "amt_mean" -> "amt" and "mean")
        pl.col("variable").
        str.extract(
            r"^(.*)_(count|mean|st_dev|CV|q1|median|q3|IQR|min|max|range|skewness|kurtosis|lower_bound|upper_bound|lower_outlier|upper_outlier|total_outlier)$",
            1)
            .alias("feature"),
        pl.col("variable")
        .str.extract(
            r"^(.*)_(count|mean|st_dev|CV|q1|median|q3|IQR|min|max|range|skewness|kurtosis|lower_bound|upper_bound|lower_outlier|upper_outlier|total_outlier)$",
            2)
            .alias("stat"),
        pl.lit(df_label).alias("subset")
    ])
    .pivot(on="stat", index=["feature", "subset"], values="value")
    )
