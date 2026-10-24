###################################################################
#
# --- SCRIPT: eda_num_multivariate_correlation.py ---
#
# GOAL: in the context of EDA, a multivariate analysis is being
#       executed to understand and represent whether there is 
#       correlation between each pair of numerical predictors.
#       The statistical tool used is Pearson's correlation index.
#
# INPUT: none
#
# OUTPUT:
#   1. FunPearsonCorr function.
#
###################################################################


import polars as pl
from plotnine import *


def FunPearsonCorr(
    df: pl.DataFrame
):
    # excerpt numerical predictors without target value
    data_train_num_no_target = (
        df
        .drop("is_fraud")
        )
    
    # correlation matrix (conversion to pandas is mandatory)
    corr_matrix_pd = (
        data_train_num_no_target
        .to_pandas()
        .corr(method = "pearson")
        .reset_index()
        )
    
    # pivot corr_matrix to feed plotnine
    corr_matrix_long_pl = (
        pl.from_pandas(corr_matrix_pd)
        .unpivot(
            index = "index",
            variable_name="variable",
            value_name="correlation"
            )
        .with_columns(
            pl.col("correlation").round(3)
            )
        )
    
    # create heat map with plotnine
    correlation_heatmap = (
        ggplot(
            data = corr_matrix_long_pl,
            mapping = aes(
                x = "index",
                y = "variable",
                fill = "correlation"
                )
            )
        + geom_tile()
        + geom_text(
            aes(
                label="correlation"),
            size=10
            )
        + scale_fill_gradient2(
            low="blue",
            mid="white",
            high="red",
            limits=(-1, 1)
            )
        + theme_minimal()
        + labs(
            title="Numerical features - correlation heatmap (Pearson)",
            x="",
            y=""
            )
        + theme(axis_text_x = element_text(rotation = 45,
                                           hjust = 1),
                plot_title = element_text(size = 9,
                                          face = "bold")
                )
        )
    return correlation_heatmap