###################################################################
#
# --- SCRIPT: eda_cat_multivariate_association.py ---
#
# GOAL: in the context of multivariate analysis (association between
#       predictors only), this scripts allows to measure
#       association between categorical predictors and expose it
#       into an heatmap.
#
# INPUT: none
#
# OUTPUT:
#   1. FunCramerV_multivariate function.
#
###################################################################

# eda multivariate analysis. Categorical variables

import polars as pl
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats.contingency import association


def FunCramerV_multivariate(
    df: pl.DataFrame
    ):

    # categorical features' selection
    data_train_cat_no_target = (
        df
        .select(
            pl.col(pl.Categorical),
            pl.col("job_aggregate")
            )
    )

    # conversion to pandas
    data_train_cat_no_target_pd = (
        data_train_cat_no_target.to_pandas()
    )

    # pandas index creation with column names
    columns = data_train_cat_no_target_pd.columns

    # number of columns
    n_cols = len(columns)

    # cramer's matrix initialization (empty with all zeros)
    cramer_matrix = np.zeros((n_cols, n_cols))

    # double for-cycle to create cramer matrix
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                cramer_matrix[i, j] = 1.0
            else:
                contingency_table = pd.crosstab(
                    data_train_cat_no_target_pd[columns[i]],
                    data_train_cat_no_target_pd[columns[j]]
                    )
                
                cramer_matrix[i, j] = association(
                    contingency_table,
                    method = "cramer"
                    )
    
    # convert cramer_matrix into a pd DataFrame        
    cramer_pd = (
        pd.DataFrame(
            cramer_matrix,
            index=columns, 
            columns=columns)
        .reset_index()
        )
    
    # convert cramer_matrix (pd) into pl and unpivot it so as to prepare it for 
    # visualization
    cramer_long_pl = (
        pl.from_pandas(cramer_pd)
        .unpivot(
            index = "index",
            variable_name = "variable",
            value_name = "cramer_V"
            )
        .with_columns(pl.col("cramer_V").round(3))
        )
    
    # 
    Cramer_V_multivariate_heatmap = (
        ggplot(
            data = cramer_long_pl,
            mapping = aes(
                x = "index",
                y = "variable",
                fill = "cramer_V"
            )
        )
        + geom_tile()
        + geom_text(
            aes(label = "cramer_V"),
            size = 10
            )  
        + scale_fill_gradient(
            low = "white",
            high = "purple",
            limits = (0, 1)
            )
        + theme_minimal()
        + labs(
            title = "Multivariate analysis - Categorical features - Association heatmap (Cramer's V)",
            x = "",
            y = ""
            )
        + theme(axis_text_x = element_text(rotation = 45,
                                           hjust = 1),
                plot_title = element_text(size = 9,
                                          face = "bold")
                )
    )

    # return final object
    return Cramer_V_multivariate_heatmap
        
            
        