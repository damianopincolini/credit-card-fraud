###################################################################
#
# --- SCRIPT: eda_cramer_V_cat_feature.py ---
#
# GOAL: in the context of EDA, this script provides with a function
#       that calculate and expose in a table the Cramer's V
#       association index between each single categorical
#       predictor and the boolean target value.
#
# INPUT: none
#
# OUTPUT:
#   1. FunCramerV function.
#
###################################################################

# Cramer's V for categorical features.staticmethod

import polars as pl
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association


def FunCramerV(
    df: pl.DataFrame,
    target: str,
    cat_features: list[str]
    ):

    # create a list with results
    results = []

    # reiterate contingency table and Cramer's computation
    for col in cat_features:
        # select a two-column dataset where target values couples
        # each categorical predictor.
        df_couple = df.select([target, col])

        # contingency table
        contingency_table = (
            df_couple.pivot(
                on=target,
                index=col,
                values=target,
                aggregate_function="len"
            )
            .drop(col)
            .to_numpy()
        )

        # p_value and Cramer's V computation
        _, p_value, _, _ = chi2_contingency(contingency_table)
        # Note the need for unpacking chi2_contingency outputs,
        # using underscores ('_') as placeholders to discard the
        # chi2 statistic, degrees of freedom and expected
        # frequencies.
        v_cramer = association(contingency_table, method="cramer")

        # append single result
        results.append(
            {
                "Features": col,
                "p_value": "< 0.0001" if p_value < 0.0001 else round(float(p_value), 4),
                "Cramer_V": round(v_cramer, 4),
                "Significant": "Yes" if p_value < 0.05 else "No",
            }
        )

    # turn results list into a Polars DataFrame
    return pl.DataFrame(results).sort(pl.col("Cramer_V"), descending=True)





