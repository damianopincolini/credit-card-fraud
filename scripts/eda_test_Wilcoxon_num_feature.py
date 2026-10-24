###################################################################
#
# --- SCRIPT: eda_test_wilcoxon_num_feature.py ---
#
# GOAL: in the eda context a need to fulfill is to understand the
#       association between the boolean target value and each of
#       numerical predictor.
#       Actually Mann-Whitney U has been used even though the 
#       function has been named Wilcoxon...
#
# INPUT: none
#
# OUTPUT:
#   1. FunWilcoxonTest function.
#
###################################################################

# EDA: test Wilokoxon for numerical features

import polars as pl
from scipy.stats import mannwhitneyu


def FunWilcoxonTest(df: pl.DataFrame, target: str, numericalFeatures: list[str]):
    # initialization of a new list that gathers results
    results = []

    # creation of sub-dataset for each value of target column
    target_0 = df.filter(pl.col("is_fraud") == 0)
    target_1 = df.filter(pl.col("is_fraud") == 1)

    # sub-dataset length
    length_target_0 = target_0.height
    length_target_1 = target_1.height

    for col in numericalFeatures:
        vector_0 = target_0.select(col).to_numpy().ravel()
        vector_1 = target_1.select(col).to_numpy().ravel()
        # ravel() flattens the 2D column array into a 1D vector to ensure compatibility
        # with SciPy's statistical tests.

        statistics_u, p_value = mannwhitneyu(
            vector_0, vector_1, alternative="two-sided"
        )
        # Execute the two-sided Mann-Whitney U test to compare distributions between
        # target groups.
        # Unpack the results into the test statistic (U) and its corresponding p-value.

        effect_size = 1 - (2 * statistics_u) / (length_target_0 * length_target_1)
        # Calculate the Rank-Biserial Correlation as the effect size for the Mann-Whitney U test.
        # # This scales the U statistic to a range between -1 and +1, indicating the strength
        # # and direction of the separation between the two target groups

        results.append(
            {
                "Features": col,
                "U_statistics": statistics_u,
                "p_value": p_value,
                "Effect Size": round(effect_size, 4),
                "Significant": "Yes" if p_value < 0.05 else "No",
            }
        )
    df_results = pl.DataFrame(results)
    return df_results.sort(pl.col("Effect Size").abs(), descending=True)
