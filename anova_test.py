"""Implementation of ANOVA tests (Analysis of Variance)
to check statistical significance of differences between treatment groups.
"""

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


def one_way_anova(df: pd.DataFrame, threshold: float) -> tuple:
    """Function calculates statistics for one-way ANOVA test:
    numerical results of experiments over multiple groups split over one factor.
    :param df: DataFrame containing results for three or more groups,
    where column names represent groups
    :param threshold: Significance threshold
    :return: Tuple of two elements (F-value of the test, p-value from the F-distribution)
    """
    groups = [df[column] for column in df.columns]
    f_value, p_value = stats.f_oneway(*groups)

    if p_value <= threshold:
        print('Distributions are statistically different.')
    else:
        print('Distributions are not statistically different.')

    return f_value, p_value


def two_way_anova(df: pd.DataFrame) -> pd.DataFrame:
    """Function calculates statistics for two-way ANOVA test:
    results of experiments over multiple objects split over two factors.
    :param df: DataFrame containing results of the experiment
    where first two columns represent factors (categorical values) and
    the third column represents the experiment outcome (numerical value)
    :return: DataFrame containing ANOVA table with p-values
    for each individual factor and their interaction
    """
    factor_1, factor_2, result = df.columns
    model = ols(f'{result} ~ C({factor_1}) + C({factor_2}) + C({factor_1}):C({factor_2})', data=df).fit()
    return sm.stats.anova_lm(model, typ=2)
