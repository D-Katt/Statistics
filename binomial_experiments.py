"""Implementation of statistical significance test
for binomial experiments.
"""

import numpy as np
from scipy.stats import norm


def get_binary_stats() -> tuple:
    """Function transforms user input for two experiments
    into binomial distributions, calculates Z-score
    and compares it with the threshold level.
    :return: Tuple containing two elements (Z-score, p-value)
    """
    n_participants_1 = int(input('Number of respondents in Group 1 = '))
    n_positives_1 = int(input('Number of positive answers in Group 1 = '))

    n_participants_2 = int(input('Number of respondents in Group 2 = '))
    n_positives_2 = int(input('Number of positive answers in Group 2 = '))

    threshold = float(input('Threshold significance = '))

    share_1 = n_positives_1 / n_participants_1
    share_2 = n_positives_2 / n_participants_2

    std_1 = np.std(
        [1 for _ in range(n_positives_1)] +
        [0 for _ in range(n_participants_1 - n_positives_1)]
    )
    std_2 = np.std(
        [1 for _ in range(n_positives_2)] +
        [0 for _ in range(n_participants_2 - n_positives_2)]
    )

    print(f'Group 1 standard deviation = {std_1}')
    print(f'Group 2 standard deviation = {std_2}')

    z_score = (share_2 - share_1) / (std_1 / np.sqrt(n_participants_2))
    print(f'Z-score = {z_score}')

    p_value = norm.sf(abs(z_score))
    print(f'p-value = {p_value}')

    if p_value <= threshold:
        print('Distributions are statistically different.')
    else:
        print('Distributions are not statistically different.')

    return z_score, p_value


z, p = get_binary_stats()
