import numpy as np
from scipy.stats import norm, laplace

def gaussian_interval_score(y_true, y_mean, y_std,
                            lower_q = 0.01, upper_q = 0.99, res_q = 0.01):
    """
        Negatively oriented Interval score assuming the Gaussian distribution.

        Smaller values are better.
    """

    scores = []

    q_values = np.arange(lower_q, upper_q, res_q)

    for q in q_values:
        low_q  = 0.5 - (q / 2.0)
        high_q = 0.5 + (q / 2.0)
        alpha = 1.0 - q

        lower = norm.ppf(low_q,  loc=y_mean, scale=y_std)
        upper = norm.ppf(high_q, loc=y_mean, scale=y_std)
        factor = 2.0 / alpha

        below_correct = (lower > y_true).astype(float)
        above_correct = (upper < y_true).astype(float)

        score = upper - lower + factor * ((lower - y_true) * below_correct +  (y_true - upper) * above_correct)
        mean_score = np.mean(score)
        scores.append(mean_score)

    return np.mean(scores)

def laplace_interval_score(y_true, y_mean, y_std,
                            lower_q = 0.01, upper_q = 0.99, res_q = 0.01):
    """
        Negatively oriented Interval score assuming the Laplace distribution.

        Smaller values are better.
    """

    scores = []

    q_values = np.arange(lower_q, upper_q, res_q)

    for q in q_values:
        low_q  = 0.5 - (q / 2.0)
        high_q = 0.5 + (q / 2.0)
        alpha = 1.0 - q

        lower = laplace.ppf(low_q,  loc=y_mean, scale=y_std)
        upper = laplace.ppf(high_q, loc=y_mean, scale=y_std)
        factor = 2.0 / alpha

        below_correct = (lower > y_true).astype(float)
        above_correct = (upper < y_true).astype(float)

        score = upper - lower + factor * ((lower - y_true) * below_correct +  (y_true - upper) * above_correct)
        mean_score = np.mean(score)
        scores.append(mean_score)

    return np.mean(scores)