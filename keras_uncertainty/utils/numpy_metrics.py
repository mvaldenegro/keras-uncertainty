# Metrics useful for uncertainty quantification, implemented using numpy.

import numpy as np

def accuracy(y_true, y_pred):
    """
        Simple categorical accuracy.
    """
    return np.mean(y_true == y_pred)

EPSILON = 1e-7

def numpy_regression_nll(y_true_mean, y_pred_mean, y_pred_variance, epsilon=1e-6):
    """
        Negative log-likelihood loss/metric for regression, based on Gaussian assumptions.
        Reference: Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, Lakshminarayanan et al 2017.
        Needs true mean, and predicted mean and variances as inputs.

    """
    return 0.5 * np.mean(np.log(y_pred_variance + epsilon) + np.square(y_true_mean - y_pred_mean) / (y_pred_variance + epsilon))

def numpy_classification_nll(y_true, y_pred):
    """
        Negative log-likelihood or negative log-probability loss/metric for classiication.
        Reference: Evaluating Predictive Uncertainty Challenge, Quiñonero-Candela et al, 2006.
        It sums over classes: log(y_pred) for true class and log(1.0 - pred) for not true class, and then takes average across samples.
    """
    y_pred = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

    return -np.mean(np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=-1), axis=-1)

# For backwards compatibility
numpy_negative_log_likelihood = numpy_classification_nll

def numpy_entropy(probs, axis=-1, eps=1e-6):
    return -np.sum(probs * np.log(probs + eps), axis=axis)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

mse = mean_squared_error
mae = mean_absolute_error

def get_metric(identifier):
    if callable(identifier):
        return identifier

    if identifier in globals():
        return globals()[identifier]
    
    raise ValueError("Unknown numpy metric {}".format(identifier))