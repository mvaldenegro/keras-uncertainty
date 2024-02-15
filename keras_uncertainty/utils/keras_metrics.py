import keras
from keras import ops

def negative_log_likelihood(y_true, y_pred, epsilon=1e-6):
    """
        Negative log-likelihood or negative log-probability loss/metric.
        Reference: Evaluating Predictive Uncertainty Challenge, Quiñonero-Candela et al, 2006.
        It sums over classes: log(y_pred) for true class and log(1.0 - pred) for not true class, and then takes average across samples.
    """
    y_pred = ops.clip(y_pred, epsilon, 1.0 - epsilon)

    return -ops.mean(ops.sum(y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred), axis=-1), axis=-1)

def entropy(y_true, y_pred, epsilon=1e-6):
    """
        Standard entropy over class probabilities.
        It sums y_pred * K.log(y_pred + epsilon) over class probabilities, and then takes average over samples
    """
    return ops.mean(-ops.sum(y_pred * ops.log(y_pred + epsilon), axis=-1), axis=-1)
