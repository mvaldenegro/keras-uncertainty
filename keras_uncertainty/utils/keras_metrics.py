import keras_uncertainty.backend as K

def negative_log_likelihood(y_true, y_pred):
    """
        Negative log-likelihood or negative log-probability loss/metric.
        Reference: Evaluating Predictive Uncertainty Challenge, Quiñonero-Candela et al, 2006.
        It sums over classes: log(y_pred) for true class and log(1.0 - pred) for not true class, and then takes average across samples.
    """
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    return -K.mean(K.sum(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1), axis=-1)

def entropy(y_true, y_pred):
    """
        Standard entropy over class probabilities.
        It sums y_pred * K.log(y_pred + epsilon) over class probabilities, and then takes average over samples
    """
    return K.mean(-K.sum(y_pred * K.log(y_pred + K.epsilon()), axis=-1), axis=-1)
