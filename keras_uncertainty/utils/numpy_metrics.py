# Metrics useful for uncertainty quantification, implemented using numpy.

import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def negative_log_likelihood(y_true, y_pred):
    return 0

def entropy(y_true, y_pred):
    return 0