# Utils to estimate uncertainty calibration

from .numpy_metrics import accuracy

import numpy as np
from itertools import tee

EPSILON = 1e-5

#From itertools recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def classifier_calibration_error(y_pred, y_true, y_confidences, metric="mae", num_bins=10):
    """
        Estimates calibration error for a classifier.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins)

    error = 0

    for start, end in pairwise(bin_edges):
        indices = np.where(np.logical_and(y_confidences >= start, y_confidences < end))
        filt_preds = y_pred[indices]
        filt_classes = y_true[indices]
        filt_confs = y_confidences[indices]

        if len(filt_confs) > 0:
            bin_acc = accuracy(filt_classes, filt_preds)
            bin_conf = np.mean(filt_confs)

            print(bin_acc, bin_conf)

            error += abs(bin_conf - bin_acc)

    return error / num_bins

def classifier_calibration_curve(y_pred, y_true, y_confidences, metric="mae", num_bins=10):
    """
        Estimates the calibration plot for a classifier and returns the points in the plot.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)
    curve_conf = []
    curve_acc = []

    for start, end in pairwise(bin_edges):
        indices = np.where(np.logical_and(y_confidences >= start, y_confidences < end))
        filt_preds = y_pred[indices]
        filt_classes = y_true[indices]
        filt_confs = y_confidences[indices]

        if len(filt_confs) > 0:
            bin_acc = accuracy(filt_classes, filt_preds)
            bin_conf = np.mean(filt_confs)

            print(bin_acc, bin_conf)

            curve_conf.append(bin_conf)
            curve_acc.append(bin_acc)
        else:
            p = np.mean([start, end])
            curve_conf.append(p)
            curve_acc.append(p)

    return curve_conf, curve_acc

def regressor_calibration_curve(y_pred, y_true, y_confidences):
    return None