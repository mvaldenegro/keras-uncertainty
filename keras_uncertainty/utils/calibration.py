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

def validate_calibration_data(y_pred, y_true, y_confidences):
    if len(y_true.shape) > 2:
        raise ValueError("y_true should be a 2D array, found shape {}".format(y_true.shape))

    if len(y_true.shape) == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()

    if len(y_pred.shape) > 2:
        raise ValueError("y_true should be a 2D array, found shape {}".format(y_true.shape))

    if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if len(y_confidences.shape) != 2:
        raise ValueError("y_confidences should exactly be a 2D array (samples, probs), found shape {}".format(y_confidences.shape))

    return y_pred, y_true, y_confidences
    

def classifier_calibration_error(y_pred, y_true, y_confidences, metric="mae", num_bins=10, weighted=False):
    """
        Estimates calibration error for a classifier.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)

    errors = []
    weights = []

    for start, end in pairwise(bin_edges):
        indices = np.where(np.logical_and(y_confidences >= start, y_confidences < end))
        filt_preds = y_pred[indices]
        filt_classes = y_true[indices]
        filt_confs = y_confidences[indices]

        if len(filt_confs) > 0:
            bin_acc = accuracy(filt_classes, filt_preds)
            bin_conf = np.mean(filt_confs)

            error = abs(bin_conf - bin_acc)
            weight = len(filt_confs)

            errors.append(error)            
            weights.append(weight)

    errors = np.array(errors)
    weights = np.array(weights) / sum(weights)

    if weighted:
        return sum(errors * weights)

    return np.mean(errors)

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

            curve_conf.append(bin_conf)
            curve_acc.append(bin_acc)
        else:
            p = np.mean([start, end])
            curve_conf.append(p)
            curve_acc.append(p)

    return curve_conf, curve_acc

def regressor_calibration_curve(y_pred, y_true, y_confidences):
    return None

def classifier_accuracy_confidence_curve(y_pred, y_true, y_confidences, num_points=20):
    candidate_confs = np.linspace(0.0, 0.99, num=num_points)

    out_confidences = []
    out_accuracy = []

    for confidence in candidate_confs:
        examples_idx = np.where(y_confidences >= confidence)
        filt_preds = y_pred[examples_idx]
        filt_true = y_true[examples_idx]

        acc = accuracy(filt_true, filt_preds)

        out_confidences.append(confidence)
        out_accuracy.append(acc)

    return out_confidences, out_accuracy