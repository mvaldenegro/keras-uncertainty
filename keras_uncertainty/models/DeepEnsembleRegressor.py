import numpy as np
import keras

import keras.backend as K

def deep_ensemble_regression_nll_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    sigma_sq = y_pred[:, 1] + K.epsilon()

    return 0.5 * K.mean(K.log(sigma_sq) + K.square(y_true - mu) / sigma_sq)

class DeepEnsembleRegressor:
    """
        Implementation of a Deep Ensemble for regression.
        Assumes that a model outputs at least a 2D vector, where the first feature dimension is the mean, and the second feature dimension is the variance squared.
    """
    def __init__(self, model_fn, num_estimators):
        """
            Builds a Deep Ensemble given a function to make model instances, and the number of estimators.
        """
        self.model_fn = model_fn
        self.num_estimators = num_estimators
        self.estimators = [None] * num_estimators

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
            Fits the Deep Ensemble, each estimator is fit independently on the same data.
        """

        for i in range(self.num_estimators):
            self.estimators[i] = self.model_fn()
            self.estimators[i].fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, X, batch_size=32):
        """
            Makes a prediction. Predictions from each estimator are used to build a gaussian mixture and its mean and standard deviation returned.
        """
        
        means = []
        variances = []

        for estimator in self.estimators:
            preds = estimator.predict(X, batch_size=batch_size)
            means.append(preds[:, 0])
            variances.append(preds[:, 1])

        means = np.array(means)
        variances = np.array(variances)
        
        mixture_mean = np.mean(means, axis=0)[0]
        mixture_var  = np.mean(variances + np.square(means)) - mixture_mean
                
        return mixture_mean, np.sqrt(mixture_var)
