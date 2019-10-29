# Some implementation ideas taken from https://medium.com/@albertoarrigoni/paper-review-code-deep-ensembles-nips-2017-c5859070b8ce

import numpy as np
import keras

import keras.backend as K

def deep_ensemble_regression_nll_loss(sigma_sq, epsilon = 1e-6):
    """
        Regression loss for a Deep Ensemble, using the negative log-likelihood loss.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.4
    """
    def nll_loss(y_true, y_pred):
        return 0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - y_pred) / (sigma_sq + epsilon))

    return nll_loss

class DeepEnsembleRegressor:
    """
        Implementation of a Deep Ensemble for regression.
        Uses two models, one for training and another for inference/testing. The user has to provide a model function that returns
        the train and test models, and use the provided deep_ensemble_nll_loss for training.
    """
    def __init__(self, model_fn, num_estimators):
        """
            Builds a Deep Ensemble given a function to make model instances, and the number of estimators.

            For training it uses a model that only outputs the mean, while the loss uses both the mean and variance produced by the model.
            For testing, a model that shares weights with the training model is used, but the testing model outputs both mean and variance. The final
            prediction is made with a mixture of gaussians, where each gaussian is one trained model instance.
        """
        self.model_fn = model_fn
        self.num_estimators = num_estimators
        self.train_estimators = [None] * num_estimators
        self.test_estimators = [None] * num_estimators

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
            Fits the Deep Ensemble, each estimator is fit independently on the same data.
        """

        for i in range(self.num_estimators):
            models = self.model_fn()

            if type(models) is not tuple:
                raise ValueError("model_fn should return a tuple")

            if len(models) is not 2:
                raise ValueError("model_fn returned a tuple of unexpected size ({} vs 2)".format(len(models)))

            train_model, test_model = models
            train_model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

            self.train_estimators[i] = train_model
            self.test_estimators[i] = test_model
            

    def predict(self, X, batch_size=32):
        """
            Makes a prediction. Predictions from each estimator are used to build a gaussian mixture and its mean and standard deviation returned.
        """
        
        means = []
        variances = []

        for estimator in self.test_estimators:
            mean, var  = estimator.predict(X, batch_size=batch_size)
            means.append(mean)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        
        mixture_mean = np.mean(means, axis=0)
        mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
        mixture_var[mixture_var < 0.0] = 0.0
                
        return mixture_mean, np.sqrt(mixture_var)
