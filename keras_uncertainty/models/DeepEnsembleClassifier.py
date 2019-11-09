import numpy as np
import keras

class AdversarialExampleGenerator:
    pass

class DeepEnsembleClassifier:
    """
        Implementation of a Deep Ensemble for classification.
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

    def fit_generator(self, generator, epochs=10, **kwargs):
        """
            Fits the Deep Ensemble, each estimator is fit independently on the same data.
        """

        for i in range(self.num_estimators):
            self.estimators[i] = self.model_fn()
            self.estimators[i].fit_generator(generator, epochs=epochs, **kwargs)

    def predict(self, X, batch_size=32):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        for estimator in self.estimators:
            predictions.append(np.expand_dims(estimator.predict(X, batch_size=batch_size, verbose=0), axis=0))

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        
        return mean_pred
