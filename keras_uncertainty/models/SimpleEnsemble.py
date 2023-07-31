import numpy as np

import os
import yaml

from pydoc import locate

from .DeepEnsembleClassifier import DeepEnsemble
import keras_uncertainty.backend as K

class SimpleEnsemble(DeepEnsemble):
    """
        Implementation of a a simple Ensemble, for multiple tasks (not necessarily classification or regression).
        In comparison with a Deep Ensemble, this implementation can be used with any loss function.
    """
    def __init__(self, model_fn=None, num_estimators=None, models=None):
        """
            Builds a Deep Ensemble given a function to make model instances, and the number of estimators.
        """
        super().__init__(model_fn=model_fn, num_estimators=num_estimators,
                         needs_test_estimators=False, models=models)

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
            Fits the Deep Ensemble, each estimator is fit independently on the same data.
        """

        for i in range(self.num_estimators):
            self.train_estimators[i].fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def fit_generator(self, generator, epochs=10, **kwargs):
        """
            Fits the Deep Ensemble, each estimator is fit independently on the same data.
        """

        for i in range(self.num_estimators):
            self.train_estimators[i].fit_generator(generator, epochs=epochs, **kwargs)

    def predict(self, X, batch_size=32, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        if num_ensembles is None:
            estimators = self.test_estimators
        else:
            estimators = self.test_estimators[:num_ensembles]

        for estimator in estimators:
            predictions.append(np.expand_dims(estimator.predict(X, batch_size=batch_size, verbose=0, **kwargs), axis=0))

        if self.multi_output:
            predictions = self.divide_outputs(predictions, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = np.mean(predictions[i], axis=0)
                std_pred = np.std(predictions[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def predict_generator(self, generator, steps=None, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        if num_ensembles is None:
            estimators = self.test_estimators
        else:
            estimators = self.test_estimators[:num_ensembles]

        for estimator in estimators:
            predictions.append(np.expand_dims(estimator.predict_generator(generator, steps=steps, **kwargs), axis=0))

        if self.multi_output:
            predictions = self.divide_outputs(predictions, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = np.mean(predictions[i], axis=0)
                std_pred = np.std(predictions[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def __call__(self, inputs, num_ensembles=None, **kwargs):
        predictions = []

        if num_ensembles is None:
            estimators = self.test_estimators
        else:
            estimators = self.test_estimators[:num_ensembles]

        if self.multi_output:
            output_names = [x.name for x in estimators[0].outputs]
            for estimator in estimators:
                prediction = estimator(inputs, **kwargs)
                pred = [K.expand_dims(x, axis=0) for x in prediction]
                predictions.append(pred)    

            predictions = self.divide_outputs_symbolic(predictions, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = K.mean(predictions[i], axis=0)
                std_pred = K.std(predictions[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs        

        for estimator in estimators:
            prediction = estimator(inputs, **kwargs)
            pred = K.expand_dims(prediction, axis=0)
            predictions.append(pred)

        predictions = K.concatenate(predictions, axis=0)

        mean_pred = K.mean(predictions, axis=0)
        std_pred = K.std(predictions, axis=0)

        return mean_pred, std_pred
        