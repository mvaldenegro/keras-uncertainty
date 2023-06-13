import numpy as np

import os
import yaml

from pydoc import locate
from itertools import chain

import keras_uncertainty
load_model = keras_uncertainty.backend.models.load_model

class AdversarialExampleGenerator:
    pass

METADATA_FILENAME = "metadata.yml"

class DeepEnsemble(object):
    def __init__(self, model_fn=None, num_estimators=None, models=None, needs_test_estimators=False):
        self.needs_test_estimators = needs_test_estimators

        if models is None:
            assert model_fn is not None and num_estimators is not None
            assert num_estimators > 0
            
            self.num_estimators = num_estimators
            self.train_estimators = [None] * num_estimators 
            self.test_estimators = [None] * num_estimators

            for i in range(self.num_estimators):
                if self.needs_test_estimators:
                    estimators = model_fn()

                    if type(estimators) is not tuple:
                        raise ValueError("model_fn should return a tuple")

                    if len(estimators) != 2:
                        raise ValueError("model_fn returned a tuple of unexpected size ({} vs 2)".format(len(estimators)))

                    train_est, test_est = estimators
                    self.train_estimators[i] = train_est
                    self.test_estimators[i] = test_est
                else:
                    est = model_fn()
                    self.train_estimators[i] = est
                    self.test_estimators[i] = est

        else:
            assert model_fn is None and num_estimators is None

            self.train_estimators = models
            self.test_estimators = models

            self.num_estimators = len(models)

        self.multi_output = self.num_outputs > 1

    @property
    def trainable_variables(self):
        train_var = [estimator.trainable_variables for estimator in self.test_estimators]
        return [var for var in chain.from_iterable(train_var)]

    @property
    def num_outputs(self):
        return len(self.test_estimators[0].outputs)

    def divide_outputs(self, ensemble_samples, num_outputs):
        output = [None] * num_outputs

        for out_idx in range(num_outputs):
            output[out_idx] = np.array([e[out_idx] for e in ensemble_samples])

        return output

    def save(self, folder, filename_pattern="model-ensemble-{}.hdf5"):
        """
            Save a Deep Ensemble into a folder, using individual HDF5 files for each ensemble member.
            This allows for easily loading individual ensembles. Metadata is saved to allow loading of the whole ensemble.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        model_metadata = {}

        for i in range(self.num_estimators):
            filename = os.path.join(folder, filename_pattern.format(i))
            self.test_estimators[i].save(filename)

            print("Saved estimator {} to {}".format(i, filename))

            model_metadata[i] = filename_pattern.format(i)

        metadata = {"models": model_metadata, "class": self.__module__}

        with open(os.path.join(folder, METADATA_FILENAME), 'w') as outfile:
            yaml.dump(metadata, outfile)

    @staticmethod
    def load(folder):
        """
            Load a Deep Ensemble model from a folder containing individual HDF5 files.
        """
        metadata = {}

        with open(os.path.join(folder, METADATA_FILENAME)) as infile:
            metadata = yaml.full_load(infile)

        models = []

        for _, filename in metadata["models"].items():
            models.append(load_model(os.path.join(folder, filename)))

        clazz = locate(metadata["class"])

        return clazz(models=models)

    def get_weights(self):
        return [estimator.get_weights() for estimator in self.test_estimators]

    def set_weights(self, weights):
        for estimator, weight in zip(self.test_estimators, weights):
            estimator.set_weights(weight)

    def summary(self):
        for estimator in self.test_estimators:
            estimator.summary()

class DeepEnsembleClassifier(DeepEnsemble):
    """
        Implementation of a Deep Ensemble for classification.
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

    def predict(self, X, batch_size=32, num_ensembles=None, return_std = False, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        if num_ensembles is None:
            estimators = self.test_estimators
        else:
            estimators = self.test_estimators[:num_ensembles]

        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        for estimator in estimators:
            predictions.append(np.expand_dims(estimator.predict(X, batch_size=batch_size, **kwargs), axis=0))

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        
        if return_std:
            std_pred = np.std(predictions, axis=0)

            return mean_pred, std_pred

        return mean_pred

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

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        
        return mean_pred

