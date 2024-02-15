import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, average

class DeepSubEnsemble:
    def __init__(self, trunk_network_fn=None, task_network_fn=None, num_estimators=None,
                 trunk_model=None, task_models=None, needs_test_estimators=False):
        self.needs_test_estimators = needs_test_estimators

        if trunk_model is None and task_models is None:
            assert trunk_network_fn is not None and task_network_fn is not None and num_estimators is not None
            assert num_estimators > 0

            self.num_estimators = num_estimators

            self.trunk_network = trunk_network_fn()
            self.train_task_networks = [None] * num_estimators 
            self.test_task_networks = [None] * num_estimators            

            if self.needs_test_estimators:
                self.task_network_fn = task_network_fn
            else:
                for i in range(self.num_estimators):
                    est = task_network_fn()
                    self.train_task_networks[i] = est
                    self.test_task_networks[i] = est

        else:
            assert trunk_network_fn is None and task_network_fn is None and num_estimators is None

            self.trunk_network = trunk_model

            self.train_task_networks = task_models
            self.test_task_networks = task_models

            self.num_estimators = len(task_models)

    @staticmethod
    def combine_trunk_task_classification(trunk_model, task_model):
        inp =  Input(trunk_model.input_shape[1:])
        x = trunk_model(inp)
        out = task_model(x)

        model = Model(inp, out)

        return model

    @staticmethod
    def build_classification_ensemble(trunk_model, task_models):
        inp = Input(shape=trunk_model.input_shape[1:])

        trunk = trunk_model(inp)
        task_outputs = []

        for task in task_models:
            task_outputs.append(task(trunk))

        output = average(task_outputs, axis=-1)
        model = Model(inp, output)

        return model

    def compile(self, loss, optimizer, metrics=[]):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

class DeepSubEnsembleClassifier(DeepSubEnsemble):
    def __init__(self, trunk_network_fn=None, task_network_fn=None, num_estimators=None,
                 trunk_model=None, task_models=None):

        super().__init__(trunk_network_fn=trunk_network_fn, task_network_fn=task_network_fn, num_estimators=num_estimators,
                         trunk_model=trunk_model, task_models=task_models, needs_test_estimators=False)

    def predict(self, X, batch_size=32, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        trunk_pred = self.trunk_network.predict(X, batch_size=batch_size, **kwargs)

        if num_ensembles is None:
            task_networks = self.test_task_networks
        else:
            task_networks = self.test_task_networks[:num_ensembles]

        for task_network in task_networks:
            task_pred = task_network.predict(trunk_pred, batch_size=batch_size, **kwargs)
            predictions.append(np.expand_dims(task_pred, axis=0))

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        
        return mean_pred

    def predict_generator(self, generator, steps=None, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are averaged and probabilities normalized.
        """
        
        predictions = []

        trunk_pred = self.trunk_network.predict_generator(generator, steps=steps, **kwargs)

        if num_ensembles is None:
            task_networks = self.test_task_networks
        else:
            task_networks = self.test_task_networks[:num_ensembles]

        for task_network in task_networks:
            task_pred = task_network.predict(trunk_pred, **kwargs)
            predictions.append(np.expand_dims(task_pred, axis=0))

        predictions = np.concatenate(predictions)
        mean_pred = np.mean(predictions, axis=0)
        mean_pred = mean_pred / np.sum(mean_pred, axis=1, keepdims=True)
        
        return mean_pred

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
            Fits the Deep Sub-Ensemble, each task network is fit independently on the same data.
        """

        # First fit the trunk and one task network:
        trunk_task = self.combine_trunk_task_classification(self.trunk_network, self.train_task_networks[0])
        trunk_task.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        trunk_task.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

        # Freeze layers in the trunk model
        for layer in self.trunk_network.layers:
            layer.trainable = False

        # Then train the remaining task networks
        for i in range(1, self.num_estimators):

            trunk_task = self.combine_trunk_task_classification(self.trunk_network, self.train_task_networks[i])
            trunk_task.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

            trunk_task.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def fit_generator(self, generator, epochs=10, steps_per_epoch=None, **kwargs):
        """
            Fits the Deep Sub-Ensemble, each task network is fit independently on the same data.
        """

        # First fit the trunk and one task network:
        trunk_task = self.combine_trunk_task(self.trunk_network, self.train_task_networks[0])
        trunk_task.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)

        # Freeze layers in the trunk model
        for layer in self.trunk_network.layers:
            layer.trainable = False

        # Then train the remaining task networks
        for i in range(1, self.num_estimators):

            trunk_task = self.combine_trunk_task(self.trunk_network, self.train_task_networks[i])
            trunk_task.fit(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)