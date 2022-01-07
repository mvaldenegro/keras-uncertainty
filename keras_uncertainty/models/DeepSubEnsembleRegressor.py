import numpy as np

import keras
import keras.backend as K

from keras.models import Model
from keras.layers import average, Lambda, Input

from .DeepSubEnsembleClassifier import DeepSubEnsemble

class DeepSubEnsembleRegressor(DeepSubEnsemble):
    def __init__(self, trunk_network_fn=None, task_network_fn=None, num_estimators=None,
                 trunk_model=None, task_models=None):

        super().__init__(trunk_network_fn=trunk_network_fn, task_network_fn=task_network_fn, num_estimators=num_estimators,
                         trunk_model=trunk_model, task_models=task_models, needs_test_estimators=True)
    
    def predict(self, X, batch_size=32, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are used to build a gaussian mixture and its mean and standard deviation returned.
        """
        
        means = []
        variances = []        

        trunk_pred = self.trunk_network.predict(X, batch_size=batch_size, **kwargs)

        if num_ensembles is None:
            task_networks = self.test_task_networks
        else:
            task_networks = self.test_task_networks[:num_ensembles]

        for task_network in task_networks:
            mean, var  = task_network.predict(trunk_pred, batch_size=batch_size, **kwargs)
            means.append(mean)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        
        mixture_mean = np.mean(means, axis=0)
        mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
        mixture_var[mixture_var < 0.0] = 0.0
                
        return mixture_mean, np.sqrt(mixture_var)

    def predict_generator(self, generator, steps=None, num_ensembles=None, **kwargs):
        """
            Makes a prediction. Predictions from each estimator are used to build a gaussian mixture and its mean and standard deviation returned.
        """
        
        means = []
        variances = []

        trunk_pred = self.trunk_network.predict_generator(generator, steps=steps, **kwargs)

        if num_ensembles is None:
            task_networks = self.test_task_networks
        else:
            task_networks = self.test_task_networks[:num_ensembles]

        for task_network in task_networks:
            mean, var  = task_network.predict(trunk_pred, batch_size=batch_size, **kwargs)
            means.append(mean)
            variances.append(var)

        means = np.array(means)
        variances = np.array(variances)
        
        mixture_mean = np.mean(means, axis=0)
        mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
        mixture_var[mixture_var < 0.0] = 0.0
                
        return mixture_mean, np.sqrt(mixture_var)

    def combine_trunk_task_regression(self):        
        trunk_inp = self.trunk_network.input
        test_task_network = self.task_network_fn()
        mean, var = test_task_network(self.trunk_network.output)

        train_model = Model(trunk_inp, mean)
        train_model.compile(loss=self.loss(var), optimizer=self.optimizer, metrics=self.metrics)        

        return train_model, test_task_network

    def fit(self, X, y, epochs=10, batch_size=32, **kwargs):
        """
            Fits the Deep Sub-Ensemble, each task network is fit independently on the same data.
        """

        # First fit the trunk and one task network:
        trunk_task, test_network = self.combine_trunk_task_regression()
        self.test_task_networks[0] = test_network

        trunk_task.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

        # Freeze layers in the trunk model
        for layer in self.trunk_network.layers:
            layer.trainable = False

        # Then train the remaining task networks
        for i in range(1, self.num_estimators):
            trunk_task, test_network = self.combine_trunk_task_regression()
            self.test_task_networks[i] = test_network

            trunk_task.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def fit_generator(self, generator, epochs=10, steps_per_epoch=None, **kwargs):
        """
            Fits the Deep Sub-Ensemble, each task network is fit independently on the same data.
        """

        # First fit the trunk and one task network:
        trunk_task, test_network = self.combine_trunk_task_regression()
        self.test_task_networks[0] = test_network

        trunk_task.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)

        # Freeze layers in the trunk model
        for layer in self.trunk_network.layers:
            layer.trainable = False

        # Then train the remaining task networks
        for i in range(1, self.num_estimators):
            trunk_task, test_network = self.combine_trunk_task_regression()
            self.test_task_networks[i] = test_network

            trunk_task.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)