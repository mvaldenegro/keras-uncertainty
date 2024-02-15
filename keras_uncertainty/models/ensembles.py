import keras
from keras import ops

class Ensemble:
    """
        Ensemble implementation, supporting regression and classification.

        Ensembles build @num_estimator models, train them, and combine their predictions.
        This implementation is a simple ensemble, for uncertainty disentanglement use DisentanglingEnsemble.
    """

    def __init__(self, model_fn, num_estimators=5, ensembling_mode="classification", multi_output=False):
        assert ensembling_mode in ["classification", "regression"]
        assert model_fn is not None

        self.ensembling_mode = ensembling_mode
        self.multi_output = multi_output
        self.model_fn = model_fn
        self.num_estimators = num_estimators
        self.estimators = [self.model_fn() for i in range(self.num_estimators)]

        self.dispatch_combine = None

        if self.ensembling_mode == "classification":
            self.dispatch_combine = self.combine_samples_classification
        
        if self.ensembling_mode == "regression":
            self.dispatch_combine = self.combine_samples_regression

    def fit(self, x, y=None, epochs=10, batch_size=32, **kwargs):
        for i in range(self.num_estimators):
            self.estimators[i].fit(x, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, inputs, batch_size=32, num_ensembles=None, verbose=0, **kwargs):
        predictions = []

        if num_ensembles is None:
            estimators = self.estimators
        else:
            estimators = self.estimators[:num_ensembles]

        for estimator in estimators:
            predictions.append(ops.expand_dims(estimator.predict(inputs, batch_size=batch_size, verbose=verbose, **kwargs), axis=0))

        if self.multi_output:
            predictions = self.divide_outputs(predictions, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = ops.mean(predictions[i], axis=0)
                std_pred = ops.std(predictions[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        predictions = ops.concatenate(predictions)
        return self.dispatch_combine(predictions, numpy=True)

    def predict_samples(self, inputs, **kwargs):
        pass

    def __call__(self, inputs, num_ensembles=None, **kwargs):
        predictions = []

        if num_ensembles is None:
            estimators = self.estimators
        else:
            estimators = self.estimators[:num_ensembles]

        for estimator in estimators:
            predictions.append(ops.expand_dims(estimator(inputs, **kwargs), axis=0))

        if self.multi_output:
            predictions = self.divide_outputs_symbolic(predictions, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = ops.mean(predictions[i], axis=0)
                std_pred = ops.std(predictions[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        predictions = ops.concatenate(predictions)
        return self.dispatch_combine(predictions)

    def combine_samples_classification(self, samples, numpy=False):
        mean_pred = ops.mean(samples, axis=0)
        mean_pred = mean_pred / ops.sum(mean_pred, axis=1, keepdims=True)

        if numpy:
            return mean_pred.numpy()

        return mean_pred

    def combine_samples_regression(self, samples, numpy=False):
        mean_pred = ops.mean(samples, axis=0)
        std_pred = ops.std(samples, axis=0)

        if numpy:
            return mean_pred.numpy(), std_pred.numpy()

        return mean_pred, std_pred

    def divide_outputs(self, samples):
        pass

class EnsembleClassifier(Ensemble):
    def __init__(self, model_fn, num_estimators=5, multi_output=False):
        Ensemble.__init__(self, model_fn, num_estimators=num_estimators, ensembling_mode="classification", multi_output=multi_output)

class EnsembleRegressor(Ensemble):
    def __init__(self, model_fn, num_estimators=5, multi_output=False):
        Ensemble.__init__(self, model_fn, num_estimators=num_estimators, ensembling_mode="regression", multi_output=multi_output)