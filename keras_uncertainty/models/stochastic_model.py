import numpy as np
import keras_uncertainty.backend as K

class StochasticModel:
    """
        Stochastic model, requiring several forward passes to produce an estimate of the posterior predictive distribution.
        This class just wraps a keras model to enable dropout at inference time.
    """
    def __init__(self, model, num_samples=10):
        """
            Builds a stochastic model from a keras model. The model should already be trained.
        """
        self.model = model
        self.num_samples = num_samples
    
    def predict_samples(self, x, num_samples=None, batch_size=32, multi_output=False, **kwargs):
        """
            Performs num_samples predictions using the model, and returns the produced output samples.
        """

        if num_samples is None:
            num_samples = self.num_samples

        assert num_samples > 0
        samples = [None] * num_samples

        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        for i in range(num_samples):
            samples[i] = self.model.predict(x, batch_size=1, **kwargs)

        if multi_output:
            return samples
        else:
            return np.array(samples)

class StochasticClassifier(StochasticModel):
    def __init__(self, model, num_samples=10):
        super().__init__(model, num_samples)

    def predict(self, inp, num_samples=None, batch_size=32, **kwargs):
        """
            Performs a prediction given input inp using MC Dropout, and returns the averaged probabilities of model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, **kwargs)
        mean_probs = np.mean(samples, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)

        return mean_probs

class  StochasticRegressor(StochasticModel):
    def __init__(self, model, num_samples=10):
        super().__init__(model, num_samples)

    def predict(self, inp, num_samples=None, batch_size=32, output_scaler=None, **kwargs):
        """
            Performs a prediction  given input inp using MC Dropout, and returns the mean and standard deviation of the model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, **kwargs)

        if output_scaler is not None:
            samples = list(map(lambda x: output_scaler.inverse_transform(x), samples))

        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)

        return mean_pred, std_pred    

class TwoHeadStochasticRegressor(StochasticModel):
    """
        A stochastic model that has two ouput heads, one for mean and another for variance, useful for aleatoric/epistemic uncertainty estimation.
    """
    def __init__(self, model, num_samples=10):
        super().__init__(model, num_samples)

    def predict(self, inp, num_samples=None, batch_size=32, output_scaler=None, disentangle_uncertainty=False, **kwargs):
        """
            Performs a prediction given input inp and returns the mean and standard deviation of the model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, multi_output=True, **kwargs)
        mean_samples, var_samples = [x[0] for x in samples], [x[1] for x in samples]

        if output_scaler is not None:
            mean_samples = list(map(lambda x: output_scaler.inverse_transform(x), mean_samples))
            var_samples = list(map(lambda x: output_scaler.inverse_transform(x), var_samples))

        means = np.array(mean_samples)
        variances = np.array(var_samples)
        
        mixture_mean = np.mean(means, axis=0)
        mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
        mixture_var[mixture_var < 0.0] = 0.0
                
        if disentangle_uncertainty:
            epi_var = np.var(means, axis=0)
            ale_var = np.mean(variances, axis=0)

            return mixture_mean, np.sqrt(ale_var), np.sqrt(epi_var)

        return mixture_mean, np.sqrt(mixture_var)

class KernelDensityStochasticModel(StochasticModel):
    def __init__(self, model, num_samples=10, bandwidth=1.0):
        super().__init__(model, num_samples)