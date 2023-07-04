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
        self.multi_output = len(model.outputs) > 1
    
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

    def call_samples(self, x, num_samples=None, multi_output=False, **kwargs):
        """
            Performs num_samples predictions by calling the model, and returns the produced output samples.
        """

        if num_samples is None:
            num_samples = self.num_samples

        assert num_samples > 0
        samples = [None] * num_samples

        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        for i in range(num_samples):
            samples[i] = self.model(x, **kwargs)

        return samples

    #TODO Find a way to keep output names
    def divide_outputs(self, multi_output_samples, num_outputs):
        output = [None] * num_outputs

        for out_idx in range(num_outputs):
            output[out_idx] = np.array([e[out_idx] for e in multi_output_samples])

        return output

    @property
    def layers(self):
        return self.model.layers

    @property
    def num_outputs(self):
        return len(self.model.outputs)

    @property
    def outputs(self):
        return self.model.outputs

    @property
    def output(self):
        return self.model.output

class StochasticClassifier(StochasticModel):
    def __init__(self, model, num_samples=10):
        super().__init__(model, num_samples)


    def predict(self, inp, num_samples=None, batch_size=32, **kwargs):
        """
            Performs a prediction given input inp taking multiple stochastic samples, and returns the averaged probabilities of model output.
        """

        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, multi_output=self.multi_output, **kwargs)

        if self.multi_output:
            samples = self.divide_outputs(samples, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                mean_probs = np.mean(samples[i], axis=0)
                mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)
                outputs.append(mean_probs)

            return outputs
        
        mean_probs = np.mean(samples, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)

        return mean_probs

    def __call__(self, inputs, num_samples=None, **kwargs):
        samples = self.call_samples(inputs, num_samples, multi_output=self.multi_output, **kwargs)
        
        if self.multi_output:
            outputs = []

            for i in range(self.num_outputs):
                mean_probs = K.mean(samples[i], axis=0)                
                outputs.append(mean_probs)

            return outputs

        mean_probs = K.mean(samples, axis=0)

        return mean_probs

class StochasticRegressor(StochasticModel):
    def __init__(self, model, num_samples=10):
        super().__init__(model, num_samples)

    def predict(self, inp, num_samples=None, batch_size=32, output_scaler=None, **kwargs):
        """
            Performs a prediction  given input inp using MC Dropout, and returns the mean and standard deviation of the model output.
            For multi-output models, the mean and standard deviations are returned for each output head, in the same order (mean_1, std_1, mean_2, std_2, ..., mean_n, std_n)
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, multi_output=self.multi_output, **kwargs)
        
        if self.multi_output:
            samples = self.divide_outputs(samples, self.num_outputs)
            outputs = []

            for i in range(self.num_outputs):
                if output_scaler is not None:
                    samples[i] = list(map(lambda x: output_scaler.inverse_transform(x), samples[i]))

                mean_pred = np.mean(samples[i], axis=0)
                std_pred = np.std(samples[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        if output_scaler is not None:
            samples = list(map(lambda x: output_scaler.inverse_transform(x), samples))

        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)

        return mean_pred, std_pred

    def __call__(self, inputs, num_samples=None, **kwargs):
        samples = self.call_samples(inputs, num_samples, multi_output=self.multi_output, **kwargs)
        
        if self.multi_output:
            outputs = []

            for i in range(self.num_outputs):
                mean_pred = K.mean(samples[i], axis=0)
                std_pred = K.std(samples[i], axis=0)
                
                outputs.append(mean_pred)
                outputs.append(std_pred)

            return outputs

        mean_pred = K.mean(samples, axis=0)
        std_pred = K.std(samples, axis=0)

        return mean_pred, std_pred

class TwoHeadStochasticRegressor(StochasticModel):
    """
        A stochastic model that has two ouput heads, one for mean and another for variance, useful for aleatoric/epistemic uncertainty estimation.
    """
    def __init__(self, model, num_samples=10, variance_type="linear_variance"):
        super().__init__(model, num_samples)

        assert variance_type in ["logit", "linear_std", "linear_variance"]
        self.variance_type = variance_type

    """
        Preprocesses and interprets the variance output prodcued by the model, producing a standard deviation.
    """
    def preprocess_variance_output(self, var_input):
        if self.variance_type is "logit":
            return np.exp(var_input)

        if self.variance_type is "linear_variance":
            return np.sqrt(var_input)
        
        return var_input

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
        stds = self.preprocess_variance_output(variances)
        
        mixture_mean = np.mean(means, axis=0)
        mixture_var  = np.mean(np.square(stds) + np.square(means), axis=0) - np.square(mixture_mean)
        mixture_var[mixture_var < 0.0] = 0.0
        mixture_std = np.sqrt(mixture_var)
                                
        if disentangle_uncertainty:            
            epi_std = np.std(means, axis=0)
            ale_std = np.mean(stds, axis=0)

            return mixture_mean, ale_std, epi_std

        return mixture_mean, mixture_std

class KernelDensityStochasticModel(StochasticModel):
    def __init__(self, model, num_samples=10, bandwidth=1.0):
        super().__init__(model, num_samples)