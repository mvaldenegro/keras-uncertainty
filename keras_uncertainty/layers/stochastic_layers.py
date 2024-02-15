import keras
from keras.layers import Layer, Dropout
from keras import ops, random

class SamplingSoftmax(Layer):
    """
        Softmax activation with Gaussian logits. Receives mean/variance logits and computes the softmax output through sampling.
    """
    def __init__(self, num_samples=50, temperature=1.0, variance_type="linear_std") -> None:
        """
        Args:
            num_samples: Number of samples used to compute the softmax approximation.
                         This parameter controls the trade-off between computation and approximation quality.
            variance_type: Assumptions made on the variance input, possible values are:
                logit: Input is a variance logit, an exponential transformation will be applied to produce standard deviation.
                linear_std: Input is standard deviation, no transformations are applied.
                linear_variance: Input is variance, square root will be applied to obtain standard deviation.
        """
        super().__init__()

        assert variance_type in ["logit", "linear_std", "linear_variance"]

        self.num_samples = num_samples
        self.temperature = temperature
        self.variance_type = variance_type         

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        mean_is, var_s = input_shape

        return [(None, mean_is[-1])]

    def preprocess_variance_input(self, var_input):
        if self.variance_type is "logit":
            return ops.exp(var_input)

        if self.variance_type is "linear_variance":
            return ops.sqrt(var_input)
        
        return var_input

    def call(self, inputs):
        assert len(inputs) == 2, "This layer requires exactly two inputs (mean and variance logits)"

        logit_mean, logit_var = inputs
        logit_std = self.preprocess_variance_input(logit_var)
        logit_shape = (ops.shape(logit_mean)[0], self.num_samples, ops.shape(logit_mean)[-1])

        logit_mean = ops.expand_dims(logit_mean, axis=1)
        logit_mean = ops.repeat(logit_mean, self.num_samples, axis=1)

        logit_std = ops.expand_dims(logit_std, axis=1)
        logit_std = ops.repeat(logit_std, self.num_samples, axis=1)

        logit_samples = random.normal(logit_shape, mean=logit_mean, stddev=logit_std)
        
        # Apply max normalization for numerical stability
        logit_samples = logit_samples - ops.max(logit_samples, axis=-1, keepdims=True)

        # Apply temperature scaling to logits
        logit_samples = logit_samples / self.temperature

        prob_samples = ops.softmax(logit_samples, axis=-1)
        probs = ops.mean(prob_samples, axis=1)

        # This is required due to approximation error, without it probabilities can sum to 1.01 or 0.99
        probs = probs / ops.sum(probs, axis=-1, keepdims=True)

        return probs

    def get_config(self):
        config = {'num_samples': self.num_samples,
                  'temperature': self.temperature,
                  'variance_type': self.variance_type}
        base_config = super(SamplingSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# TODO
class SamplingSigmoid:
    pass

class SamplingLambda:
    pass

class StochasticDropout(Dropout):
    """
        Applies Dropout to the input, independent of the training phase.

        Used to easily implement MC-Dropout. It is a drop-in replacement for
        the standard Keras Dropout layer, but note that this layer applies
        dropout at the training and inference phases.
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(StochasticDropout, self).__init__(rate, noise_shape, seed, **kwargs)
    
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            return random.dropout(inputs, self.rate, noise_shape=self.noise_shape, seed=self.seed)

        return inputs
