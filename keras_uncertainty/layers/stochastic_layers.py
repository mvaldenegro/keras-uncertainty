from keras.layers import Layer
import keras.backend as K

class SamplingSoftmax(Layer):
    """
        Softmax activation with Gaussian logits. Receives mean/variance logits and computes the softmax output through sampling.
    """
    def __init__(self, num_samples=10, temperature=1.0) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.temperature = temperature

    def compute_output_shape(self, input_shape):
        return [(None, input_shape[-1])]

    def call(self, inputs):
        assert len(inputs) == 2, "This layers requires exactly two inputs (mean and variance logits)"

        logit_mean, logit_var = inputs
        logit_std = K.sqrt(logit_var)
        logit_shape = (K.shape(logit_mean)[0], self.num_samples, K.shape(logit_mean)[-1])

        logit_mean = K.expand_dims(logit_mean, axis=1)
        logit_mean = K.repeat_elements(logit_mean, self.num_samples, axis=1)

        logit_std = K.expand_dims(logit_std, axis=1)
        logit_std = K.repeat_elements(logit_std, self.num_samples, axis=1)

        logit_samples = K.random_normal(logit_shape, mean=logit_mean, stddev=logit_std)

        # Apply temperature scaling to logits
        logit_samples = logit_samples / self.temperature

        prob_samples = K.softmax(logit_samples, axis=-1)
        probs = K.mean(prob_samples, axis=1)

        # This is required due to approximation error, without it probabilities can sum to 1.01 or 0.99
        probs = probs / K.sum(probs, axis=-1, keepdims=True)

        return probs

from keras.layers import Dropout

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
            noise_shape = self._get_noise_shape(inputs)

            return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

        return inputs
