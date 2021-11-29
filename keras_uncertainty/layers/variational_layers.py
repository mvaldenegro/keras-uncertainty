import numpy as np

import keras
import keras.backend as K
from keras.layers import Layer
from keras import activations, initializers, regularizers, constraints

from keras_uncertainty.distributions import gaussian

# Code partially based on http://krasserm.github.io/2019/03/14/bayesian-neural-networks/

class VariationalDense(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior=True,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior = prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)
        self.uses_learning_phase = True

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return [(None, self.units)]

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.normal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.normal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = K.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * K.random_normal(self.kernel_mu.shape)

        bias_sigma = K.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * K.random_normal(self.bias_mu.shape)

        loss = self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + self.kl_loss(bias, self.bias_mu, bias_sigma)

        self.add_loss(K.in_train_phase(loss, 0.0))

        # This always produces stochastic outputs
        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        return self.kl_weight * K.mean(gaussian.log_probability(w, mu, sigma) - self.prior * self.log_prior_prob(w))

    def log_prior_prob(self, w):
        return K.log(self.prior_pi_1 * gaussian.probability(w, 0.0, self.prior_sigma_1) +
                     self.prior_pi_2 * gaussian.probability(w, 0.0, self.prior_sigma_2))

    def get_config(self):
        config = {'units': self.units,
                  'kl_weight': self.kl_weight,
                  'activation': self.activation.__name__,
                  #'bias': self.bias,
                  'prior': self.prior,
                  'prior_sigma_1': self.prior_sigma_1,
                  'prior_sigma_2': self.prior_sigma_2,
                  'prior_pi_1': self.prior_pi_1}
        base_config = super(VariationalDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class VariationalConv(keras.layers.Layer):
    def __init__(self, rank, filters, kernel_size, kl_weight, strides=1, padding="valid", dilation_rate=1, activation="linear", use_bias_distribution = False, **kwargs):
        super().__init__(**kwargs)

        assert rank == len(kernel_size)

        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.kl_weight = kl_weight
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias_distribution = use_bias_distribution

    def build(self, input_shape):
        self.kernel_distribution, self.bias_distribution  = self.build_kernel_bias_distribution(input_shape, use_bias_distribution=self.use_bias_distribution)

    def call(self, inputs):
        output = self.apply_kernel(inputs)
        output += self.apply_bias(output)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        from keras.utils import conv_utils

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)        

    def build_kernel_bias_distributions(self, input_shape, use_bias_distribution=False):
        raise NotImplementedError("This method should be overriden in a subclass")

    def apply_kernel(self, inputs):
         raise NotImplementedError("This method should be overriden in a subclass")

    def apply_bias(self, inputs):
        raise NotImplementedError("This method should be overriden in a subclass")

    def get_config(self):
        return {}

    def conv(self, inputs, kernel):
        conv_dict = {
            1: K.conv1d,
            2: K.conv2d,
            3: K.conv3d
        }

        return conv_dict[self.rank](inputs, kernel, strides=self.strides, padding=self.padding, data_format="channels_last", dilation_rate=self.dilation_rate)

    #def log_prior_prob(self, parameter, distribution):
    #    return K.log(self.prior_pi_1 * gaussian.probability(w, 0.0, self.prior_sigma_1) +
    #                 self.prior_pi_2 * gaussian.probability(w, 0.0, self.prior_sigma_2))

    def kl_loss(self, parameter, distribution):
        return self.kl_weight * K.mean(distribution.log_probability(parameter))
        #return self.kl_weight * K.mean(gaussian.log_probability(w, mu, sigma) - self.prior * self.log_prior_prob(w))

from keras_uncertainty.distributions.gaussian import GaussianDistribution

class VariationalConvND(VariationalConv):
    def __init__(self, rank, filters, kernel_size, kl_weight, strides=1, padding="valid", dilation_rate=1, activation="linear", **kwargs):
        super().__init__(rank, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)

    def build_kernel_bias_distribution(self, input_shape, use_bias_distribution=False):
        input_channels = input_shape[-1]
        kernel_shape = self.kernel_size + (input_channels, self.filters)

        mean = self.add_weight(name="kernel_mean", shape=kernel_shape, initializer="glorot_uniform")
        var = self.add_weight(name="kernel_var", shape=kernel_shape, initializer="glorot_uniform")
        std = K.softplus(var)

        kernel_distribution = GaussianDistribution(mean, std, kernel_shape)

        bias_shape = (self.filters, )
        mean = self.add_weight(name="bias_mean", shape=bias_shape, initializer="glorot_uniform")

        if use_bias_distribution:
            var = self.add_weight(name="bias_var", shape=bias_shape, initializer="glorot_uniform")
            std = K.softplus(var)
        else:
            std = K.zeros(bias_shape)

        bias_distribution = GaussianDistribution(mean, std, bias_shape)

        return kernel_distribution, bias_distribution

    def apply_kernel(self, inputs):
        kernel = self.kernel_distribution.sample()
        
        return self.conv(inputs, kernel)

    def apply_bias(self, inputs):
        bias = self.bias_distribution.sample()

        return K.bias_add(inputs, bias, data_format="channels_last")

class VariationalConv1D(VariationalConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=1, padding="valid", dilation_rate=1, activation="linear", **kwargs):
        super().__init__(1, filters, kernel_size, strides, padding, dilation_rate, activation, **kwargs)

class VariationalConv2D(VariationalConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=(1, 1), padding="valid", dilation_rate=1, activation="linear", **kwargs):
        super().__init__(2, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)

class VariationalConv3D(VariationalConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=(1, 1, 1), padding="valid", dilation_rate=(1, 1), activation="linear", **kwargs):
        super().__init__(3, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)