import numpy as np

from keras_uncertainty.distributions import gaussian, rademacher

import keras
from keras.layers import Layer
from keras import activations, initializers, ops, random

# Code partially based on http://krasserm.github.io/2019/03/14/bayesian-neural-networks/

class FlipoutDense(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 initializer_sigma=0.1,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5,
                 bias_distribution=False,
                  **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.initializer_sigma = initializer_sigma
        self.uses_learning_phase = True
        self.bias_distribution = bias_distribution

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (None, self.units)

    def build(self, input_shape):
        feature_dims = input_shape[-1]
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(feature_dims, self.units),
                                         initializer=initializers.random_normal(stddev=self.initializer_sigma),
                                         trainable=True)
        
        # -3.0 is an approximation for 0.0 with softplus, softplus(-3.0) ~ 0.0
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(feature_dims, self.units),
                                          initializer=initializers.random_normal(mean=-3.0, stddev=self.initializer_sigma),
                                          trainable=True)

        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.random_normal(stddev=self.initializer_sigma),
                                       trainable=True)

        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.random_normal(mean=-3.0, stddev=self.initializer_sigma),
                                        trainable=self.bias_distribution)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = ops.softplus(self.kernel_rho)
        kernel_perturb = kernel_sigma * random.normal(self.kernel_mu.shape)
        kernel = self.kernel_mu + kernel_perturb

        if self.bias_distribution:
            bias_sigma = ops.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * random.normal(self.bias_mu.shape)
        else:
            bias = self.bias_mu

        loss = self.kl_loss(kernel, self.kernel_mu, kernel_sigma)

        if self.bias_distribution:
            loss += self.kl_loss(bias, self.bias_mu, bias_sigma)

        self.add_loss(loss)

        input_shape = ops.shape(inputs)
        batch_shape = input_shape[:-1][0]
        sign_input = rademacher.sample(input_shape)
        #sign_output = rademacher.sample(ops.stack([batch_shape, ops.expand_dims(self.units, 0)], axis=0))
        sign_output = rademacher.sample([batch_shape, self.units])
        perturbed_inputs = ops.dot(inputs * sign_input, kernel_perturb) * sign_output

        outputs = ops.dot(inputs, self.kernel_mu)
        outputs += perturbed_inputs
        outputs += bias

        # This always produces stochastic outputs
        return self.activation(outputs)

    def kl_loss(self, w, mu, sigma):
        return self.kl_weight * ops.mean(gaussian.log_probability(w, mu, sigma) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        return ops.log(self.prior_pi_1 * gaussian.probability(w, 0.0, self.prior_sigma_1) +
                       self.prior_pi_2 * gaussian.probability(w, 0.0, self.prior_sigma_2))

    def get_config(self):
        config = {'units': self.units,
                  'kl_weight': self.kl_weight,
                  'activation': self.activation.__name__,
                  #'bias': self.bias,
                  'prior_sigma_1': self.prior_sigma_1,
                  'prior_sigma_2': self.prior_sigma_2,
                  'prior_pi': self.prior_pi_1}
        base_config = super(FlipoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

from .variational_layers import VariationalConvND

class FlipoutConvND(VariationalConvND):
    def __init__(self, rank, filters, kernel_size, kl_weight, strides=1, padding="valid", dilation_rate=(1, 1, 1), activation="linear", **kwargs):
        super().__init__(rank, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)

    def apply_kernel(self, inputs):
        kernel = self.kernel_distribution.mean
        kernel_perturb = self.kernel_distribution.sample_perturbation()
        
        loss = self.kl_loss(kernel, self.kernel_distribution)
        self.add_loss(loss)
    
        input_shape = ops.shape(inputs)
        batch_shape = input_shape[:-1]
        sign_input = rademacher.sample(input_shape)
        
        perturbed_inputs = self.conv(inputs * sign_input, kernel_perturb)
        sign_output = rademacher.sample(ops.shape(perturbed_inputs))
        perturbed_inputs = perturbed_inputs * sign_output

        outputs = self.conv(inputs, kernel)
        outputs += perturbed_inputs
        
        return outputs

class FlipoutConv1D(FlipoutConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=1, padding="valid", dilation_rate=1, activation="linear", **kwargs):
        super().__init__(1, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)

class FlipoutConv2D(FlipoutConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=(1, 1), padding="valid", dilation_rate=(1, 1), activation="linear", **kwargs):
        super().__init__(2, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)

class FlipoutConv3D(FlipoutConvND):
    def __init__(self, filters, kernel_size, kl_weight, strides=(1, 1, 1), padding="valid", dilation_rate=(1, 1, 1), activation="linear", **kwargs):
        super().__init__(3, filters, kernel_size, kl_weight, strides, padding, dilation_rate, activation, **kwargs)
