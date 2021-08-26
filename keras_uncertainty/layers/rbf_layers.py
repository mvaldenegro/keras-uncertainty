import numpy as np

import keras
import keras.backend as K
from keras.layers import Layer

import tensorflow as tf

def add_gradient_penalty(model, lambda_coeff=0.5, penalty_type="two-sided"):
    term = K.gradients(K.sum(model.output, axis=1), model.input)
    term = K.square(term)

    if penalty_type == "two-sided":
        penalty = (term - 1) ** 2
    elif penalty_type == "one-sided":
        penalty = K.max(0, term - 1)
    else:
        raise ValueError("Invalid penalty type {}, valid values are [one-sided, two-sided]".format(penalty_type))

    penalty = lambda_coeff * penalty
    penalty = K.in_train_phase(penalty, K.zeros(shape=(1,)))

    model.add_loss(penalty)

class RBFClassifier(Layer):
    """
        Implementation of direct uncertainty quantification (DUQ)
        Reference: 
    """
    def __init__(self, num_classes, length_scale, centroid_dims=2, kernel_initializer="he_normal", gamma=0.99, **kwargs):
        Layer.__init__(self, **kwargs)        
        self.num_classes = num_classes
        self.centroid_dims = centroid_dims
        self.length_scale = length_scale
        self.gamma = gamma

        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        in_features = input_shape[-1]

        self.centroids = self.add_weight(name="centroids", shape=(self.centroid_dims, self.num_classes), dtype="float32", trainable=False, initializer="zeros")
        self.kernels = self.add_weight(name="kernels", shape=(self.centroid_dims, self.num_classes, in_features), initializer=self.kernel_initializer)

        self.m = np.zeros(shape=(self.centroid_dims, self.num_classes))
        self.n = np.ones(shape=(self.num_classes))

    def compute_output_shape(self, input_shape):
        return [(None, self.num_classes)]

    def call(self, inputs, training=None):
        z = tf.einsum("ij,mnj->imn", inputs, self.kernels)
        out = self.rbf(z)

        return out

    def rbf(self, z):
        z = z - self.centroids
        z = K.mean(K.square(z), axis=1) / (2.0 * self.length_scale ** 2)
        z = K.exp(-z)

        return z
    
    def update_centroids(self, inputs, targets):
        kernels = K.get_value(self.kernels)
        z = np.einsum("ij,mnj->imn", inputs, kernels)

        # Here we assume that targets is one-hot encoded.
        class_counts = np.sum(targets, axis=0)
        centroids_sum = np.einsum("ijk,ik->jk", z, targets)

        self.n = self.n * self.gamma + (1 - self.gamma) * class_counts
        self.m = self.m * self.gamma + (1 - self.gamma) * centroids_sum

        K.set_value(self.centroids, self.m / self.n)

    def get_config(self):
        cfg = Layer.get_config(self)
        cfg["num_classes"] = self.num_classes
        cfg["length_scale"] = self.length_scale
        cfg["centroid_dims"] = self.centroid_dims
        cfg["kernel_initializer"] = self.kernel_initializer
        cfg["gamma"] = self.gamma

        return cfg