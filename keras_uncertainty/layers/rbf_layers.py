import numpy as np

import keras
import keras.backend as K
from keras.layers import Layer
from keras.regularizers import l2

import tensorflow as tf
from tqdm import trange
import math

# To make batches from array or iterable, reference https://stackoverflow.com/a/8290508/349130
def make_batches(iter_x, iter_y, batch_size=32):
    l = len(iter_x)
 
    for ndx in range(0, l, batch_size):
        x = iter_x[ndx:min(ndx + batch_size, l)]
        y = iter_y[ndx:min(ndx + batch_size, l)]
        
        yield x, y

def find_rbf_layer(model):
    rbf_layers = []

    for layer in model.layers:
        if type(layer) is RBFClassifier:
            rbf_layers.append(layer)

    if len(rbf_layers) == 1:
        return rbf_layers[0]
    
    raise ValueError("Multiple RBF layers detected, current training loop assumes only one RBF layer, cannot proceed. You can use your own custom training loop")

def duq_training_loop(model, input_feature_model, x_train, y_train, epochs=10, batch_size=32, validation_data=None, penalty_type="two-sided", lambda_coeff=0.5):
    rbf_layer = find_rbf_layer(model)
    num_batches = math.ceil(x_train.shape[0] / batch_size)
    factor = 0.5
    
    for epoch in range(epochs):
        t = trange(num_batches, desc='Epoch {} / {}'.format(epoch, epochs))

        metric_loss_values = {}
        metric_names = model.metrics_names

        for metric_name in metric_names:
            metric_loss_values[metric_name] = []

        for i, (x_batch_train, y_batch_train) in zip(t, make_batches(x_train, y_train)):
            loss_metrics = model.train_on_batch(x_batch_train, y_batch_train)
            
            x_batch_rbf = input_feature_model.predict(x_batch_train)
            rbf_layer.update_centroids(x_batch_rbf, y_batch_train)

            metric_means = []

            for name, value in zip(metric_names, loss_metrics):
                metric_loss_values[name].append(value)
                metric_means.append(np.mean(metric_loss_values[name]))
 
            desc = " ".join(["{}: {:.3f}".format(name, value) for name, value in zip(metric_names, metric_means)])
            t.set_description('Epoch {} / {} - '.format(epoch + 1, epochs) + desc)
            t.refresh()

        if validation_data is not None:
            x_val, y_val = validation_data
            val_loss_metrics = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=0)

            desc = " ".join(["{}: {:.3f}".format(name, value) for name, value in zip(metric_names, val_loss_metrics)])
            print("Validation metrics: {}".format(desc))

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

def add_l2_regularization(model, l2_strength=1e-4):
    for layer in model.layers:
        for tw in layer.trainable_weights:
            model.add_loss(l2(l2_strength)(tw))

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