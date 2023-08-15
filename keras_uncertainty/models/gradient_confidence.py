import numpy as np

from keras_uncertainty.utils import predict_batches
import keras_uncertainty.backend as K

AGGREGATION_FNS = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "std": np.std,
    "l1_norm": lambda x: np.linalg.norm(x, ord=1, axis=-1),
    "l2_norm": lambda x: np.linalg.norm(x, ord=2, axis=-1)
}

class GradientClassificationConfidence:
    """
        Implementation of gradient uncertainty for classifiers.

        It only needs to provide a model, an aggregation function (to transform gradient to a scalar),
        and optionally a loss to compute the gradient from.
        Reference: Oberdiek et al. Classification Uncertainty of Deep Neural Networks
                   Based on Gradient Information. 2018. https://arxiv.org/abs/1805.08440

    """
    def __init__(self, model, num_classes=None, aggregation="l2_norm", loss=None):
        self.model = model
        self.aggregation = aggregation
        self.num_classes = None
        self.agg_fn = AGGREGATION_FNS[self.aggregation]

        if loss is None:
            self.loss = K.losses.get(model.loss)
        else:
            self.loss = K.losses.get(loss)

        self.gradient_fn = self.compute_gradient_fn()

    def predict(self, x, verbose=0):
        return predict_batches(self.predict_sample, x, batch_size=1, progress_bar = verbose == 1)
        
    def predict_sample(self, x):
        grads = self.gradient_fn([x])
        aggregate = self.agg_fn(grads[0])
        return aggregate

    # The gradient function can only process one sample at a time (not batches),
    # as tf.gradients aggregates gradients in a batch, and does not allow accessing batch gradients.
    def compute_gradient_fn(self):
        if self.num_classes is None:
            num_classes = self.model.output.shape[1]
        else:
            num_classes = self.num_classes

        pred = K.one_hot(K.argmax(self.model.output, axis=1), num_classes=num_classes)
        loss = self.loss(self.model.output, pred)
        loss_grad = K.gradients(loss, self.model.trainable_weights)
        loss_grad = [K.flatten(x) for x in loss_grad]
        loss_grad = K.concatenate(loss_grad, axis=-1)
        grad_fn = K.function([self.model.input], [loss_grad])

        return grad_fn

    
