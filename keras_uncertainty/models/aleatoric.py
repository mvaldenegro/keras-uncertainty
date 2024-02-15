import keras
from keras import ops
from keras.models import Model

#Useful adapter to implement multi-backend train steps.
class BackendTrainStepModelAdaptor(Model):
    def train_step(self, data):
        backend = keras.backend.backend()

        if backend == 'tensorflow':
            return self.tensorflow_train_step(data)

        if backend == "jax":
            return self.jax_train_step(data)

        if backend == "pytorch":
            return self.pytorch_train_step(data)

        raise ValueError("Unknown keras backend {}".format(backend))

    def tensorflow_train_step(self, data):
        raise NotImplementedError()

    def jax_train_step(self, data):
        raise NotImplementedError()

    def pytorch_train_step(self, data):
        raise NotImplementedError()

class TwoHeadModel(BackendTrainStepModelAdaptor):
    """
        Two headed model, for aleatoric uncertainty estimation, assumes that the loss takes predicted mean and variance, which requires a custom training loop.

        The model should have two outpus, mean and variance. Loss function should be a loss for aleatoric uncertainty estimation, like gaussian negative log-likelihood.
    """

    def tensorflow_train_step(self, data):
        import tensorflow as tf
        
        x, y = data

        with tf.GradientTape() as tape:
            y_pred_mean, y_pred_var = self(x, training=True)  # Forward pass

            loss = self.loss(y, y_pred_mean, y_pred_var)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred_mean)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}