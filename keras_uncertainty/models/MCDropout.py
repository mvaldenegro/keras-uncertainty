import numpy as np
import keras
import keras.backend as K

from keras_uncertainty.utils import predict_batches

class MCDropoutModel:
    """
        Monte Carlo Dropout/DropConnect implementation over a keras model.
        This class just wraps a keras model to enable dropout at inference time.
    """
    def __init__(self, model):
        """
            Builds a MC Dropout model from a keras model. The model should already be trained.
        """

        self.model = model
        self.mc_func = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[-1].output])
        self.mc_pred = lambda x: self.mc_func([x, 1])
    
    def predict_samples(self, x, num_samples=10, batch_size=32, **kwargs):
        """
            Performs a prediction using MC Dropout, and returns the produced output samples from the model.
        """

        assert num_samples > 0

        samples = [None] * num_samples
        
        for i in range(num_samples):
            samples[i] = predict_batches(self.mc_pred, x, batch_size=batch_size)

        return np.array(samples)

class MCDropoutClassifier(MCDropoutModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10, batch_size=32, **kwargs):
        """
            Performs a prediction given input inp using MC Dropout, and returns the averaged probabilities of model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, **kwargs)
        mean_probs = np.mean(samples, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)

        return mean_probs

class MCDropoutRegressor(MCDropoutModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10, batch_size=32, output_scaler=None, **kwargs):
        """
            Performs a prediction  given input inp using MC Dropout, and returns the mean and standard deviation of the model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size, **kwargs)

        if output_scaler is not None:
            samples = list(map(lambda x: output_scaler.inverse_transform(x), samples))

        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)

        return mean_pred, std_pred    

