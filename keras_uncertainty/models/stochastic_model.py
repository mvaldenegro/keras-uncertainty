import numpy as np
import keras
import keras.backend as K

class StochasticModel:
    """
        Stochastic model, requiring several forward passes to produce an estimate of the posterior predictive distribution.
        This class just wraps a keras model to enable dropout at inference time.
    """
    def __init__(self, model):
        """
            Builds a stochastic model from a keras model. The model should already be trained.
        """
        self.model = model
    
    def predict_samples(self, x, num_samples=10, batch_size=32):
        """
            Performs num_samples predictions using the model, and returns the produced output samples.
        """

        assert num_samples > 0

        samples = [None] * num_samples

        for i in range(num_samples):
            samples[i] = self.model.predict(x, batch_size=batch_size, verbose=0)

        return np.array(samples)

class StochasticClassifier(StochasticModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10, batch_size=32):
        """
            Performs a prediction given input inp using MC Dropout, and returns the averaged probabilities of model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size)
        mean_probs = np.mean(samples, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)

        return mean_probs

class  StochasticRegressor(StochasticModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10, batch_size=32, output_scaler=None):
        """
            Performs a prediction  given input inp using MC Dropout, and returns the mean and standard deviation of the model output.
        """
        samples = self.predict_samples(inp, num_samples, batch_size=batch_size)

        if output_scaler is not None:
            samples = list(map(lambda x: output_scaler.inverse_transform(x), samples))

        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)

        return mean_pred, std_pred    

