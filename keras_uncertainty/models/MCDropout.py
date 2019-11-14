import numpy as np
import keras
import keras.backend as K

class MCDropoutModel:
    """
        Monte Carlo Dropout implementation over a keras model.
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
    
    def predict_samples(self, x, num_samples=10):
        """
            Performs a prediction using MC Dropout, and returns the produced output samples from the model.
        """

        samples = [None] * num_samples

        for i in range(num_samples):
            samples[i] = self.mc_pred(x)[0]

        return np.array(samples)

class MCDropoutClassifier(MCDropoutModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10):
        """
            Performs a prediction given input inp using MC Dropout, and returns the averaged probabilities of model output.
        """
        samples = self.predict_samples(inp, num_samples)
        mean_probs = np.mean(samples, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs, axis=1, keepdims=True)

        return mean_probs

class MCDropoutRegressor(MCDropoutModel):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, inp, num_samples=10):
        """
            Performs a prediction  given input inp using MC Dropout, and returns the mean and standard deviation of the model output.
        """
        samples = self.predict_samples(inp, num_samples)

        samples = np.array(samples)
        mean_pred = np.mean(samples, axis=0)[0]
        std_pred = np.std(samples, axis=0)[0]

        return mean_pred, std_pred    

