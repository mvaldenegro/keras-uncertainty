import numpy as np

import keras

from keras_uncertainty.models import TwoHeadStochasticRegressor

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)

    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

def sampling_softmax(mean_logit, std_logit, num_samples=10):
    logit_shape = (mean_logit.shape[0], num_samples, mean_logit.shape[-1])

    logit_mean = np.expand_dims(mean_logit, axis=1)
    logit_mean = np.repeat(logit_mean, num_samples, axis=1)

    logit_std = np.expand_dims(std_logit, axis=1)
    logit_std = np.repeat(logit_std, num_samples, axis=1)

    logit_samples = np.random.normal(size=logit_shape, loc=logit_mean, scale=logit_std)

    prob_samples = softmax(logit_samples, axis=-1)
    probs = np.mean(prob_samples, axis=1)

    # This is required due to approximation error, without it probabilities can sum to 1.01 or 0.99
    probs = probs / np.sum(probs, axis=-1, keepdims=True) 

    return probs

class DisentangledStochasticClassifier(TwoHeadStochasticRegressor):
    def __init__(self, model, epi_num_samples=10, ale_num_samples=100) -> None:
        super(DisentangledStochasticClassifier, self).__init__(model, num_samples=epi_num_samples)

        self.epi_num_samples = epi_num_samples
        self.ale_num_samples = ale_num_samples

    def predict(self, inp, num_samples=None, batch_size=32):
        y_logits_mean, y_logits_std_ale, y_logits_std_epi = TwoHeadStochasticRegressor.predict(self, inp, num_samples=num_samples, batch_size=batch_size, disentangle_uncertainty=True)

        y_probs = sampling_softmax(y_logits_mean, y_logits_std_ale + y_logits_std_epi, num_samples=self.ale_num_samples)
        y_probs_epi = sampling_softmax(y_logits_mean, y_logits_std_epi, num_samples=self.ale_num_samples)
        y_probs_ale = sampling_softmax(y_logits_mean, y_logits_std_ale, num_samples=self.ale_num_samples)

        return y_probs, y_probs_ale, y_probs_epi

class DisentangledEnsembleClassifier:
    def __init__(self):
        pass