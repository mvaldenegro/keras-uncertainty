import keras
import keras.backend as K

# Losses commonly used in uncertainty quantification and probabilistic forecasting

def regression_gaussian_nll_loss(variance_tensor, epsilon=1e-5):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.4
    """
    def nll(y_true, y_pred):
        return 0.5 * K.mean(K.log(variance_tensor + epsilon) + K.square(y_true - y_pred) / (variance_tensor + epsilon))

    return nll

def pinball_loss(tau: float):
    """
        Standard pinball loss for quantile regression.
    """

    def pinball(y_true, y_pred):
        err = y_true - y_pred
        return K.mean(K.maximum(tau * err, (tau - 1.0) * err), axis=-1)

    return pinball

def brier_score(y_true, y_pred):
    """
        Mean squared error on the probabilities.
    """
    return K.mean(K.square(y_true - y_pred))