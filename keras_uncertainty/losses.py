import keras_uncertainty.backend as K

# Losses commonly used in uncertainty quantification and probabilistic forecasting

def regression_gaussian_nll_loss(variance_tensor, epsilon=1e-8, variance_logits=False):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll(y_true, y_pred):
        #if variance_logits:
        #    variance_tensor = K.exp(variance_tensor)

        return 0.5 * K.mean(K.log(variance_tensor + epsilon) + K.square(y_true - y_pred) / (variance_tensor + epsilon))

    return nll

def regression_gaussian_beta_nll_loss(variance_tensor, beta=0.5, epsilon=1e-8, variance_logits=False):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def beta_nll(y_true, y_pred):
        #if variance_logits:
        #    variance_tensor = K.exp(variance_tensor)

        beta_sigma_sq = K.stop_gradient(K.pow(variance_tensor, 2.0 * beta))
        return 0.5 * K.mean(beta_sigma_sq * (K.log(variance_tensor + epsilon) + K.square(y_true - y_pred) / (variance_tensor + epsilon)))

    return beta_nll

def regression_laplace_nll_loss(spread_tensor, epsilon=1e-8, variance_logits=False):
    """
        Laplace negative log-likelihood for regression, with spread parameter estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll(y_true, y_pred):
        return K.mean(K.log(2.0 * spread_tensor + epsilon) + K.abs(y_true - y_pred) / (spread_tensor + epsilon))

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