import keras
from keras import ops

# Losses commonly used in uncertainty quantification and probabilistic forecasting

def regression_gaussian_nll_loss(epsilon=1e-8, variance_logits=False, **kwargs):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll(y_true, y_pred_mean, y_pred_var):
        #if variance_logits:
        #    variance_tensor = K.exp(variance_tensor)

        return 0.5 * ops.mean(ops.log(y_pred_var + epsilon) + ops.square(y_true - y_pred_mean) / (y_pred_var + epsilon))

    return nll

def regression_gaussian_beta_nll_loss(variance_tensor, beta=1.0, epsilon=1e-8, variance_logits=False):
    """
        Gaussian negative log-likelihood for regression, with variance estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def beta_nll(y_true, y_pred):
        #if variance_logits:
        #    variance_tensor = K.exp(variance_tensor)

        beta_sigma_sq = ops.stop_gradient(ops.power(variance_tensor, beta))
        return 0.5 * ops.mean(beta_sigma_sq * (ops.log(variance_tensor + epsilon) + ops.square(y_true - y_pred) / (variance_tensor + epsilon)))

    return beta_nll

def regression_laplace_nll_loss(spread_tensor, epsilon=1e-8, variance_logits=False, **kwargs):
    """
        Laplace negative log-likelihood for regression, with spread parameter estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def nll(y_true, y_pred):
        return ops.mean(ops.log(2.0 * spread_tensor + epsilon) + ops.abs(y_true - y_pred) / (spread_tensor + epsilon))

    return nll

def regression_laplace_beta_nll_loss(spread_tensor, beta=1.0, epsilon=1e-8, variance_logits=False):
    """
        Laplace negative log-likelihood for regression, with spread parameter estimated by the model.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.
    """
    def beta_nll(y_true, y_pred):
        beta_sp = ops.stop_gradient(ops.pow(spread_tensor, beta))
        return ops.mean(beta_sp * (ops.log(2.0 * spread_tensor + epsilon) + ops.abs(y_true - y_pred) / (spread_tensor + epsilon)))

    return beta_nll

def quantile_loss(tau: float):
    """
        Standard pinball loss for quantile regression.
    """

    def quantile(y_true, y_pred):
        err = y_true - y_pred
        return ops.mean(ops.maximum(tau * err, (tau - 1.0) * err), axis=-1)

    return quantile

def brier_score(y_true, y_pred):
    """
        Mean squared error on the probabilities.
    """
    return ops.mean(ops.square(y_true - y_pred))