import numpy as np
import math

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor, StochasticRegressor, TwoHeadStochasticRegressor
from keras_uncertainty.layers import DropConnectDense, VariationalDense, FlipoutDense, StochasticDropout
from keras_uncertainty.metrics import gaussian_interval_score
from keras_uncertainty.utils import regressor_calibration_error, regressor_error_confidence_curve, regressor_calibration_curve
from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

from keras_uncertainty.utils.calibration import regressor_calibration_curve

def toy_function(input):
    output = []

    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)

        out = math.sin(inp) + np.random.normal(0, std)
        output.append(10 * out)

    return np.array(output)

def train_standard_model(x_train, y_train, train_domain, test_domain):
    inp = Input(shape=(1,))
    x = Dense(32, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    mean = Dense(1, activation="linear")(x)
    var = Dense(1, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=100)

    train_mean_pred, train_var_pred = pred_model.predict(train_domain)
    train_std_pred = np.sqrt(train_var_pred)
    test_mean_pred, test_var_pred = pred_model.predict(test_domain)
    test_std_pred = np.sqrt(test_var_pred)

    return train_mean_pred, train_std_pred, test_mean_pred, test_std_pred

def train_dropout_model(x_train, y_train, train_domain, test_domain, prob=0.2):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(1,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))
    model.add(Dense(1, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=100)

    mc_model = StochasticRegressor(model)
    train_pred_mean, train_pred_std = mc_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = mc_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_dropconnect_model(x_train, y_train, train_domain, test_domain, prob=0.05, noise_shape=None):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(1,), prob=prob, noise_shape=noise_shape))
    model.add(DropConnectDense(32, activation="relu", prob=prob, noise_shape=noise_shape))
    model.add(DropConnectDense(1, activation="linear", noise_shape=noise_shape))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=100)

    mc_model = StochasticRegressor(model)
    train_pred_mean, train_pred_std = mc_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = mc_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_ensemble_model(x_train, y_train, train_domain, test_domain):
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(32, activation="relu")(inp)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

        return train_model, pred_model
    
    model = DeepEnsembleRegressor(model_fn, num_estimators=10)
    model.fit(x_train, y_train, verbose=2, epochs=100)
    train_pred_mean, train_pred_std = model.predict(train_domain)
    test_pred_mean, test_pred_std = model.predict(test_domain)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_bayesbackprop_model(x_train, y_train, train_domain, test_domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(VariationalDense(32, kl_weight, **prior_params, prior=True, activation="relu", input_shape=(1,)))
    model.add(VariationalDense(32, kl_weight, **prior_params, prior=True, activation="relu"))
    model.add(VariationalDense(1, kl_weight, **prior_params, prior=True, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=2500)

    st_model = StochasticRegressor(model)
    train_pred_mean, train_pred_std = st_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = st_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_flipout_model(x_train, y_train, train_domain, test_domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, **prior_params, prior=False, bias_distribution=True, activation="relu", input_shape=(1,)))
    model.add(FlipoutDense(32, kl_weight, **prior_params, prior=False, bias_distribution=True, activation="relu"))
    model.add(FlipoutDense(1, kl_weight, **prior_params, prior=False, bias_distribution=True, activation="linear"))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.fit(x_train, y_train, verbose=2, epochs=700)
    
    st_model = StochasticRegressor(model)
    train_pred_mean, train_pred_std = st_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = st_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_flipout_nll_model(x_train, y_train, train_domain, test_domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches

    inp = Input(shape=(1,))
    x = FlipoutDense(32, kl_weight, activation="relu",)(inp)
    x = FlipoutDense(32, kl_weight, activation="relu")(x)
    mean = FlipoutDense(1, kl_weight, activation="linear")(x)
    var = FlipoutDense(1, kl_weight, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=700)

    st_model = TwoHeadStochasticRegressor(pred_model)
    train_pred_mean, train_pred_std = st_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = st_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std

def train_flipout_beta_nll_model(x_train, y_train, train_domain, test_domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches

    inp = Input(shape=(1,))
    x = FlipoutDense(32, kl_weight, activation="relu",)(inp)
    x = FlipoutDense(32, kl_weight, activation="relu")(x)
    mean = FlipoutDense(1, kl_weight, activation="linear")(x)
    var = FlipoutDense(1, kl_weight, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_beta_nll_loss(var), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=700)

    st_model = TwoHeadStochasticRegressor(pred_model)
    train_pred_mean, train_pred_std = st_model.predict(train_domain, num_samples=50)
    test_pred_mean, test_pred_std = st_model.predict(test_domain, num_samples=50)

    return train_pred_mean, train_pred_std, test_pred_mean, test_pred_std


METHODS = {
    "Classical NN": train_standard_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
    "Flipout + NLL": train_flipout_nll_model,
    "Flipout + Beta-NLL": train_flipout_beta_nll_model
}

NUM_SAMPLES = 30

if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=2, ncols=len(METHODS.keys()), figsize=(20, 3))
    methods = list(METHODS.keys())

    x_train = np.linspace(-4.0, 4.0, num=1200)
    x_test = np.linspace(-7.0, 7.0, num=200)

    y_train = toy_function(x_train)
    y_test = toy_function(x_test)

    domain = np.linspace(-7.0, 7.0, num=1000)
    domain = domain.reshape((-1, 1))

    domain_y = toy_function(domain)

    print(axes.shape)

    for i, key in enumerate(methods):
        ax = axes[0][i]
        calib_ax = axes[1][i]

        y_train_pred_mean, y_train_pred_std, y_test_pred_mean, y_test_pred_std = METHODS[key](x_train, y_train, x_train, x_test)

        train_score = gaussian_interval_score(y_train, y_train_pred_mean, y_train_pred_std)
        train_calib_err = regressor_calibration_error(y_train_pred_mean, y_train, y_train_pred_std)

        test_score = gaussian_interval_score(y_test, y_test_pred_mean, y_test_pred_std)
        test_calib_err = regressor_calibration_error(y_test_pred_mean, y_test, y_test_pred_std)

        train_confs, train_errors = regressor_error_confidence_curve(y_train_pred_mean, y_train, y_train_pred_std, normalize_std=True)
        test_confs, test_errors = regressor_error_confidence_curve(y_test_pred_mean, y_test, y_test_pred_std, normalize_std=True)

        ax.plot(train_confs, train_errors, label="Train")
        ax.plot(test_confs, test_errors, label="Test")
        
        ax.set_title("{}\nIS: {:.2f} CE: {:.2f}".format(key, test_score, test_calib_err))
        ax.set_ylim([0.0, 40.0])
        ax.set_xlabel("Standard Deviation")
        ax.set_ylabel("Mean Absolute Error")

        train_confs, train_accs = regressor_calibration_curve(y_train_pred_mean, y_train, y_train_pred_std)
        calib_ax.plot(train_confs, train_accs, label="Train")

        test_confs, test_accs = regressor_calibration_curve(y_test_pred_mean, y_test, y_test_pred_std)
        calib_ax.plot(test_confs, test_accs, label="Test")

        calib_ax.set_title("Reliability Plot")
        calib_ax.set_xlim([0.0, 1.0])
        calib_ax.set_ylim([0.0, 1.0])
        calib_ax.set_xlabel("Confidence")
        calib_ax.set_ylabel("Accuracy")
        
    plt.legend(loc="upper right")            
    plt.show()