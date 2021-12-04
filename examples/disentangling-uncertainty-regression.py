import numpy as np
import math

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor, TwoHeadStochasticRegressor
from keras_uncertainty.layers import DropConnectDense, FlipoutDense, StochasticDropout
from keras_uncertainty.losses import regression_gaussian_nll_loss, regression_gaussian_beta_nll_loss
from keras_uncertainty.datasets import toy_regression_monotonic_sinusoid

import matplotlib.pyplot as plt

def train_stochastic_model(trunk_model, x_train, y_train, domain, epochs=200, dense_layer=Dense, **kwargs):
    inp = Input(shape=(1,))
    x = trunk_model(inp)
    mean = dense_layer(1, activation="linear", **kwargs)(x)
    var = dense_layer(1, activation="softplus", **kwargs)(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=regression_gaussian_beta_nll_loss(var, beta=2.0), optimizer="adam")
    train_model.fit(x_train, y_train, verbose=2, epochs=epochs)

    st_model = TwoHeadStochasticRegressor(pred_model)
    pred_mean, pred_ale, pred_epi = st_model.predict(domain, num_samples=NUM_SAMPLES, disentangle_uncertainty=True)

    return pred_mean, pred_ale, pred_epi

def classic_model(prob=0.2):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(1,)))
    model.add(Dense(32, activation="relu"))

    return model

def dropout_model(prob=0.2):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(1,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))

    return model

def dropconnect_model(prob=0.05):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(1,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))

    return model

def flipout_model(kl_weight):
    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, activation="relu", input_shape=(1,)))
    model.add(FlipoutDense(32, kl_weight, activation="relu"))

    return model

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        inp = Input(shape=(1,))
        x = Dense(32, activation="relu")(inp)
        x = Dense(32, activation="relu")(x)
        mean = Dense(1, activation="linear")(x)
        var = Dense(1, activation="softplus")(x)

        train_model = Model(inp, mean)
        pred_model = Model(inp, [mean, var])

        train_model.compile(loss=regression_gaussian_beta_nll_loss(var, beta=2.0), optimizer="adam")

        return train_model, pred_model
    
    model = DeepEnsembleRegressor(model_fn, num_estimators=10)
    model.fit(x_train, y_train, verbose=2, epochs=EPOCHS)
    pred_mean, pred_ale, pred_epi = model.predict(domain, disentangle_uncertainty=True)

    return pred_mean, pred_ale, pred_epi

def train_classic_model(x_train, y_train, domain):
    model = classic_model()

    return train_stochastic_model(model, x_train, y_train, domain, epochs=EPOCHS)

def train_dropout_model(x_train, y_train, domain):
    model = dropout_model()

    return train_stochastic_model(model, x_train, y_train, domain, epochs=EPOCHS)

def train_dropconnect_model(x_train, y_train, domain):
    model = dropconnect_model()

    return train_stochastic_model(model, x_train, y_train, domain, dense_layer=DropConnectDense, epochs=EPOCHS, prob=0.10)

def train_flipout_model(x_train, y_train, domain):
    kl_weight = 32 / x_train.shape[0]
    model = flipout_model(kl_weight)

    return train_stochastic_model(model, x_train, y_train, domain, dense_layer=FlipoutDense, epochs=EPOCHS_FLIPOUT, kl_weight=kl_weight)

def plot_regression_uncertainty(axis, x, y_mean, y_std, title=""):
    y_mean = y_mean.reshape((-1,))
    y_std = y_std.reshape((-1,))

    y_std_up1 = y_mean + y_std
    y_std_down1 = y_mean - y_std

    axis.fill_between(x, y_std_down1, y_std_up1, color=(0, 0, 0.9, 0.7))
    axis.plot(x, y_mean, '.', color=(0, 0.9, 0.0, 0.8), markersize=0.2)
    axis.set_title(title)

    
    axis.set_ylim([-20.0, 20.0])
    axis.axvline(x= 10.0, color="black", linestyle="dashed")
    axis.get_xaxis().set_ticks([])
    axis.get_yaxis().set_ticks([])    

METHODS = {
    "Classical NN": train_classic_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
}

NUM_SAMPLES = 30
EPOCHS = 700
EPOCHS_FLIPOUT = 700

if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=3, ncols=len(METHODS.keys()), figsize=(20, 8))
    methods = list(METHODS.keys())

    x_train, y_train, x_test, y_test = toy_regression_monotonic_sinusoid(num_samples=1000, ood_samples=200)

    domain = np.concatenate([x_train, x_test])
    domain = domain.reshape((-1, 1))

    for i, key in enumerate(methods):
        ax_total = axes[0][i]
        ax_ale = axes[1][i]
        ax_epi = axes[2][i]

        y_pred_mean, y_pred_std_ale, y_pred_std_epi = METHODS[key](x_train, y_train, domain)
        y_pred_std = y_pred_std_ale + y_pred_std_epi

        plot_regression_uncertainty(ax_total, domain.ravel(), y_pred_mean, y_pred_std, title=key)
        plot_regression_uncertainty(ax_epi, domain.ravel(), y_pred_mean, y_pred_std_epi)
        plot_regression_uncertainty(ax_ale, domain.ravel(), y_pred_mean, y_pred_std_ale)

        if i == 0:
            ax_total.set_ylabel("Total\nUncertainty")
            ax_ale.set_ylabel("Aleatoric\nUncertainty")
            ax_epi.set_ylabel("Epistemic\nUncertainty")

    plt.savefig("uncertainty-toy-regression.png", bbox_inches="tight")
    plt.show()