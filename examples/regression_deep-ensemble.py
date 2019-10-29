#!/bin/python3

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor, deep_ensemble_regression_nll_loss

import matplotlib.pyplot as plt

def toy_dataset(input):
    output = []

    for inp in input:
        std = 10 if inp > 0 else 1
        out = inp ** 3 + np.random.normal(0, std)
        output.append(out)

    return np.array(output)

def mlp_model():
    inp = Input(shape=(1,))
    x = Dense(10, activation="relu")(inp)
    x = Dense(20, activation="relu")(x)
    x = Dense(30, activation="relu")(x)
    mean = Dense(1, activation="linear")(x)
    var = Dense(1, activation="softplus")(x)

    train_model = Model(inp, mean)
    pred_model = Model(inp, [mean, var])

    train_model.compile(loss=deep_ensemble_regression_nll_loss(var), optimizer="adam")

    return train_model, pred_model

if __name__ == "__main__":
    x_train = np.linspace(-3.0, 3.0, num=1200)
    x_test = np.linspace(-6.0, 6.0, 200)

    y_train = toy_dataset(x_train)
    y_test = toy_dataset(x_test)

    model = DeepEnsembleRegressor(mlp_model, 10)
    model.fit(x_train, y_train, epochs=200)

    y_pred_mean, y_pred_std = model.predict(x_test)

    y_pred_up = y_pred_mean + y_pred_std
    y_pred_down = y_pred_mean - y_pred_std

    plt.plot(x_test, x_test ** 3, color='red', label="Ground truth x**3")
    plt.plot(x_test, y_test, '.', color='purple', markersize=12, label="Ground truth Points")
    plt.plot(x_test, y_pred_up, color='blue', label="Confidence Interval")
    plt.plot(x_test, y_pred_down, color='blue')

    plt.legend(loc="upper right")

    plt.show()
