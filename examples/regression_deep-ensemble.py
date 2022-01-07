#!/bin/python3

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor
from keras_uncertainty.losses import regression_gaussian_nll_loss

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def toy_dataset(input):
    output = []

    for inp in input:
        std = 10 if inp > 0 else 2
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

    train_model.compile(loss=regression_gaussian_nll_loss(var), optimizer="adam")

    return train_model, pred_model

if __name__ == "__main__":
    x_train = np.linspace(-4.0, 4.0, num=1200)
    x_test = np.linspace(-7.0, 7.0, 200)

    y_train = toy_dataset(x_train)
    y_test = toy_dataset(x_test)

    model = DeepEnsembleRegressor(mlp_model, 5)
    model.fit(x_train, y_train, epochs=200)

    model.save("regression-ens")

    y_pred_mean, y_pred_std = model.predict(x_test)
    y_pred_mean = y_pred_mean.reshape((-1,))
    y_pred_std = y_pred_std.reshape((-1,))

    print("y pred mean shape: {}, y_pred_std shape: {}".format(y_pred_mean.shape, y_pred_std.shape))

    y_pred_up_1 = y_pred_mean + y_pred_std
    y_pred_down_1 = y_pred_mean - y_pred_std

    y_pred_up_2 = y_pred_mean + 2.0 * y_pred_std
    y_pred_down_2 = y_pred_mean - 2.0 * y_pred_std

    y_pred_up_3 = y_pred_mean + 3.0 * y_pred_std
    y_pred_down_3 = y_pred_mean - 3.0 * y_pred_std
    
    plt.plot(x_test, y_test, '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
    plt.plot(x_test, x_test ** 3, color='red', label="Ground truth 4x**3")

    plt.fill_between(x_test, y_pred_down_3, y_pred_up_3, color=(0, 0, 0.9, 0.2), label="Three Sigma Confidence Interval")
    plt.fill_between(x_test, y_pred_down_2, y_pred_up_2, color=(0, 0, 0.9, 0.5), label="Two Sigma Confidence Interval")
    plt.fill_between(x_test, y_pred_down_1, y_pred_up_1, color=(0, 0, 0.9, 0.7), label="One Sigma Confidence Interval")
        
    plt.legend(loc="upper left")
    plt.title("Deep Ensemble Regression of $x ^ 3 +$ noise with input-dependent noise")

    plt.show()
