#!/bin/python3

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Layer
from keras import ops

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleRegressor, TwoHeadModel
from keras_uncertainty.losses import regression_gaussian_nll_loss
from keras_uncertainty.datasets import toy_regression_monotonic_sinusoid

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

#keras.config.disable_traceback_filtering()

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
    mean = Dense(1, activation="linear", name="mean")(x)
    var = Dense(1, activation="softplus", name="var")(x)

    model = TwoHeadModel(inputs=inp, outputs=[mean, var])
    model.compile(loss=regression_gaussian_nll_loss(), optimizer="adamw")

    return model

from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    #x_train = np.linspace(-4.0, 4.0, num=1200)
    #x_test = np.linspace(-7.0, 7.0, 200)

    #y_train = toy_dataset(x_train)
    #y_test = toy_dataset(x_test)

    x_train, y_train, x_test, y_test = toy_regression_monotonic_sinusoid(num_samples=1000, ood_samples=200)

    x_test = np.concatenate([x_train, x_test], axis=0)
    y_test = np.concatenate([y_train, y_test], axis=0)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_train = x_scaler.fit_transform(x_train.reshape(-1, 1))
    x_test = x_scaler.transform(x_test.reshape(-1, 1))

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    #x_train, y_train, x_test, y_test = toy_regression_monotonic_sinusoid(num_samples=1000, ood_samples=200)

    model = DeepEnsembleRegressor(mlp_model, 5)
    model.fit(x_train, y_train, epochs=500, batch_size=128)

    #model.save("regression-ens")
    
    y_pred_mean, y_pred_std = model.predict(x_test)
    y_pred_mean = y_pred_mean.reshape((-1,))
    y_pred_std = y_pred_std.reshape((-1,))

    x_test = x_test.flatten() 

    print("y pred mean shape: {}, y_pred_std shape: {}".format(y_pred_mean.shape, y_pred_std.shape))

    y_pred_up_1 = y_pred_mean + y_pred_std
    y_pred_down_1 = y_pred_mean - y_pred_std

    y_pred_up_2 = y_pred_mean + 2.0 * y_pred_std
    y_pred_down_2 = y_pred_mean - 2.0 * y_pred_std

    y_pred_up_3 = y_pred_mean + 3.0 * y_pred_std
    y_pred_down_3 = y_pred_mean - 3.0 * y_pred_std
    
    plt.plot(x_test, y_test, '.', color=(0, 0.9, 0.0, 0.8), markersize=12, label="Ground truth Points")
    #plt.plot(x_test, x_test ** 3, color='red', label="Ground truth 4x**3")

    plt.fill_between(x_test, y_pred_down_3, y_pred_up_3, color=(0, 0, 0.9, 0.2), label="Three Sigma Confidence Interval")
    plt.fill_between(x_test, y_pred_down_2, y_pred_up_2, color=(0, 0, 0.9, 0.5), label="Two Sigma Confidence Interval")
    plt.fill_between(x_test, y_pred_down_1, y_pred_up_1, color=(0, 0, 0.9, 0.7), label="One Sigma Confidence Interval")
        
    plt.legend(loc="upper left")
    plt.title("Deep Ensemble Regression of $x ^ 3 +$ noise with input-dependent noise")

    plt.show()
