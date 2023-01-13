#!/bin/python3
# This is a ensemble regression example, but using a custom training loop through GradientTape
# Ensembles are callable and thus gradients can be propagated thru them, or even trained in a end-to-end fashion

import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input

import keras_uncertainty
from keras_uncertainty.models import SimpleEnsemble
from keras_uncertainty.utils import make_batches

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import trange
import math

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

    train_model = Model(inp, mean)
    train_model.compile(loss="mean_squared_error", optimizer="adam")

    return train_model

EPOCHS = 300
BATCH_SIZE = 32

if __name__ == "__main__":
    x_train = np.linspace(-4.0, 4.0, num=1200)
    x_test = np.linspace(-7.0, 7.0, 200)

    y_train = toy_dataset(x_train)
    y_test = toy_dataset(x_test)

    model = SimpleEnsemble(mlp_model, 5)
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam()

    num_batches = math.ceil(x_train.shape[0] / BATCH_SIZE)

    for epoch in range(EPOCHS):
        t = trange(num_batches, desc='Epoch {} / {}'.format(epoch, EPOCHS))
        losses = []

        for step, (x_batch, y_batch) in enumerate(make_batches(x_train, y_train)):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss))
            
            t.set_description('Epoch {} / {} - Loss {:.3f}'.format(epoch + 1, EPOCHS, np.mean(losses)))
            t.refresh()

    y_pred_mean, y_pred_std = model.predict(x_test)
    y_pred_mean = y_pred_mean.reshape((-1,))
    y_pred_std = y_pred_std.reshape((-1,))

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
    plt.axvline(x= -4.0, color="black", linestyle="dashed")
    plt.axvline(x=  4.0, color="black", linestyle="dashed")
        
    plt.legend(loc="upper left")
    plt.title("Ensemble Regression of $x ^ 3 +$ noise with input-dependent noise")

    plt.show()
