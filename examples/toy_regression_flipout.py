import math

import keras_uncertainty
from keras_uncertainty.layers import FlipoutDense
from keras_uncertainty.models import StochasticRegressor
Input = keras_uncertainty.backend.layers.Input
Model = keras_uncertainty.backend.models.Model

import numpy as np
import matplotlib.pyplot as plt

def mlp_model(num_batches, prior_params):
    inp = Input(shape=(1,))
    kl_weight = 1.0 / num_batches

    x = FlipoutDense(16, kl_weight, activation="relu", **prior_params)(inp)
    x = FlipoutDense(16, kl_weight, activation="relu", **prior_params)(x)
    #x = BayesByBackpropDense(16, kl_weight, activation="relu", **prior_params)(x)

    out = FlipoutDense(1, kl_weight, **prior_params)(x)
    model = Model(inp, out)
    
    return model

def toy_function(input):
    output = []

    for inp in input:
        std = max(0.15 / (1.0 + math.exp(-inp)), 0)

        out = math.sin(inp) + np.random.normal(0, std)
        output.append(10 * out)

    return np.array(output)

BATCH_SIZE = 32
FWD_PASSES = 10
prior_params = {
    'prior_sigma_1': 1.0, 
    'prior_sigma_2': 0.5, 
    'prior_pi': 0.5 
}

if __name__ == "__main__":
    x_train = np.linspace(-math.pi, math.pi, num=500)
    x_test = np.concatenate([np.linspace(-2.0 * math.pi, -math.pi, 100), np.linspace(math.pi, 2.0 * math.pi, 100)])

    y_train = toy_function(x_train)
    y_test = toy_function(x_test)

    full_test = np.linspace(-2.0 * math.pi, 2.0 * math.pi, 200)

    num_batches = x_train.shape[0] / BATCH_SIZE

    model = mlp_model(num_batches, prior_params)
    model.summary()
    model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])

    model.fit(x_train, y_train, epochs=800, verbose=1)

    st_model = StochasticRegressor(model)

    y_pred_mean, y_pred_std = st_model.predict(full_test, num_samples=100)
    y_pred_mean = y_pred_mean.ravel()
    y_pred_std = y_pred_std.ravel()

    plt.plot(full_test, y_pred_mean, color='red')
    plt.scatter(x_train, y_train    , marker='+', color='blue')
    plt.fill_between(full_test, y_pred_mean + y_pred_std, y_pred_mean - y_pred_std, alpha=0.5)
    plt.show()