import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

import keras_uncertainty
from keras_uncertainty.models import MCDropoutClassifier, DeepEnsembleClassifier, StochasticClassifier, GradientClassificationConfidence
from keras_uncertainty.layers import duq_training_loop, add_gradient_penalty, add_l2_regularization
from keras_uncertainty.layers import DropConnectDense, VariationalDense, RBFClassifier, FlipoutDense
from keras_uncertainty.utils import numpy_entropy

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

def train_standard_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    preds = model.predict(domain)
    entropy = uncertainty(preds)

    return entropy

def train_dropout_model(x_train, y_train, domain, prob=0.2):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(prob))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    mc_model = MCDropoutClassifier(model)
    preds = mc_model.predict(domain, num_samples=100)
    entropy = uncertainty(preds)

    return entropy

def train_dropconnect_model(x_train, y_train, domain, prob=0.05):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))
    model.add(DropConnectDense(2, activation="softmax", prob=prob))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, verbose=2, epochs=50)

    mc_model = MCDropoutClassifier(model)
    preds = mc_model.predict(domain, num_samples=100)
    entropy = uncertainty(preds)

    return entropy

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(2,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    model = DeepEnsembleClassifier(model_fn, num_estimators=10)
    model.fit(x_train, y_train, verbose=2, epochs=50)
    preds = model.predict(domain)
    entropy = uncertainty(preds)

    return entropy

def train_bayesbackprop_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(VariationalDense(32, kl_weight, **prior_params, prior=False, activation="relu", input_shape=(2,)))
    model.add(VariationalDense(32, kl_weight, **prior_params, prior=False, activation="relu"))
    model.add(VariationalDense(2, kl_weight, **prior_params, prior=False, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", "sparse_categorical_crossentropy"])

    model.fit(x_train, y_train, verbose=2, epochs=1000)
    st_model = StochasticClassifier(model)

    preds = st_model.predict(domain, num_samples=100)
    entropy = uncertainty(preds)

    return entropy

def train_flipout_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, **prior_params, prior=False, bias_distribution=False, activation="relu", input_shape=(2,)))
    model.add(FlipoutDense(32, kl_weight, **prior_params, prior=False, bias_distribution=False, activation="relu"))
    model.add(FlipoutDense(2, kl_weight, **prior_params, prior=False, bias_distribution=False, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", "sparse_categorical_crossentropy"])

    model.fit(x_train, y_train, verbose=2, epochs=300)
    st_model = StochasticClassifier(model)

    preds = st_model.predict(domain, num_samples=NUM_SAMPLES)
    entropy = uncertainty(preds)

    return entropy

def train_duq_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(RBFClassifier(2, 0.1, centroid_dims=2, trainable_centroids=True))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    add_gradient_penalty(model, lambda_coeff=0.5)
    add_l2_regularization(model)

    y_train = to_categorical(y_train, num_classes=2)

    model.fit(x_train, y_train, verbose=2, epochs=50)

    preds = model.predict(domain)
    confidence = np.max(preds, axis=1)

    return confidence

def train_gradient_model(x_train, y_train, domain):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    y_train = to_categorical(y_train, num_classes=2)
    model.fit(x_train, y_train, verbose=2, epochs=50)

    grad_model = GradientClassificationConfidence(model, aggregation="l1_norm")
    conf = grad_model.predict(domain)

    return np.array(conf)

METHODS = {
    "Classical NN": train_standard_model,
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    "5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
    "DUQ": train_duq_model,
    "Gradient L1": train_gradient_model
}

NUM_SAMPLES = 30

if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=1, ncols=len(METHODS.keys()), figsize=(20, 3))
    methods = list(METHODS.keys())

    x, y = make_moons(n_samples=500, noise=0.1, random_state=749)
    min_x, max_x = [-2, -2] , [3, 2]
    res = 0.08

    xx, yy = np.meshgrid(np.arange(min_x[0], max_x[0], res), np.arange(min_x[1], max_x[1], res))
    domain = np.c_[xx.ravel(), yy.ravel()]

    for i, key in enumerate(methods):
        ax = axes[i]

        domain_conf = METHODS[key](x, y, domain)
        domain_conf = domain_conf.reshape(xx.shape)

        cont = ax.contourf(xx, yy, domain_conf)
        scat = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="binary")
        ax.set_title(key)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    plt.savefig("uncertainty-two-moons.png", bbox_inches="tight")
    plt.show()