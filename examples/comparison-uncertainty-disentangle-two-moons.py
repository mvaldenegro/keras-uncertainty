import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical

import keras_uncertainty
from keras_uncertainty.models import DeepEnsembleClassifier, StochasticClassifier
from keras_uncertainty.models import DisentangledStochasticClassifier
from keras_uncertainty.layers import StochasticDropout
from keras_uncertainty.layers import DropConnectDense, VariationalDense, FlipoutDense, SamplingSoftmax
from keras_uncertainty.utils import numpy_entropy

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True
})

from scipy import stats

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

def two_head_model(trunk_model, num_classes=2, num_samples=100):
    inp = Input(shape=(2,))
    x = trunk_model(inp)
    logit_mean = Dense(num_classes, activation="linear")(x)
    logit_var = Dense(num_classes, activation="softplus")(x)
    probs = SamplingSoftmax(num_samples=num_samples, variance_type="linear_std")([logit_mean, logit_var])
    
    train_model = Model(inp, probs, name="train_model")
    pred_model = Model(inp, [logit_mean, logit_var], name="pred_model")

    train_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return train_model, pred_model

def train_eval_stochastic_model(trunk_model, x_train, y_train, domain, epochs=300):
    train_model, pred_model = two_head_model(trunk_model)
    train_model.fit(x_train, y_train, verbose=2, epochs=epochs, batch_size=BATCH_SIZE)

    fin_model = DisentangledStochasticClassifier(pred_model, epi_num_samples=NUM_SAMPLES)
    pred_mean, pred_ale_std, pred_epi_std = fin_model.predict(domain, batch_size=BATCH_SIZE)
    ale_entropy = uncertainty(pred_ale_std)
    epi_entropy = uncertainty(pred_epi_std)

    return ale_entropy, epi_entropy

def train_dropout_model(x_train, y_train, domain, prob=0.5):
    model = Sequential()
    model.add(Dense(32, activation="relu", input_shape=(2,)))
    model.add(StochasticDropout(prob))
    model.add(Dense(32, activation="relu"))
    model.add(StochasticDropout(prob))

    return train_eval_stochastic_model(model, x_train, y_train, domain, epochs=500)

def train_dropconnect_model(x_train, y_train, domain, prob=0.25):
    model = Sequential()
    model.add(DropConnectDense(32, activation="relu", input_shape=(2,), prob=prob))
    model.add(DropConnectDense(32, activation="relu", prob=prob))    

    return train_eval_stochastic_model(model, x_train, y_train, domain, epochs=500)

def train_ensemble_model(x_train, y_train, domain):
    def model_fn():
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(2,)))
        model.add(Dense(32, activation="relu"))

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    model = DeepEnsembleClassifier(model_fn, num_estimators=5)
    model.fit(x_train, y_train, verbose=2, epochs=50)
    pred_mean, pred_ale_std, pred_epi_std = model.predict(domain, disentangle_uncertainty=True)
    ale_entropy = uncertainty(pred_ale_std)
    epi_entropy = uncertainty(pred_epi_std)

    return ale_entropy, epi_entropy

def train_flipout_model(x_train, y_train, domain):
    num_batches = x_train.shape[0] / 32
    kl_weight = 1.0 / num_batches
    prior_params = {
        'prior_sigma_1': 5.0, 
        'prior_sigma_2': 2.0, 
        'prior_pi': 0.5
    }

    model = Sequential()
    model.add(FlipoutDense(32, kl_weight, **prior_params, activation="relu", input_shape=(2,)))
    model.add(FlipoutDense(32, kl_weight, **prior_params, activation="relu"))

    return train_eval_stochastic_model(model, x_train, y_train, domain, epochs=1000)

METHODS = {
    "Dropout": train_dropout_model,
    "DropConnect": train_dropconnect_model,
    #"5 Ensembles": train_ensemble_model,
    "Flipout": train_flipout_model,
}

NUM_SAMPLES = 50
BATCH_SIZE = 256

if __name__ == "__main__":
    fig, axes = plt.subplots(ncols=len(METHODS.keys()), nrows=2, figsize=(6, 10), squeeze=False)    
    methods = list(METHODS.keys())
  
    min_x, max_x = [-2, -2] , [3, 2]
    res = 0.05

    xx, yy = np.meshgrid(np.arange(min_x[0], max_x[0], res), np.arange(min_x[1], max_x[1], res))
    domain = np.c_[xx.ravel(), yy.ravel()]

    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap
    cmap = pl.cm.binary
    my_cmap = cmap(np.arange(cmap.N))
    #my_cmap[:, 0] = 0.0
    my_cmap[:, -1] = 0.7
    my_cmap = ListedColormap(my_cmap)

    x, y = make_moons(n_samples=500, noise=0.1, random_state=749)

    for j, method_name in enumerate(methods):
        ax_ale = axes[0][j]
        ax_epi = axes[1][j]

        ale_entropy, epi_entropy = METHODS[method_name](x, y, domain)
        ale_entropy = ale_entropy.reshape(xx.shape)
        epi_entropy = epi_entropy.reshape(xx.shape)

        cf_ale = ax_ale.contourf(xx, yy, ale_entropy, vmin=0.3, vmax=0.75, antialiased=True)
        ax_ale.scatter(x[:, 0], x[:, 1], c=y, cmap=my_cmap)
        ax_ale.get_xaxis().set_ticks([])
        ax_ale.get_yaxis().set_ticks([])
        ax_ale.autoscale(False)

        cf_epi = ax_epi.contourf(xx, yy, epi_entropy, vmin=0.3, vmax=0.75, antialiased=True)
        ax_epi.scatter(x[:, 0], x[:, 1], c=y, cmap=my_cmap)
        ax_epi.get_xaxis().set_ticks([])
        ax_epi.get_yaxis().set_ticks([])
        ax_epi.autoscale(False)

        if j == 0:
            ax_ale.set_ylabel("Aleatoric")
            ax_epi.set_ylabel("Epistemic")
        
        ax_ale.set_title(method_name)
        
        if not (j == len(METHODS) - 1):
            ax_ale.get_xaxis().set_ticks([])
            ax_epi.get_xaxis().set_ticks([])
            
    #plt.savefig("output.pdf", bbox_inches="tight")
    #fig.tight_layout()
    plt.show()