import keras_uncertainty.backend as K

def sample(shape):
    samples = K.random_binomial(shape, p=0.5)

    return 2.0 * samples - 1.0