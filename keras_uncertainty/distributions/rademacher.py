import keras
from keras import random, ops

def sample(shape):
    inp = ops.ones(shape)
    samples = random.dropout(inp, rate=0.5)

    return 2.0 * samples - 1.0