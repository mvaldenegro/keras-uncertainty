import numpy as np
import math
import keras.backend as K

NegHalfLog2PI = -0.5 * math.log(2.0 * math.pi)
InvSqrt2PI = 1.0 / (math.sqrt(2.0 * math.pi))

def log_probability(x, mu, sigma):
    return NegHalfLog2PI - K.log(sigma) - 0.5 * K.square((x - mu) / sigma)

def probability(x, mu, sigma):
    x= K.square((x - mu) / sigma)

    return InvSqrt2PI * (1.0 / sigma) * K.exp(-0.5 * x)