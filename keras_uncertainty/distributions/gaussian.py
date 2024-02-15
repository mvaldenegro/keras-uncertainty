import numpy as np
import math

import keras
from keras import random
from keras import ops

NegHalfLog2PI = -0.5 * math.log(2.0 * math.pi)
InvSqrt2PI = 1.0 / (math.sqrt(2.0 * math.pi))

def log_probability(x, mu, sigma):
    return NegHalfLog2PI - ops.log(sigma) - 0.5 * ops.square((x - mu) / sigma)

def probability(x, mu, sigma):
    x= ops.square((x - mu) / sigma)

    return InvSqrt2PI * (1.0 / sigma) * ops.exp(-0.5 * x)

class GaussianDistribution:
    def __init__(self, mean, std, shape):
        self.mean = mean
        self.std = std
        self.shape = shape

    def sample(self):
        return random.normal(self.shape, self.mean, self.std)

    def sample_perturbation(self):
        return random.normal(self.shape, ops.zeros(self.shape), self.std)

    def log_probability(self, x):
        return NegHalfLog2PI - ops.log(self.std) - 0.5 * ops.square((x - self.mean) / self.std)
