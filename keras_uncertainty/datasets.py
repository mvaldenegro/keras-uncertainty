import numpy as np
import math

# Synthetic datasets useful for demos or evaluation of uncertainty quantification

def sample_function(function, range, size=500):
    x = np.linspace(range[0], range[1], num=size)
    y = []

    for v in x:
        y.append(function(v))

    y = np.array(y)

    return x, y
        
def toy_regression_sinusoid(num_samples=1000, ood_samples=0):
    """
        Returns samples from a sinusoid with added heteroscendatic gaussian noise.

        The function is sin(x) + e, 
    """
    
    def fn(inp):
        std = max(0.15 / (1.0 + math.exp(-inp)), 0.0)

        return math.sin(inp) + np.random.normal(0, std)

    x_train, y_train = sample_function(fn, [-4.0, 4.0], size=num_samples)

    if ood_samples > 0:
        x_testA, y_testA = sample_function(fn, [4, 7], size=ood_samples // 2)
        x_testB, y_testB = sample_function(fn, [-7, -4], size=ood_samples // 2)
        x_test = np.concatenate([x_testA, x_testB])
        y_test = np.concatenate([y_testA, y_testB])

        return x_train, y_train, x_test, y_test
    
    return x_train, y_train

def toy_regression_monotonic_sinusoid(num_samples=500, ood_samples=0, disentangle_uncertainty=False):
    """
        Returns samples from a sinusoid with increasing frequency and added heteroscendatic gaussian noise.

        The function is x sin(x) + e1 x + e2, where e1 and e2 are samples from a Gaussian with mean = 0 and std = 0.3.
    """
    def fn_truth(x):
        return x * math.sin(x)

    def fn_noise(x):
        e1, e2 = np.random.normal(0.0, 0.3), np.random.normal(0.0, 0.3)

        return e1 * x + e2

    x_train, y_train_epi = sample_function(fn_truth, [0.0, 10.0], size=num_samples)
    x_train, y_train_ale = sample_function(fn_noise, [0.0, 10.0], size=num_samples)

    if ood_samples > 0:
        x_test, y_test_epi = sample_function(fn_truth, [10.0, 15.0], size=ood_samples)
        x_test, y_test_ale = sample_function(fn_noise, [10.0, 15.0], size=ood_samples)

        if not disentangle_uncertainty:
            return x_train, y_train_epi + y_train_ale, x_test, y_test_epi + y_test_ale
        
        return x_train, y_train_epi, y_train_ale, x_test, y_test_epi, y_test_ale

    if not disentangle_uncertainty:
        return x_train, y_train_epi + y_train_ale
    
    return x_train, y_train_epi, y_train_ale