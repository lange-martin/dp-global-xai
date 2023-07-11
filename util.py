import numpy as np

def inverse_laplace(p, mean, scale):
    return mean - scale * np.sign(p - 0.5) * np.log(1 - 2 * np.abs(p - 0.5))