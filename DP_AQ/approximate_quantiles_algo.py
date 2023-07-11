
# Code by Haim Kaplan, Shachar Schnapp and Uri Stemmer
# https://github.com/ShacharSchnapp/DP_AQ
# Paper: Kaplan, Haim, Shachar Schnapp, and Uri Stemmer. "Differentially Private Approximate Quantiles."
#        arXiv preprint arXiv:2110.05429 (2021).

import numpy as np
from copy import deepcopy
from DP_AQ.single_quantile_algo import single_quantile


def get_epsilon(n_quantiles, epsilon, swap, cdp):
    layers = np.log2(n_quantiles) + 1

    if swap:
        composition = 2 * layers
    else:
        composition = layers

    if cdp:
        epsilon = np.sqrt((2 * epsilon) / composition)
    else:
        epsilon = epsilon / composition

    return epsilon


def split_by_number(array, m):
    return array[array <= m], array[array >= m]


def gaussian_noise(array, bounds, rng, scale=0.00001):
    data = deepcopy(array)
    return np.sort(data + rng.normal(0, scale, len(array))), (bounds[0] - 4 * scale, bounds[1] + 4 * scale)


def approximate_quantiles_algo(array, quantiles, bounds, epsilon, swap=False, cdp=False, seed=None):
    rng = np.random.default_rng(seed=seed)

    epsilon = get_epsilon(len(quantiles), epsilon, swap, cdp)
    array, bounds = gaussian_noise(array, bounds, rng)

    def algo_helper(array, quantiles, bounds):
        m = len(quantiles)
        a, b = bounds
        if m == 0:
            return []

        if m == 1:
            return [
                single_quantile(array, bounds, quantiles[0], epsilon=epsilon, swap=False, rng=rng)]

        q_mid = quantiles[m // 2]
        v = single_quantile(array, bounds, q_mid, epsilon=epsilon, swap=False, rng=rng)
        d_l, d_u = split_by_number(array, v)
        q_l, q_u = np.array_split(quantiles[quantiles != q_mid], 2)
        q_l, q_u = q_l / q_mid, (q_u - q_mid) / (1 - q_mid)

        return algo_helper(d_l, q_l, (a, v)) + [v] + algo_helper(d_u, q_u, (v, b))

    return algo_helper(np.sort(array), quantiles, bounds)
