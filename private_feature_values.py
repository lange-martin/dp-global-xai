import numpy as np
from DP_AQ.approximate_quantiles_algo import approximate_quantiles_algo
from matplotlib import pyplot as plt


class DPQuantiles:
    def __init__(self, x, x_min, x_max, seed):
        self.x = np.sort(x)
        self.x_min = x_min
        self.x_max = x_max
        self.rng = np.random.default_rng(seed=seed)

    def get_dp_quantiles(self, q_ratios, epsilon):
        q_list = approximate_quantiles_algo(self.x, q_ratios, (self.x_min, self.x_max), epsilon,
                                            swap=True, seed=self.rng)

        return np.array(q_list)

    def test_dp_quantiles(self, q_ratios, epsilon):
        private_quantiles = self.get_dp_quantiles(q_ratios, epsilon)

        n = len(self.x)
        indices_of_quantiles = np.array([min(int(q_ratio * n), n - 1) for q_ratio in q_ratios])
        actual_quantiles = np.array(self.x[indices_of_quantiles])

        print(f'Private Quantiles: {private_quantiles}')
        print(f'Actual  Quantiles: {actual_quantiles}')
