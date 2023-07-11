import numpy as np
import logging

from data_loader import HeartDisease
from data_loader import BikeSharing
from data_loader import AdultIncome
from explainers.accumulated_local_effects.ale import AccumulatedLocalEffects
from explainers.plot_explainer import DPFeatureEffectExplainer
from private_feature_values import DPQuantiles
from util import inverse_laplace


# DP ALE
class DPAccumulatedLocalEffects(DPFeatureEffectExplainer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('ale')
        self.explainer_name = 'DP ALE (ε=1)'

        # default values
        self.epsilon_quantiles = 1 / 3
        self.epsilon_effects = 1 / 3
        self.epsilon_histogram = 1 / 3

    def set_privacy_parameters(self, epsilon, is_int, x_min, x_max, y_min, y_max, seed):
        self.epsilon_quantiles = 4 * epsilon / 9
        self.epsilon_effects = 4 * epsilon / 9
        self.epsilon_histogram = epsilon / 9

        self.explainer_name = f'DP ALE (ε={epsilon})'

        self.is_int = is_int

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.rng = np.random.default_rng(seed=seed)

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=20, class_num=None, verbose=False):
        x_train_np = x_train.to_numpy()
        x_train_np = x_train_np.astype(float)

        # add noise if numeric feature to ensure that each sample has a different value (this should result in bins
        # of roughly equal size)
        if not is_cat:
            noise_range = np.abs(self.x_max - self.x_min) / len(x_train_np)
            x_train_np[:, feature_index] += self.rng.uniform(low=0, high=noise_range, size=len(x_train_np))

        # sort samples by respective feature
        x_train_np = x_train_np[np.argsort(x_train_np[:, feature_index])]

        num_samples = len(x_train_np)

        self.is_categorical = is_cat
        self.feature_name = x_train.columns[feature_index]
        self.feature_values = x_train_np[:, feature_index]

        if is_cat:
            border_indices, samples_per_x_value, x_values, num_x_values = self._calc_x_values_cat()
        else:
            border_indices, samples_per_x_value, x_values, num_x_values = self._calc_x_values_num(num_samples, num_x_values)

        # array for effects of feature values (y-axis in plot)
        effects = np.zeros(shape=num_x_values)

        # calculations for scale of noise
        pred_range = self.y_max - self.y_min
        # "replace sample" sensitivity (before calculating mean, this is only for sum)
        scale = 2 * pred_range * (1 / self.epsilon_effects)
        noise = np.zeros(shape=num_x_values)

        for cur_effect in range(1, num_x_values):
            index_start = border_indices[cur_effect - 1]
            index_end = border_indices[cur_effect]
            lower_value = x_values[cur_effect - 1]
            upper_value = x_values[cur_effect]

            if index_start != index_end:
                # copy all samples of this quantile twice to alter them with the lower and upper border
                x_train_low = x_train_np[index_start:index_end].copy()
                x_train_high = x_train_np[index_start:index_end].copy()
                x_train_low[:, feature_index] = lower_value
                x_train_high[:, feature_index] = upper_value

                # get predictions for altered samples
                if class_num is None:
                    preds_high = pred_func(x_train_high)
                    preds_low = pred_func(x_train_low)
                else:
                    preds_high = pred_func(x_train_high)[:, class_num]
                    preds_low = pred_func(x_train_low)[:, class_num]

                # clip predictions according to given maximum and minimum predictions
                np.clip(preds_high, self.y_min, self.y_max, out=preds_high)
                np.clip(preds_low, self.y_min, self.y_max, out=preds_low)

                # difference in prediction for each altered sample in the quantile
                pred_diffs = preds_high - preds_low
            else:
                pred_diffs = np.zeros(shape=(1,))

            effects[cur_effect] = effects[cur_effect] + np.sum(pred_diffs)
            noise[cur_effect] = self.rng.laplace(scale=scale)
            effects[cur_effect] = effects[cur_effect] + noise[cur_effect]
            effects[cur_effect] = effects[cur_effect] / samples_per_x_value[cur_effect - 1]

            # calculate effect by adding previous effect and average prediction difference (relative to ideal number
            # of samples per quantile. This is necessary so that sensitivity of the function remains low).
            if cur_effect > 0:
                effects[cur_effect] = effects[cur_effect] + effects[cur_effect - 1]

        # center the ALE
        effects = effects - np.mean(effects)

        # calculate values for feature distribution histogram
        self._calc_histogram(num_x_values)

        self.x_values = x_values
        self.y_values = effects

        return self

    def _calc_x_values_cat(self):
        # print effect of each category for categorical feature (disregard setting of num_x_values)
        categories = np.unique(self.feature_values)
        num_x_values = categories.shape[0]

        # create one interval per category for a categorical feature
        border_indices = np.zeros(shape=num_x_values)
        border_indices[1:] = np.where(self.feature_values[:-1] != self.feature_values[1:])[0] + 1
        border_indices = border_indices.astype(int)

        # use privacy budget of quantiles for noisy min of categorical counts
        self.epsilon_effects = self.epsilon_quantiles + self.epsilon_effects
        # it's simple to show that report noisy min can be reduced to report noisy max (invert the counting queries
        # to count every category except the chosen one and use report noisy max)
        self.y_values_hist = np.empty(categories.shape)
        for i, category in enumerate(categories):
            self.y_values_hist[i] = len(np.nonzero(self.feature_values == category)[0])

        # add noise:
        self.y_values_hist = self.y_values_hist + \
                             self.rng.laplace(scale=1 / self.epsilon_histogram, size=len(categories))

        # this will be used as the denominators when computing mean of effect
        # also used for sensitivity analysis in the end
        # if the count of the smallest bin is smaller than 1/5 of a uniform distribution, then we cap the value
        # at one fifth of the uniform distribution in order not to add too much noise
        samples_per_x_value = self.y_values_hist

        x_values = categories

        return border_indices, samples_per_x_value, x_values, num_x_values

    def _calc_x_values_num(self, num_samples, num_x_values):
        # get private quantile values for numeric features
        quantile_ratios = np.array([i / (num_x_values - 1) for i in range(num_x_values)])
        x_values = DPQuantiles(self.feature_values, self.x_min, self.x_max, self.rng)\
            .get_dp_quantiles(quantile_ratios, self.epsilon_quantiles)

        border_indices = np.zeros(shape=num_x_values, dtype=int)
        for i in range(num_x_values):
            above_quantile_bools = self.feature_values >= x_values[i]
            if above_quantile_bools.any():
                # first index where the condition is true, so the first value which is larger than the quantile value
                border_indices[i] = np.argmax(above_quantile_bools)
            else:
                border_indices[i] = num_samples - 1

        samples_per_x_value = np.repeat(num_samples / num_x_values, num_x_values)

        return border_indices, samples_per_x_value, x_values, num_x_values


def demo_all_features(data_loader, with_original=False, epsilon=10, skip=None):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, pred_min, pred_max = data_loader.load_privacy_parameters()

    rng = np.random.default_rng(seed=2)

    x_train = data.drop(outcome, axis=1)

    for i in range(len(data.columns) - 1):
        ale = DPAccumulatedLocalEffects()
        ale.set_privacy_parameters(epsilon, data_ints[i], data_mins[i], data_maxs[i], pred_min, pred_max, rng)

        is_cat = data.columns[i] not in cont_features

        if skip is not None and ((skip == 'numeric' and not is_cat) or (skip == 'categorical' and is_cat)):
            continue

        ale = ale.fit(x_train.copy(), pred_func, i, is_cat, num_x_values=20, class_num=class_num, verbose=False)
        ale.draw_plot()

        if with_original:
            orig_ale = AccumulatedLocalEffects()
            orig_ale = orig_ale.fit(x_train.copy(), pred_func, i, is_cat, class_num=class_num, verbose=False)

            orig_ale.draw_plot()


def demo_one_feature(data_loader, feature, seed=0, epsilon=10.0, compare=None, filename=None, resolution=20):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, pred_min, pred_max = data_loader.load_privacy_parameters()

    if isinstance(feature, str):
        feature = data.columns.get_loc(feature)

    x_train = data.drop(outcome, axis=1)

    ale = DPAccumulatedLocalEffects()
    ale.set_privacy_parameters(epsilon, data_ints[feature], data_mins[feature], data_maxs[feature], pred_min, pred_max, seed)

    ale = ale.fit(x_train, pred_func, feature, data.columns[feature] not in cont_features, num_x_values=resolution, class_num=class_num)

    ale.draw_plot(save=filename is not None, filename=filename, comparison_plot=compare)


if __name__ == '__main__':
    logger = logging.getLogger('ale')
    logger.addHandler(logging.StreamHandler())

    #demo_all_features(BikeSharing(), with_original=False, epsilon=2)
    demo_one_feature(BikeSharing(), 'workingday', seed=None, epsilon=2)

    #epsilon_and_seed = [(1, 10549), (1, 94531), (1, 52931), (1, 13236), (1, 95721), (1, 89268), (1, 4051), (1, 14933),
    #                    (1, 43029), (1, 12544)]
    #
    #for epsilon, seed in epsilon_and_seed:
    #    demo_one_feature(AdultIncome(), 'capital-gain', seed=seed, epsilon=epsilon)


