import numpy as np
from explainers.plot_explainer import DPFeatureEffectExplainer
from explainers.partial_dependence_plots.pdp import PartialDependencePlot
from explainers.accumulated_local_effects.ale import AccumulatedLocalEffects

from data_loader import AdultIncome, BikeSharing, HeartDisease
from util import inverse_laplace


# Generic DP Plots
class DPGenericPlot(DPFeatureEffectExplainer):
    def __init__(self, explainer=None):
        super().__init__()
        self.explainer_name = f'Generic {explainer.explainer_name} (ε=1)'
        self.orig_explainer = explainer
        self.epsilon_histogram = 1/2
        self.epsilon_pd = 1/2
        self.num_intervals = 200

    def set_privacy_parameters(self, epsilon, is_int, x_min, x_max, y_min, y_max, seed):
        self.explainer_name = f'Generic {self.orig_explainer.explainer_name} (ε={epsilon})'
        self.epsilon_histogram = epsilon / 9
        self.epsilon_pd = (8 * epsilon) / 9
        self.is_int = is_int
        self.x_min = x_min
        self.x_max = x_max

        if isinstance(self.orig_explainer, AccumulatedLocalEffects):
            self.y_max = max(np.abs(y_min), np.abs(y_max))
            self.y_min = -self.y_max
        else:
            self.y_min = y_min
            self.y_max = y_max

        self.rng = np.random.default_rng(seed=seed)

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=100, class_num=None, verbose=False):
        self.feature_name = x_train.columns[feature_index]
        self.is_categorical = is_cat
        x_train = x_train.to_numpy()
        self.feature_values = x_train[:, feature_index]
        self.rng.shuffle(x_train, axis=0)

        num_samples = x_train.shape[0]
        samples_per_interval = num_samples / self.num_intervals

        if is_cat:
            # we assume categories are known and not a specific property of this dataset
            categories = np.unique(x_train[:, feature_index])
            num_x_values = len(categories)
        else:
            categories = None

        all_x_values = []
        all_y_values = []

        for i in range(self.num_intervals):
            start_index = min(num_samples - 1, int(i * samples_per_interval))
            end_index = min(num_samples, int((i+1) * samples_per_interval))

            x_values, y_values = self._fit_single(x_train[start_index:end_index], pred_func, feature_index,
                                                  num_x_values, class_num)
            all_x_values.append(x_values)
            all_y_values.append(y_values)

        if is_cat:
            interpolated_x = categories
        else:
            interpolated_x = np.linspace(self.x_min, self.x_max, num=num_x_values)

        interpolated_y_all = np.empty((self.num_intervals, num_x_values))

        for i in range(self.num_intervals):
            interpolated_y_all[i] = np.interp(interpolated_x, all_x_values[i], all_y_values[i])

        np.clip(interpolated_y_all, self.y_min, self.y_max, interpolated_y_all)

        denominator = (self.num_intervals * self.epsilon_pd)

        sensitivity_y = num_x_values * (self.y_max - self.y_min)

        self.error = inverse_laplace(0.975, 0, sensitivity_y / denominator)

        x_values = interpolated_x
        noise = self.rng.laplace(scale=sensitivity_y/denominator, size=num_x_values)
        y_values = np.mean(interpolated_y_all, axis=0) + noise

        # calculate values for feature distribution histogram
        self._calc_histogram(num_x_values)

        self.x_values = x_values
        self.y_values = y_values

        return self

    def _fit_single(self, x_train, pred_func, feature_index, num_x_values, class_num=None):
        exp = self.orig_explainer
        exp = exp.fit(x_train, pred_func, feature_index, self.is_categorical, num_x_values, class_num=class_num)

        x_values, y_values = exp.get_x_y_values()

        # save list of values and list of average predictions for those values
        return x_values, y_values


def demo_all_features(data_loader, explainer, with_original=False, skip=None):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, pred_min, pred_max = data_loader.load_privacy_parameters()

    rng = np.random.default_rng(seed=0)

    for i in range(len(data.columns) - 1):
        is_cat = data.columns[i] not in cont_features

        if skip is not None and ((skip == 'numeric' and not is_cat) or (skip == 'categorical' and is_cat)):
            continue

        exp = DPGenericPlot(explainer)
        exp.set_privacy_parameters(10, data_ints[i], data_mins[i], data_maxs[i], pred_min, pred_max, rng)
        exp.fit(data.drop(outcome, axis=1), pred_func, i, is_cat, num_x_values=20, class_num=class_num)
        exp.draw_plot()

        if with_original:
            orig = explainer.fit(data.drop(outcome, axis=1), pred_func, i, is_cat, class_num=class_num)
            orig.draw_plot()


def demo_one_feature(data_loader, explainer, feature, seed=1214, epsilon=0.5, num_splits=200, compare=None,
                     filename=None, resolution=20):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, pred_min, pred_max = data_loader.load_privacy_parameters()

    if isinstance(feature, str):
        feature = data.columns.get_loc(feature)

    is_categorical = data.columns[feature] not in cont_features

    exp = DPGenericPlot(explainer)
    exp.num_intervals = num_splits
    exp.set_privacy_parameters(epsilon, data_ints[feature], data_mins[feature], data_maxs[feature], pred_min, pred_max, seed)
    exp.fit(data.drop(outcome, axis=1), pred_func, feature, is_categorical, num_x_values=resolution, class_num=class_num)
    exp.draw_plot(save=filename is not None, filename=filename, comparison_plot=compare)


if __name__ == '__main__':
    exp = AccumulatedLocalEffects()
    #demo_all_features(AdultIncome(), exp, with_original=False, skip=None)
    demo_one_feature(AdultIncome(), AccumulatedLocalEffects(), 'age', seed=36230, epsilon=10)

    #epsilon_and_seed = [(0.5, 28182), (1, 49419), (2, 76217), (5, 85843), (10, 41125)] # HeartDisease Age
    #epsilon_and_seed = [(0.5, 85062), (1, 90207), (2, 80197), (5, 3583), (10, 96231)] # AdultIncome age
    #epsilon_and_seed = [(0.5, 58546), (1, 30377), (2, 2739), (5, 70975), (10, 32622)]  # BikeSharing hr
    #epsilon_and_seed = [(5, 58987), (5, 88437), (5, 6289), (5, 13503), (5, 63406), (5, 74442), (5, 48884),
    #                    (5, 43999), (5, 32554), (5, 81802)]  # AdultIncome capital-gain all 10
    #
    #for epsilon, seed in epsilon_and_seed:
    #    demo_one_feature(AdultIncome(), PartialDependencePlot(), 'capital-gain', seed=seed, epsilon=epsilon)
