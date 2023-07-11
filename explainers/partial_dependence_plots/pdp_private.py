import numpy as np

from data_loader import AdultIncome, BikeSharing, HeartDisease
from explainers.partial_dependence_plots.pdp import PartialDependencePlot
from explainers.plot_explainer import DPFeatureEffectExplainer
from util import inverse_laplace


# DP PDP
class DPPartialDependencePlot(DPFeatureEffectExplainer):
    def __init__(self):
        super().__init__()
        self.explainer_name = "DP PDP (ε=1)"

        self.epsilon_histogram = 1/3
        self.epsilon_pd = 1/3
        self.epsilon_x_values = 1 / 3

    def set_privacy_parameters(self, epsilon, is_int, x_min, x_max, y_min, y_max, seed):
        # split the privacy budget
        self.epsilon_pd = 8 * epsilon / 9
        self.epsilon_histogram = epsilon / 9

        self.explainer_name = f"DP PDP (ε={epsilon})"

        self.is_int = is_int

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.rng = np.random.default_rng(seed=seed)

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=20, class_num=None, verbose=False):

        x_train_np = x_train.to_numpy()

        self.is_categorical = is_cat
        self.feature_name = x_train.columns[feature_index]
        self.feature_values = x_train_np[:, feature_index]

        if is_cat:
            # we assume categories are known and not a specific property of this dataset
            x_values = np.unique(x_train_np[:, feature_index])
        else:
            # get equidistant feature values for numeric features. these will be used as x values for the plot
            x_values = np.linspace(self.x_min, self.x_max, num=num_x_values)

        x_values = np.unique(x_values)
        num_x_values = len(x_values)

        # array with average predictions values for x values.
        partial_dependence = np.zeros(num_x_values)

        # iterate over each x value and calculate its partial dependence
        for i, feature_value in enumerate(x_values):
            changed_data = x_train_np.copy()
            changed_data[:, feature_index] = feature_value

            if class_num is None:
                predictions = pred_func(changed_data)
            else:
                predictions = pred_func(changed_data)[:, class_num]

            # clip predictions according to given maximum and minimum predictions
            np.clip(predictions, self.y_min, self.y_max, out=predictions)

            # calculate the average prediction for this value
            partial_dependence[i] = np.mean(predictions)

        # add noise to ensure differential privacy
        pred_range = self.y_max - self.y_min
        scale = (num_x_values / x_train_np.shape[0]) * pred_range * (1 / self.epsilon_pd)
        partial_dependence = partial_dependence + self.rng.laplace(scale=scale, size=num_x_values)
        self.error = inverse_laplace(0.975, 0, scale)

        # calculate values for feature distribution histogram
        self._calc_histogram(num_x_values)

        # save list of values and list of average predictions for those values
        self.x_values = x_values
        self.y_values = partial_dependence

        return self


def demo_all_features(data_loader, with_original=False, epsilon=10, skip=None):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, min_pred, max_pred = data_loader.load_privacy_parameters()

    x_train = data.drop(outcome, axis=1)

    for i in range(len(data.columns) - 1):
        pdp = DPPartialDependencePlot()
        pdp.set_privacy_parameters(epsilon, data_ints[i], data_mins[i], data_maxs[i], min_pred, max_pred, 0)

        is_cat = data.columns[i] not in cont_features

        if skip is not None and ((skip == 'numeric' and not is_cat) or (skip == 'categorical' and is_cat)):
            continue

        pdp = pdp.fit(x_train, pred_func, i, is_cat, 20, class_num=class_num, verbose=False)

        if with_original:
            orig_pdp = PartialDependencePlot()
            orig_pdp = orig_pdp.fit(x_train, pred_func, i, is_cat, class_num=class_num, verbose=False)

            orig_pdp.draw_plot(save=False)
        pdp.draw_plot(save=False)


def demo_one_feature(data_loader, feature, seed=0, epsilon=0.5, compare=None, filename=None, show=True, resolution=20):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, min_pred, max_pred = data_loader.load_privacy_parameters()

    x_train = data.drop(outcome, axis=1)

    if isinstance(feature, str):
        feature = data.columns.get_loc(feature)

    pdp = DPPartialDependencePlot()

    pdp.set_privacy_parameters(epsilon, data_ints[feature], data_mins[feature], data_maxs[feature], min_pred, max_pred, seed)

    pdp = pdp.fit(x_train, pred_func, feature, data.columns[feature] not in cont_features, resolution, class_num=class_num)

    pdp.draw_plot(show=show, save=filename is not None, filename=filename, comparison_plot=compare)


if __name__ == '__main__':
    demo_all_features(AdultIncome(), with_original=True, epsilon=5, skip='categorical')
    #demo_one_feature(HeartDisease(), 'age', seed=28183)

    #epsilon_and_seed = [(0.5, 85062), (1, 90207), (2, 80197), (5, 3583), (10, 96231)] # AdultIncome age
    #epsilon_and_seed = [(0.5, 58546), (1, 30377), (2, 2739), (5, 70975), (10, 32622)]  # BikeSharing hr#
    #epsilon_and_seed = [(10, 32622), (10, 91117), (10, 7381), (10, 21318), (10, 11936), (10, 55389), (10, 30326),
    #                    (10, 62851), (10, 88667), (10, 45122)]  # BikeSharing hr all 10
    #epsilon_and_seed = [(5, 58987), (5, 88437), (5, 6289), (5, 13503), (5, 63406), (5, 74442), (5, 48884),
    #                    (5, 43999), (5, 32554), (5, 81802)]  # AdultIncome capital-gain all 10
    #
    #for epsilon, seed in epsilon_and_seed:
    #    demo_one_feature(AdultIncome(), 'capital-gain', seed=seed, epsilon=epsilon)
    #