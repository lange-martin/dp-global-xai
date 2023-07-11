import numpy as np
import logging

from data_loader import HeartDisease, BikeSharing, AdultIncome
from explainers.plot_explainer import FeatureEffectExplainer


class AccumulatedLocalEffects(FeatureEffectExplainer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('ale')
        self.explainer_name = 'ALE'

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=100, class_num=None, verbose=False):
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if not type(x_train).__module__ == np.__name__:
            x_train_np = x_train.to_numpy()
            self.feature_name = x_train.columns[feature_index]
        else:
            x_train_np = x_train
            self.feature_name = "?"

        num_samples = len(x_train_np)

        self.is_categorical = is_cat
        self.feature_values = x_train_np[:, feature_index]

        # print effect of each category for categorical feature (disregard setting of num_x_values)
        if is_cat:
            num_x_values = np.unique(self.feature_values).shape[0]

        # sort samples by respective feature to determine quantiles
        x_train_np = x_train_np[np.argsort(x_train_np[:, feature_index])]

        # num_x_values - 1 intervals have num_x_values borders, resulting in a resolution of num_x_values
        if num_x_values > 1:
            samples_per_interval = num_samples / (num_x_values - 1)
        else:
            samples_per_interval = 0

        # array for effects of feature values (y-axis in plot)
        effects = np.zeros(shape=num_x_values)
        # array for feature values (x-axis in plot)
        feature_values = np.zeros(shape=num_x_values)

        if not is_cat:
            upper_indices = [min(num_samples - 1, int((i + 1) * samples_per_interval)) for i in range(effects.shape[0] - 1)]
            border_indices = np.zeros(num_x_values)
            border_indices[1:] = np.array(upper_indices)
        else:
            # create one interval per category for a categorical feature
            feature_col = x_train_np[:, feature_index]
            border_indices = np.zeros(shape=num_x_values)
            border_indices[1:] = np.where(feature_col[:-1] != feature_col[1:])[0] + 1

        border_indices = border_indices.astype(int)
        feature_values[0] = x_train_np[0, feature_index]

        # Calling the prediction function has the highest overhead. Therefore we first collect all generated samples
        # for which we will need a prediction and then only call the prediction function once.
        all_samples = np.zeros((2 * num_samples, x_train_np.shape[1]))
        running_index = 0

        for cur_effect in range(1, effects.shape[0]):
            index_start = border_indices[cur_effect - 1]
            index_end = border_indices[cur_effect]

            self.logger.debug(f'Interval {cur_effect} start index: {index_start}')
            self.logger.debug(f'Interval {cur_effect} end index: {index_end}')

            if index_start != index_end:
                # determine the lower and upper boundary of the feature for this interval in the plot. These lie at the
                # quantiles.
                lower_border = x_train_np[index_start, feature_index]
                upper_border = x_train_np[index_end, feature_index]

                self.logger.debug(f'Interval {cur_effect} lower border: {lower_border}')
                self.logger.debug(f'Interval {cur_effect} upper border: {upper_border}')

                # add upper_border as the feature value where the effect will be shown in the plot
                feature_values[cur_effect] = upper_border

                # copy all samples of this quantile twice to alter them with the lower and upper border
                x_train_low = x_train_np[index_start:index_end].copy()
                x_train_high = x_train_np[index_start:index_end].copy()
                x_train_low[:, feature_index] = lower_border
                x_train_high[:, feature_index] = upper_border

                all_samples[running_index:running_index + index_end - index_start] = x_train_low
                running_index += index_end - index_start
                all_samples[running_index:running_index + index_end - index_start] = x_train_high
                running_index += index_end - index_start
            else:
                # if the start index and end index are the same, nothing can be compared
                # this occurs for the first category of a categorical feature
                feature_values[cur_effect] = x_train_np[index_start, feature_index]

        if class_num is None:
            predictions = pred_func(all_samples)
        else:
            predictions = pred_func(all_samples)[:, class_num]

        running_index = 0

        for cur_effect in range(1, effects.shape[0]):
            index_start = border_indices[cur_effect - 1]
            index_end = border_indices[cur_effect]

            if index_start != index_end:
                low_predictions = predictions[running_index:running_index + index_end - index_start]
                running_index += index_end - index_start
                high_predictions = predictions[running_index:running_index + index_end - index_start]
                running_index += index_end - index_start

                # determine difference in prediction for each altered sample in the quantile
                pred_diffs = high_predictions - low_predictions
            else:
                pred_diffs = np.array([0])

            # calculate effect by adding previous effect and average prediction difference
            if cur_effect > 0:
                effects[cur_effect] = effects[cur_effect - 1]
            effects[cur_effect] = effects[cur_effect] + np.mean(pred_diffs)

            self.logger.debug(f'Interval {cur_effect} num samples: {index_end - index_start}')
            self.logger.debug(f'Interval {cur_effect} effect: {effects[cur_effect]}')

        # center the ALE
        effects = effects - np.mean(effects)

        self.x_values = feature_values
        self.y_values = effects

        return self


def demo_all_features(data_loader, skip=None):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()

    for i in range(len(data.columns) - 1):
        is_cat = data.columns[i] not in cont_features

        if skip is not None and ((skip == 'numeric' and not is_cat) or (skip == 'categorical' and is_cat)):
            continue

        ale = AccumulatedLocalEffects().fit(data.drop(outcome, axis=1), pred_func, i, is_cat, 100, class_num=class_num,
                                            verbose=False)
        ale.draw_plot()


def demo_one_feature(data_loader, feature, filename=None, show=True):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()

    if isinstance(feature, str):
        feature = data.columns.get_loc(feature)

    ale = AccumulatedLocalEffects().fit(data.drop(outcome, axis=1), pred_func, feature,
                                        data.columns[feature] not in cont_features, 100, class_num=class_num)
    ale.draw_plot(show=show, save=filename is not None, filename=filename)

    return ale.get_x_y_values()


if __name__ == '__main__':
    logger = logging.getLogger('ale')
    logger.addHandler(logging.StreamHandler())

    demo_all_features(BikeSharing(), skip=None)
    #demo_one_feature(BikeSharing(), 'atemp')
