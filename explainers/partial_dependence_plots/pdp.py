import numpy as np

from data_loader import HeartDisease, AdultIncome
from explainers.plot_explainer import FeatureEffectExplainer


class PartialDependencePlot(FeatureEffectExplainer):
    def __init__(self):
        super().__init__()
        self.explainer_name = 'PDP'

    # class_num=None indicates a regression problem, otherwise a classification problem
    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=None, class_num=None, verbose=False):
        if not type(x_train).__module__ == np.__name__:
            x_train_np = x_train.to_numpy()
            self.feature_name = x_train.columns[feature_index]
            self.feature_values = x_train[self.feature_name].to_numpy()
        else:
            x_train_np = x_train
            self.feature_name = "?"
            self.feature_values = x_train[:, feature_index]

        self.is_categorical = is_cat

        # find all values of the given feature (np.unique returns a sorted array)
        all_feature_values = np.unique(self.feature_values)
        num_unique = len(all_feature_values)

        num_samples = len(x_train_np)
        num_features = len(x_train_np[0])

        # create array in which the values of the respective feature are replaced with one common value (each x value
        # in the plot)
        changed_data = np.tile(x_train_np, (num_unique, 1, 1))
        changed_data[:, :, feature_index] = np.tile(all_feature_values, num_samples).reshape(
            (num_unique, num_samples), order='F')

        # get predictions by the model for all these different samples (flatten them first)
        predictions_flat = pred_func(np.reshape(changed_data, (num_unique * num_samples, num_features)))
        # in case of classification, we need to pick the predicted values for the desired class
        if class_num is not None:
            predictions_flat = predictions_flat[:, class_num]
        # reshape the predictions so that predictions for a single equal feature value are found in one row
        predictions = np.reshape(predictions_flat, (num_unique, num_samples))

        # average over all predictions for each feature value. these are the y values in the plot
        all_prediction_avgs = np.mean(predictions, axis=1)

        # save list of values and list of average predictions for those values
        self.x_values = all_feature_values
        self.y_values = all_prediction_avgs

        return self


def demo_all_features(data_loader):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()

    for i in range(len(data.columns) - 1):
       is_categorical = data.columns[i] not in cont_features

       pdp = PartialDependencePlot().fit(data.drop(outcome, axis=1), pred_func, i, is_categorical, class_num=class_num)
       pdp.draw_plot()


def demo_one_feature(data_loader, feature, filename=None, show=True):
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()

    if isinstance(feature, str):
        feature = data.columns.get_loc(feature)

    pdp = PartialDependencePlot().fit(data.drop(outcome, axis=1), pred_func, feature,
                                      data.columns[feature] not in cont_features, num_x_values=None, class_num=class_num)
    pdp.draw_plot(show=show, save=filename is not None, filename=filename)

    return pdp.get_x_y_values()


if __name__ == '__main__':
    demo_all_features(HeartDisease())
    #demo_one_feature(AdultIncome(), 'age', filename='pdp-age.pdf')
