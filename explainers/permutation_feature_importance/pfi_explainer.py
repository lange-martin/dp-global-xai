import numpy as np
from itertools import permutations

from data_loader import HeartDisease, BikeSharing, AdultIncome


# PFI explainer without privacy
class PFIExplainer:
    def __init__(self, seed):
        # random number generator used for every non-deterministic action (here: permuting the feature values)
        self.rng = np.random.default_rng(seed=seed)

    # returns the feature importance score for feature "feature_index" (higher means more important)
    def calc(self, x_train, y_train, pred_func, feature_index, class_num=None):
        x_train_permuted = x_train.copy()
        self.rng.shuffle(x_train_permuted[:, feature_index])

        # calculate error for altered dataset. The feature in question is permuted between the samples.
        error_permuted = self._loss(x_train_permuted, y_train, pred_func, class_num)

        # error is the feature importance score
        return error_permuted

    # returns a list of all features ordered by the importance. Lower index => higher importance
    def ranking(self, x_train, y_train, pred_func, class_num=None):
        num_features = x_train.shape[1]
        feature_importances = np.zeros(num_features)

        for i in range(num_features):
            feature_importances[i] = self.calc(x_train, y_train, pred_func, i, class_num)

        # flip the result because argsort orders by ascending value
        return np.flip(np.argsort(feature_importances))

    # returns error of predictions for x_train relative to y_train (the labels)
    def _loss(self, x_train, y_train, pred_func, class_num=None):
        if class_num is not None:
            predictions = pred_func(x_train)[:, class_num]
        else:
            predictions = pred_func(x_train)

        # mean squared error
        return np.mean(np.square(predictions - y_train))


class PrivatePFIExplainer(PFIExplainer):
    def __init__(self, seed):
        super().__init__(seed)
        self.epsilon = 1
        self.min_pred = 0
        self.max_pred = 1

    # sets the privacy parameters of the explainer (epsilon value and range for predictions)
    def setup(self, epsilon, min_pred, max_pred, seed=None):
        self.epsilon = epsilon
        self.min_pred = min_pred
        self.max_pred = max_pred
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)


# DP PFI
# feature importance explainer with privacy (add laplacian noise to each score. split epsilon between features)
class DPFeatureImportance(PrivatePFIExplainer):
    def calc(self, x_train, y_train, pred_func, feature_index, class_num=None):

        x_train_permuted = x_train.copy()
        self.rng.shuffle(x_train_permuted[:, feature_index])

        # calculate error for altered dataset. The feature in question is permuted between the samples.
        error_permuted = self._loss(x_train_permuted, y_train, pred_func, class_num)

        # error is the feature importance score (omit original error, does not change ranking)
        importance = error_permuted

        # add laplacian noise. the sensitivity of the function is (2 * range^2) / n.
        noise_scale = (2 * (self.max_pred - self.min_pred)**2) / (len(x_train) * self.epsilon)
        return importance + self.rng.laplace(scale=noise_scale)

    def ranking(self, x_train, y_train, pred_func, class_num=None):
        orig_epsilon = self.epsilon

        # when calculating the ranking, the privacy budget must be split between all features
        self.epsilon = self.epsilon / x_train.shape[1]

        # then, the ranking can be calculated normally with superclass (using the private calc-method of this subclass)
        ranking = PrivatePFIExplainer.ranking(self, x_train, y_train, pred_func, class_num)

        # return epsilon to its original value
        self.epsilon = orig_epsilon

        return ranking

    def _loss(self, x_train, y_train, pred_func, class_num=None):
        if class_num is not None:
            predictions = pred_func(x_train)[:, class_num]
        else:
            predictions = pred_func(x_train)

        # clip the predictions so that the sensitivity analysis is correct, even when outliers exist
        np.clip(predictions, self.min_pred, self.max_pred, predictions)

        return np.mean(np.square(predictions - y_train))


# DP Rank Aggregation
# applied to PFI (split dataset into many subsets, calculate feature importance ranking)
# for each one without noise and let them vote on final ranking. Add laplacian noise to votes.
class DPGenericRankAggregation(PrivatePFIExplainer):
    def ranking(self, x_train, y_train, pred_func, class_num=None):
        num_samples = x_train.shape[0]
        num_features = x_train.shape[1]

        # this hyperparameter determines how many subsets/intervals the dataset will be split into.
        # More samples per interval -> better estimation of feature ranking per interval, but more affected by noise
        # Fewer samples per interval -> worse estimation of feature ranking per interval, but less affected by noise
        num_intervals = 200
        samples_per_interval = num_samples / num_intervals

        # holds feature ranking for each interval
        full_rankings = []

        for i in range(num_intervals):
            start_index = min(num_samples - 1, int(i * samples_per_interval))
            end_index = min(num_samples, int((i+1) * samples_per_interval))

            pfi = PFIExplainer(self.rng)
            ranking = pfi.ranking(x_train[start_index:end_index], y_train[start_index:end_index], pred_func, class_num=class_num)

            full_rankings.append(ranking)

        # aggregate the result by holding a vote with the positional scoring rule
        return self.aggregate_psr(num_features, full_rankings)

    # returns ranking determined by vote with positional scoring rule
    def aggregate_psr(self, num_features, full_rankings):
        full_rankings = np.array(full_rankings)
        votes = np.zeros(num_features)

        # the first "num_ranks" preferences of each ballot (member of full_rankings) receive points. The first
        # preference receives num_ranks points, the second num_ranks - 1 and so on. The lower half of the ballot
        # does not receive any points. Each ballot is a ranking of all features.
        num_ranks = num_features #int(num_features / 2)

        for i in range(num_ranks):
            score = num_ranks - i

            for j in range(num_features):
                votes[j] += score * np.count_nonzero(full_rankings[:, i] == j)

        # calculate sensitivity and add laplacian noise to the votes
        sensitivity = np.ceil(num_ranks**2) + num_ranks
        votes = votes + self.rng.laplace(scale=sensitivity / self.epsilon, size=num_features)

        # determine final ranking by sorting according to the votes. Again, flip the result because argsort sorts
        # with ascending values
        return np.flip(np.argsort(votes))


class DPFeatureImportanceDummy(PrivatePFIExplainer):
    pass


if __name__ == '__main__':
    data_loader = AdultIncome()
    pred_func, data, cont_features, outcome, class_num = data_loader.load_data()
    data_mins, data_maxs, data_ints, min_pred, max_pred = data_loader.load_privacy_parameters()

    x_train = data.drop(outcome, axis=1).to_numpy()
    y_train = data[outcome].to_numpy()

    orig_pfi = PFIExplainer(seed=0)
    ranking_orig = orig_pfi.ranking(x_train, y_train, pred_func, class_num)

    print(ranking_orig)

    pfi = DPGenericRankAggregation(seed=0)
    pfi.setup(0.2, min_pred, max_pred)
    ranking = pfi.ranking(x_train, y_train, pred_func, class_num)

    print(ranking)

