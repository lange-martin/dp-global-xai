import numpy as np
import pandas as pd
from tqdm.contrib import itertools
from tqdm import tqdm

from explainers.permutation_feature_importance.pfi_explainer import PFIExplainer
from explainers.permutation_feature_importance.pfi_explainer import PrivatePFIExplainer
from explainers.permutation_feature_importance.pfi_explainer import DPFeatureImportance
from explainers.permutation_feature_importance.pfi_explainer import DPGenericRankAggregation

from data_loader import BikeSharing, AdultIncome, HeartDisease

from multiprocessing import Pool

THREADS = 15


# slightly modified version from wikipedia: https://en.wikipedia.org/w/index.php?title=Kendall_tau_distance&oldid=1091992281
def kendall_tau_metric(ranking_original, ranking_private):
    n = len(ranking_original)
    assert len(ranking_private) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(ranking_original)
    b = np.argsort(ranking_private)
    nordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] < b[j]),
                             np.logical_and(a[i] > a[j], b[i] > b[j])).sum()
    return nordered / (n * (n - 1))


class ExperimentJob:
    def __init__(self, epsilon, dataset_index, repetition_index, x_train, y_train, min_pred, max_pred, pred_func,
                 class_num, non_private_pfi, private_pfi, seed):
        self.epsilon = epsilon
        self.dataset_index = dataset_index
        self.repetition_index = repetition_index
        self.x_train = x_train
        self.y_train = y_train
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.pred_func = pred_func
        self.class_num = class_num
        self.non_private_pfi = non_private_pfi
        self.private_pfi = private_pfi
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def evaluate_job(self):
        self.private_pfi.setup(self.epsilon, self.min_pred, self.max_pred, self.rng)

        non_private_ranking = self.non_private_pfi.ranking(self.x_train, self.y_train, self.pred_func, self.class_num)
        private_ranking = self.private_pfi.ranking(self.x_train, self.y_train, self.pred_func, self.class_num)

        similarity = kendall_tau_metric(non_private_ranking, private_ranking)

        row_in_results = {'dataset': self.dataset_index, 'epsilon': self.epsilon, 'repetition': self.repetition_index,
                          'non_private_ranking': non_private_ranking, 'private_ranking': private_ranking,
                          'similarity': similarity, 'seed': self.seed}
        return row_in_results


def call_job(job: ExperimentJob):
    return job.evaluate_job()


class Evaluator:
    def __init__(self, non_private_pfi: PFIExplainer, private_pfi: PrivatePFIExplainer, dataloaders, epsilons):
        self.non_private_pfi = non_private_pfi
        self.private_pfi = private_pfi
        self.dataloaders = dataloaders
        self.epsilons = epsilons

        self.rng = np.random.default_rng(seed=0)

    def evaluate(self, num_repetitions=1):
        jobs = []

        for d in range(len(self.dataloaders)):
            pred_func, data, cont_features, outcome_name, class_num = self.dataloaders[d].load_data()
            data_mins, data_maxs, data_ints, min_pred, max_pred = self.dataloaders[d].load_privacy_parameters()

            x_train = data.drop(outcome_name, axis=1).to_numpy()
            y_train = data[outcome_name].to_numpy()

            for epsilon, r in itertools.product(self.epsilons, range(num_repetitions)):
                job = ExperimentJob(epsilon, d, r, x_train, y_train, min_pred, max_pred, pred_func, class_num,
                                     self.non_private_pfi, self.private_pfi, self.rng.integers(0, 100000))
                jobs.append(job)

        with Pool(THREADS) as p:
            results = list(tqdm(p.imap(call_job, jobs), total=len(jobs)))

        return pd.DataFrame(results)


def run_experiments_pfi():
    epsilons = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    repetitions = 10
    data_loaders = [AdultIncome(), BikeSharing(), HeartDisease()]

    rng = np.random.default_rng(seed=0)

    eval = Evaluator(PFIExplainer(rng), DPGenericRankAggregation(rng), data_loaders,
                     epsilons=epsilons)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pfi_generic.csv', index=False)

    eval = Evaluator(PFIExplainer(rng), DPFeatureImportance(rng), data_loaders,
                     epsilons=epsilons)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pfi_specific.csv', index=False)


if __name__ == '__main__':
    run_experiments_pfi()