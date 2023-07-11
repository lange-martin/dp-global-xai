import numpy as np
import pandas as pd
import similaritymeasures
from tqdm.contrib import itertools
from tqdm import tqdm
from datetime import datetime

from evaluation.pfi_evaluation import run_experiments_pfi
from explainers.plot_explainer import FeatureEffectExplainer
from explainers.plot_explainer import DPFeatureEffectExplainer
from explainers.partial_dependence_plots.pdp import PartialDependencePlot
from explainers.partial_dependence_plots.pdp_private import DPPartialDependencePlot
from explainers.accumulated_local_effects.ale import AccumulatedLocalEffects
from explainers.accumulated_local_effects.ale_private import DPAccumulatedLocalEffects
from explainers.generic_private_plot import DPGenericPlot
from data_loader import BikeSharing, AdultIncome, HeartDisease

from multiprocessing import Pool
import os

THREADS = 15


def get_interpolated_graph(x_data, cont_features, feature_index, pred_func,
                           explainer: FeatureEffectExplainer, class_num, x_min, x_max, num_x_values=None, filename=None):
    is_cat = x_data.columns[feature_index] not in cont_features

    if num_x_values is None:
        explainer = explainer.fit(x_data, pred_func, feature_index, is_cat, class_num=class_num)
    else:
        explainer = explainer.fit(x_data, pred_func, feature_index, is_cat, num_x_values=num_x_values,
                                  class_num=class_num)

    if filename is not None:
        directory = datetime.today().strftime("%Y-%m-%d")

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        # The final path to save to
        savepath = os.path.join(directory, filename)

        explainer.draw_plot(show=False, save=True, filename=savepath)

    if is_cat:
        x, y = explainer.get_x_y_values()
        x_y = np.empty((len(x), 2))
        x_y[:, 0] = x
        x_y[:, 1] = y
    else:
        x_y = np.empty((1000, 2))
        x_y[:, 0] = np.linspace(x_min, x_max, num=1000)
        x_y[:, 1] = explainer.interpolate(x_y[:, 0])

    return x_y


class ExperimentJob:
    def __init__(self, results_original_map, dataset_name, epsilon, repetition_index, feature_index, feature_name,
                 is_feature_int, feature_min, feature_max, min_pred, max_pred, x_data, cont_features, pred_func,
                 explainer_private, num_x_values, bounds_factor, intervals, class_num, seed):
        self.results_original_map = results_original_map
        self.dataset_name = dataset_name
        self.epsilon = epsilon
        self.repetition_index = repetition_index
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.is_feature_int = is_feature_int
        if bounds_factor is None:
            self.feature_min = feature_min
            self.feature_max = feature_max
            self.min_pred = min_pred
            self.max_pred = max_pred
        else:
            self.feature_min = ((1 - bounds_factor) * (feature_min + feature_max)) / 2
            self.feature_max = ((1 + bounds_factor) * (feature_min + feature_max)) / 2
            self.min_pred = ((1 - bounds_factor) * (min_pred + max_pred)) / 2
            self.max_pred = ((1 + bounds_factor) * (min_pred + max_pred)) / 2
        self.x_data = x_data
        self.cont_features = cont_features
        self.pred_func = pred_func
        self.explainer_private = explainer_private
        self.num_x_values = num_x_values
        self.bounds_factor = bounds_factor
        self.intervals = intervals
        self.class_num = class_num
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def evaluate_job(self):
        self.explainer_private.set_privacy_parameters(self.epsilon, self.is_feature_int, self.feature_min,
                                                      self.feature_max, self.min_pred, self.max_pred, self.rng)
        if self.intervals is not None:
            self.explainer_private.num_intervals = self.intervals

        filename = f'{self.explainer_private.explainer_name}-{self.dataset_name}-{self.feature_index}-{self.repetition_index}-res{self.num_x_values}-splits{self.intervals}-bounds{self.bounds_factor}.png'
        x_y_interpolated = get_interpolated_graph(self.x_data, self.cont_features, self.feature_index,
                                                  self.pred_func, self.explainer_private, self.class_num,
                                                  self.feature_min, self.feature_max, self.num_x_values,
                                                  filename=filename)
        x_y_original = self.results_original_map[self.dataset_name, self.feature_index]

        # calculate mean squared error
        df = similaritymeasures.mse(x_y_original[:, 1], x_y_interpolated[:, 1])

        if self.num_x_values is None:
            row_in_results = {'dataset': self.dataset_name, 'epsilon': self.epsilon, 'feature': self.feature_name,
                              'feature_index': self.feature_index, 'repetition': self.repetition_index, 'mse': df,
                              'seed': self.seed}
        else:
            row_in_results = {'dataset': self.dataset_name, 'epsilon': self.epsilon, 'feature': self.feature_name,
                              'feature_index': self.feature_index, 'repetition': self.repetition_index,
                              'resolution': self.num_x_values, 'mse': df, 'seed': self.seed}

        if self.bounds_factor is not None:
            row_in_results['bounds_factor'] = self.bounds_factor
        if self.intervals is not None:
            row_in_results['intervals'] = self.intervals

        return row_in_results


def call_job(job: ExperimentJob):
    return job.evaluate_job()


class Evaluator:
    def __init__(self, explainer_orig: FeatureEffectExplainer, explainer_priv: DPFeatureEffectExplainer, dataloaders,
                 epsilons, num_x_values, features=None, bounds_factors=None, intervals=None):
        self.explainer_orig = explainer_orig
        self.explainer_priv = explainer_priv
        self.dataloaders = dataloaders
        self.epsilons = epsilons
        self.num_x_values = num_x_values
        self.features = features
        if bounds_factors is None:
            self.bounds_factors = [None]
        else:
            self.bounds_factors = bounds_factors
        if intervals is None:
            self.intervals = [None]
        else:
            self.intervals = intervals

        self.rng = np.random.default_rng(seed=0)

    def evaluate(self, num_repetitions=1):
        # list for the results (frechet distance) for each dataset
        jobs = []

        # hashmaps for original results in order to retrieve them for computation of difference to private explanation
        results_orig_map = {}

        for d in range(len(self.dataloaders)):
            d_name = self.dataloaders[d].name
            pred_func, data, cont_features, outcome_name, class_num = self.dataloaders[d].load_data()
            data_mins, data_maxs, data_ints, min_pred, max_pred = self.dataloaders[d].load_privacy_parameters()

            # determine for which features the experiment is done
            if self.features is None:
                num_features = len(data.columns) - 1
                features = range(num_features)
            else:
                features = self.features[d]

            x_data = data.drop(outcome_name, axis=1)

            # only calculate deterministic original explanation once:
            for f in features:
                filename = f'{self.explainer_orig.explainer_name}-{d_name}-f{f}.png'
                x_y_interpolated = get_interpolated_graph(x_data, cont_features, f, pred_func, self.explainer_orig,
                                                          class_num, data_mins[f], data_maxs[f], filename=filename)
                results_orig_map[d_name, f] = x_y_interpolated

            for epsilon, r, f, xs, bf, i in itertools.product(self.epsilons, range(num_repetitions), features,
                                                              self.num_x_values, self.bounds_factors, self.intervals):
                job = ExperimentJob(results_orig_map, d_name, epsilon, r, f, data.columns[f], data_ints[f], data_mins[f],
                                    data_maxs[f], min_pred, max_pred, x_data, cont_features, pred_func,
                                    self.explainer_priv, xs, bf, i, class_num, self.rng.integers(0, 100000))

                jobs.append(job)

        with Pool(THREADS) as p:
            results = list(tqdm(p.imap(call_job, jobs), total=len(jobs)))

        return pd.DataFrame(results)


def run_experiments():
    epsilons = [0.5, 1, 2, 5, 10]
    repetitions = 10
    data_loaders = [AdultIncome(), BikeSharing(), HeartDisease()]
    resolutions = [20]

    eval = Evaluator(AccumulatedLocalEffects(), DPAccumulatedLocalEffects(), data_loaders,
                     epsilons=epsilons, num_x_values=resolutions)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_specific.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPGenericPlot(AccumulatedLocalEffects()),
                     data_loaders, epsilons=epsilons, num_x_values=resolutions)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_generic.csv', index=False)

    eval = Evaluator(PartialDependencePlot(), DPPartialDependencePlot(), data_loaders,
                     epsilons=epsilons, num_x_values=resolutions)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_specific.csv', index=False)

    eval = Evaluator(PartialDependencePlot(), DPGenericPlot(PartialDependencePlot()), data_loaders,
                     epsilons=epsilons, num_x_values=resolutions)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_generic.csv', index=False)


def run_experiments_resolution():
    epsilons = [0.5, 1, 2, 5, 10]
    num_x_values = [10, 20, 50, 100]
    repetitions = 10
    data_loaders = [AdultIncome(), BikeSharing(), HeartDisease()]
    features = [[0, 9], [2], [1]]

    eval = Evaluator(PartialDependencePlot(), DPPartialDependencePlot(), data_loaders,
                     epsilons=epsilons, num_x_values=num_x_values, features=features)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_specific_resolution.csv', index=False)

    eval = Evaluator(PartialDependencePlot(), DPGenericPlot(PartialDependencePlot()), data_loaders,
                     epsilons=epsilons, num_x_values=num_x_values, features=features)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_generic_resolution.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPAccumulatedLocalEffects(), data_loaders,
                     epsilons=epsilons, num_x_values=num_x_values, features=features)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_specific_resolution.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPGenericPlot(AccumulatedLocalEffects()), data_loaders,
                     epsilons=epsilons, num_x_values=num_x_values, features=features)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_generic_resolution.csv', index=False)


def run_experiments_bounds():
    epsilons = [0.5, 1, 2, 5, 10]
    repetitions = 10
    data_loaders = [AdultIncome(), BikeSharing(), HeartDisease()]
    features = [[0], [2, 3], [1]]
    bound_factors = [0.5, 0.75, 1, 1.25, 1.5]

    eval = Evaluator(PartialDependencePlot(), DPPartialDependencePlot(), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, bounds_factors=bound_factors)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_specific_bounds.csv', index=False)

    eval = Evaluator(PartialDependencePlot(), DPGenericPlot(PartialDependencePlot()), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, bounds_factors=bound_factors)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_generic_bounds.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPAccumulatedLocalEffects(), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, bounds_factors=bound_factors)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_specific_bounds.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPGenericPlot(AccumulatedLocalEffects()), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, bounds_factors=bound_factors)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_generic_bounds.csv', index=False)


def run_experiment_splits():
    epsilons = [0.5, 1, 2, 5, 10]
    repetitions = 10
    data_loaders = [AdultIncome(), BikeSharing(), HeartDisease()]
    features = [[0], [2], [1]]
    intervals = [100, 200, 300, 400, 500]

    eval = Evaluator(PartialDependencePlot(), DPGenericPlot(PartialDependencePlot()), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, intervals=intervals)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_pdp_generic_splits.csv', index=False)

    eval = Evaluator(AccumulatedLocalEffects(), DPGenericPlot(AccumulatedLocalEffects()), data_loaders,
                     epsilons=epsilons, num_x_values=[20], features=features, intervals=intervals)
    results = eval.evaluate(num_repetitions=repetitions)
    results.to_csv('results_ale_generic_splits.csv', index=False)


if __name__ == '__main__':
    #run_experiments()
    run_experiments_resolution()
    #run_experiment_splits()
    #run_experiments_bounds()
