import numpy as np
import pandas as pd
import itertools

dataset_index = 0
epsilon_index = 1
feature_index = 2
resolution_index = 5
mse_index = 6
splits_index = 8


def set_indices(df):
    global dataset_index
    global epsilon_index
    global feature_index
    global resolution_index
    global mse_index
    global splits_index
    dataset_index = df.columns.get_loc("dataset")
    epsilon_index = df.columns.get_loc("epsilon")
    feature_index = df.columns.get_loc("feature")
    resolution_index = df.columns.get_loc("resolution")
    mse_index = df.columns.get_loc("mse")

    if "intervals" in df.columns:
        splits_index = df.columns.get_loc("intervals")


def calc_mean_single(array, dataset, epsilon, feature, resolution=None, splits=None):
    if feature is not None:
        if resolution is not None:
            indices = np.where((array[:, dataset_index] == dataset) & (array[:, epsilon_index] == epsilon) &
                               (array[:, resolution_index] == resolution) & (array[:, feature_index] == feature))
        elif splits is not None:
            indices = np.where((array[:, dataset_index] == dataset) & (array[:, epsilon_index] == epsilon) &
                               (array[:, splits_index] == splits) & (array[:, feature_index] == feature))
        else:
            indices = np.where((array[:, dataset_index] == dataset) & (array[:, epsilon_index] == epsilon) &
                               (array[:, feature_index] == feature))
    elif resolution is not None:
        indices = np.where((array[:, dataset_index] == dataset) & (array[:, epsilon_index] == epsilon) &
                           (array[:, resolution_index] == resolution))
    else:
        indices = np.where((array[:, dataset_index] == dataset) & (array[:, epsilon_index] == epsilon))

    if len(indices[0]) == 0:
        return 0, 0
    scores = array[indices, mse_index]
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score


def calc_mean(array1, array2, name1, name2, metric, split_features=False, split_resolution=False, split_splits=False):
    datasets = np.unique(array1[:, dataset_index])
    epsilons = np.unique(array1[:, epsilon_index])

    features_per_dataset = {}
    if split_features:
        for d in datasets:
            indices = np.where(array1[:, dataset_index] == d)
            features = np.unique(array1[indices, feature_index])
            features_per_dataset[d] = features
    else:
        for d in datasets:
            features_per_dataset[d] = [None]

    resolutions = [None]
    if split_resolution:
        resolutions = np.unique(array1[:, resolution_index])

    splits = [None]
    if split_splits:
        splits = np.unique(array1[:, splits_index])

    mean_results = []

    for d, e, res, splits in itertools.product(datasets, epsilons, resolutions, splits):
        for f in features_per_dataset[d]:
            mean_score1, std1 = calc_mean_single(array1, d, e, f, res, splits)
            mean_score2, std2 = calc_mean_single(array2, d, e, f, res, splits)

            row_in_results = {'dataset': d, 'epsilon': e, 'feature': f, 'resolution': res, f'mean {metric} {name1}': mean_score1,
                              f'std {name1}': std1, f'mean {metric} {name2}': mean_score2, f'std {name2}': std2, 'splits': splits}
            mean_results.append(row_in_results)

    return pd.DataFrame(mean_results)


if __name__ == '__main__':
    #ale_generic = pd.read_csv('results_ale_generic.csv')
    #ale_specific = pd.read_csv('results_ale_specific.csv')
    #pdp_generic = pd.read_csv('results_pdp_generic.csv')
    #pdp_specific = pd.read_csv('results_pdp_specific.csv')

    pdp_specific_res = pd.read_csv('results_pdp_specific_resolution.csv')
    ale_specific_res = pd.read_csv('results_ale_specific_resolution.csv')
    pdp_generic_res = pd.read_csv('results_pdp_generic_resolution.csv')
    ale_generic_res = pd.read_csv('results_ale_generic_resolution.csv')

    #pdp_generic_splits = pd.read_csv('results_pdp_generic_splits.csv').to_numpy()
    #ale_generic_splits = pd.read_csv('results_ale_generic_splits.csv').to_numpy()

    #pfi_specific = pd.read_csv('results_pfi_specific.csv')
    #pfi_generic = pd.read_csv('results_pfi_generic.csv')

    set_indices(ale_generic_res)

    #results_ale = calc_mean(ale_generic.to_numpy(), ale_specific.to_numpy(), 'ALE generic', 'ALE specific', 'MSE', True)
    #results_pdp = calc_mean(pdp_generic.to_numpy(), pdp_specific.to_numpy(), 'PDP generic', 'PDP specific', 'MSE', True)
    #results_pfi = calc_mean(pfi_specific.to_numpy(), pfi_generic.to_numpy(), 'PFI specific', 'PFI generic', 'KTM')
    results_pdp_res = calc_mean(pdp_generic_res.to_numpy(), pdp_specific_res.to_numpy(), 'PDP generic', 'PDP specific', 'MSE', True, split_resolution=True)
    results_ale_res = calc_mean(ale_generic_res.to_numpy(), ale_specific_res.to_numpy(), 'ALE generic', 'ALE specific', 'MSE', True, split_resolution=True)
    #results_pdp_splits = calc_mean(pdp_generic_splits, pdp_generic_splits, 'PDP generic', 'PDP specific', 'MSE', True, split_splits=True)
    #results_ale_splits = calc_mean(ale_generic_splits, ale_generic_splits, 'ALE generic', 'ALE specific', 'MSE', True, split_splits=True)

    #results_ale.to_csv('results_ale.csv')
    #results_pdp.to_csv('results_pdp.csv')
    #results_pfi.to_csv('results_pfi.csv')
    results_pdp_res.to_csv('results_pdp_res.csv')
    results_ale_res.to_csv('results_ale_res.csv')
    #results_pdp_splits.to_csv('results_pdp_splits.csv')
    #results_ale_splits.to_csv('results_ale_splits.csv')
