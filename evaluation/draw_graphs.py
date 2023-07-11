import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset_index = 0
epsilon_index = 1
feature_index = 2
resolution_index = 3
mse_generic_index = 4
std_generic_index = 5
mse_specific_index = 6
std_specific_index = 7
splits_index = 8


def draw_graph(data, x_values, name, with_std=False, save=False, with_feature=True, y_label='MISE', log=True, extra_index=None):
    dataset = data[0, dataset_index]

    fig, ax = plt.subplots(figsize=(10, 7))
    fontsize = 32

    if with_feature:
        feature_name = data[0, feature_index]
        #ax.set_title(f'Dataset {dataset}, feature {feature_name}', fontsize=fontsize)
    else:
        feature_name = 'nan'
        #ax.set_title(f'Dataset {dataset}', fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax.set_xlabel(f'ε', fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)

    if extra_index is not None:
        extras = np.unique(data[:, extra_index])
    else:
        extras = [None]

    for i, item in enumerate(extras):
        if extra_index is not None:
            subset_indices = np.where(data[:, extra_index] == item)
            subset = data[subset_indices]
        else:
            subset = data

        mse_generic = subset[:, mse_generic_index]
        std_generic = subset[:, std_generic_index]
        mse_specific = subset[:, mse_specific_index]
        std_specific = subset[:, std_specific_index]

        if with_std:
            ax.errorbar(x_values, mse_generic, yerr=std_generic, fmt="bo-")
            ax.errorbar(x_values, mse_specific, yerr=std_specific, fmt="ro-")
        else:
            if extra_index is not None:
                gap = 5
            else:
                gap = 0
            line1, = ax.plot(x_values, mse_generic, "bo", linewidth=3, linestyle=(0, (3**(len(extras)-i), gap)))
            line2, = ax.plot(x_values, mse_specific, "ro", linewidth=3, linestyle=(0, (3**(len(extras)-i), gap)))

            if (feature_name == 'nan' and dataset == 'Census Income') or (feature_name == 'age' and dataset == 'Adult Income' and extra_index is None):
                ax.legend([line1, line2], ['Generic', 'Explainer-specific'], fontsize=fontsize)

        #if extra_index is not None:
        #    index = 0
        #    if i % 2 == 0:
        #        index = 4
        #    plt.text(x_values[index], mse_generic[index], f'{item}', fontsize=fontsize)
        #    plt.text(x_values[index], mse_specific[index], f'{item}', fontsize=fontsize)

    # add horizontal line where MSE meets squared range of original plot by explainer
    # if the MSE lies below this point, we may expect to actually be able to tell something from the private explanation
    if name in ['ale', 'pdp']:
        range = dict_range[(name, dataset, feature_name)]
        ax.axhline(y=range**2, color="black", linestyle='dotted')

    if log:
        plt.yscale('log')

    plt.subplots_adjust(left=0.2, bottom=0.2)

    if save:
        if extra_index is not None:
            plt.savefig(f'plots/{name}-dataset-{dataset}-feature-{feature_name}-extra-{extra_index}.png')
        else:
            plt.savefig(f'plots/{name}-dataset-{dataset}-feature-{feature_name}.png')

    plt.show()
    plt.clf()


def draw_graphs_ale_pdp(explainer, save=False):
    results = pd.read_csv(f'results_{explainer}.csv').to_numpy()
    results = results[:, 1:]
    datasets = np.unique(results[:, dataset_index])
    epsilons = np.unique(results[:, epsilon_index])

    features_per_dataset = {}
    for d in datasets:
        indices = np.where(results[:, dataset_index] == d)
        features = np.unique(results[indices, feature_index])
        features_per_dataset[d] = features

    for d in datasets:
        for f in features_per_dataset[d]:
            graph_data_indices = np.where((results[:, dataset_index] == d) & (results[:, feature_index] == f))
            graph_data = results[graph_data_indices]
            draw_graph(graph_data, epsilons, name=explainer, save=save)


def draw_graphs_pfi(save=False):
    df = pd.read_csv(f'results_pfi.csv')
    df = df.drop(df.columns[0], axis=1)
    results = df.to_numpy()
    datasets = np.unique(results[:, dataset_index])
    epsilons = np.unique(results[:, epsilon_index])

    for d in datasets:
        graph_data_indices = np.where(results[:, dataset_index] == d)
        graph_data = results[graph_data_indices]
        draw_graph(graph_data, epsilons, name='pfi', save=save, with_feature=False, y_label='Kendall Tau Metric', log=False)


def draw_graphs_res(explainer, save=False):
    df = pd.read_csv(f'results_{explainer}_res.csv')
    df = df.drop(df.columns[0], axis=1)
    results = df.to_numpy()
    datasets = np.unique(results[:, dataset_index])
    epsilons = np.unique(results[:, epsilon_index])

    features_per_dataset = {}
    for d in datasets:
        indices = np.where(results[:, dataset_index] == d)
        features = np.unique(results[indices, feature_index])
        features_per_dataset[d] = features

    for d in datasets:
        for f in features_per_dataset[d]:
            graph_data_indices = np.where((results[:, dataset_index] == d) & (results[:, feature_index] == f))
            graph_data = results[graph_data_indices]
            draw_graph(graph_data, epsilons, name=explainer, save=save, extra_index=resolution_index)

def draw_graphs_splits(explainer, save=False):
    df = pd.read_csv(f'results_{explainer}_splits.csv')
    df = df.drop(df.columns[0], axis=1)
    results = df.to_numpy()
    datasets = np.unique(results[:, dataset_index])
    epsilons = np.unique(results[:, epsilon_index])

    features_per_dataset = {}
    for d in datasets:
        indices = np.where(results[:, dataset_index] == d)
        features = np.unique(results[indices, feature_index])
        features_per_dataset[d] = features

    for d in datasets:
        for f in features_per_dataset[d]:
            graph_data_indices = np.where((results[:, dataset_index] == d) & (results[:, feature_index] == f))
            graph_data = results[graph_data_indices]
            draw_graph(graph_data, epsilons, name=explainer, save=save, extra_index=splits_index)


dict_range = {('ale', 'Adult Income', 'age'): 0.20614093471419837,
('ale', 'Adult Income', 'workclass'): 0.0778476583333681,
('ale', 'Adult Income', 'education'): 0.055370145692349965,
('ale', 'Adult Income', 'education-num'): 0.24351744813246679,
('ale', 'Adult Income', 'marital-status'): 0.21533331633512867,
('ale', 'Adult Income', 'occupation'): 0.1933433062022896,
('ale', 'Adult Income', 'relationship'): 0.12975495635785253,
('ale', 'Adult Income', 'race'): 0.020564935064935067,
('ale', 'Adult Income', 'sex'): 0.02035782268118335,
('ale', 'Adult Income', 'capital-gain'): 0.8151934073892257,
('ale', 'Adult Income', 'capital-loss'): 0.33873430961597073,
('ale', 'Adult Income', 'hours-per-week'): 0.09439184289577998,
('ale', 'Adult Income', 'native-country'): 0.05683776178096586,
('ale', 'Bike Sharing', 'yr'): 79.85783681797156,
('ale', 'Bike Sharing', 'mnth'): 67.33494736491186,
('ale', 'Bike Sharing', 'hr'): 416.9171756512876,
('ale', 'Bike Sharing', 'holiday'): 107.00052777080205,
('ale', 'Bike Sharing', 'weekday'): 34.74751546926667,
('ale', 'Bike Sharing', 'workingday'): 3.4589371416682835,
('ale', 'Bike Sharing', 'weathersit'): 91.02972496066575,
('ale', 'Bike Sharing', 'atemp'): 143.6084556618095,
('ale', 'Bike Sharing', 'hum'): 40.95802013465075,
('ale', 'Bike Sharing', 'windspeed'): 59.628545117887825,
('ale', 'Heart Disease', 'sex'): 0.046106194690265494,
('ale', 'Heart Disease', 'age'): 0.2665615615615616,
('ale', 'Heart Disease', 'education'): 0.01919389214125772,
('ale', 'Heart Disease', 'smoker'): 0.019561027837259104,
('ale', 'Heart Disease', 'cigs_per_day'): 0.06176426426426426,
('ale', 'Heart Disease', 'bp_meds'): 0.04408462623413258,
('ale', 'Heart Disease', 'prevalent_stroke'): 0.04413480055020633,
('ale', 'Heart Disease', 'prevelant_hyp'): 0.06841875248311483,
('ale', 'Heart Disease', 'diabetes'): 0.03165870115265673,
('ale', 'Heart Disease', 'total_chol'): 0.2987162162162162,
('ale', 'Heart Disease', 'sys_bp'): 0.2947597597597598,
('ale', 'Heart Disease', 'dia_bp'): 0.18548798798798793,
('ale', 'Heart Disease', 'bmi'): 0.19381381381381385,
('ale', 'Heart Disease', 'heart_rate'): 0.0728978978978979,
('ale', 'Heart Disease', 'glucose'): 0.3607057057057057,
('pdp', 'Adult Income', 'age'): 0.2001645251566848,
('pdp', 'Adult Income', 'workclass'): 0.05272448898504112,
('pdp', 'Adult Income', 'education'): 0.037066556059988265,
('pdp', 'Adult Income', 'education-num'): 0.24795344544102865,
('pdp', 'Adult Income', 'marital-status'): 0.18542184182870733,
('pdp', 'Adult Income', 'occupation'): 0.10824019681599995,
('pdp', 'Adult Income', 'relationship'): 0.15070142357158928,
('pdp', 'Adult Income', 'race'): 0.02739923838868691,
('pdp', 'Adult Income', 'sex'): 0.013258877456667428,
('pdp', 'Adult Income', 'capital-gain'): 0.7321438042844777,
('pdp', 'Adult Income', 'capital-loss'): 0.3552390167471995,
('pdp', 'Adult Income', 'hours-per-week'): 0.15933923281338025,
('pdp', 'Adult Income', 'native-country'): 0.046312538952546084,
('pdp', 'Bike Sharing', 'yr'): 82.14459427339833,
('pdp', 'Bike Sharing', 'mnth'): 61.18866922838919,
('pdp', 'Bike Sharing', 'hr'): 381.49664872189936,
('pdp', 'Bike Sharing', 'holiday'): 102.09741616231864,
('pdp', 'Bike Sharing', 'weekday'): 38.17950895808525,
('pdp', 'Bike Sharing', 'workingday'): 21.869976720119865,
('pdp', 'Bike Sharing', 'weathersit'): 95.76479886639623,
('pdp', 'Bike Sharing', 'atemp'): 178.23792258287187,
('pdp', 'Bike Sharing', 'hum'): 49.21053422918027,
('pdp', 'Bike Sharing', 'windspeed'): 62.882267704589424,
('pdp', 'Heart Disease', 'sex'): 0.040259846827133516,
('pdp', 'Heart Disease', 'age'): 0.2398851203501094,
('pdp', 'Heart Disease', 'education'): 0.020046498905908094,
('pdp', 'Heart Disease', 'smoker'): 0.01583150984682713,
('pdp', 'Heart Disease', 'cigs_per_day'): 0.09354759299781182,
('pdp', 'Heart Disease', 'bp_meds'): 0.04332603938730853,
('pdp', 'Heart Disease', 'prevalent_stroke'): 0.04433260393873084,
('pdp', 'Heart Disease', 'prevelant_hyp'): 0.04874452954048139,
('pdp', 'Heart Disease', 'diabetes'): 0.03129649890590808,
('pdp', 'Heart Disease', 'total_chol'): 0.24652078774617067,
('pdp', 'Heart Disease', 'sys_bp'): 0.22603391684901533,
('pdp', 'Heart Disease', 'dia_bp'): 0.1871198030634573,
('pdp', 'Heart Disease', 'bmi'): 0.191687636761488,
('pdp', 'Heart Disease', 'heart_rate'): 0.0687636761487965,
('pdp', 'Heart Disease', 'glucose'): 0.24554157549234135
              }


if __name__ == '__main__':
    #draw_graphs_pfi(save=True)
    #draw_graphs_ale_pdp(explainer='ale', save=True)
    #draw_graphs_ale_pdp(explainer='pdp', save=True)
    draw_graphs_res(explainer='pdp', save=True)
    #draw_graphs_res(explainer='ale', save=True)

    #draw_graphs_splits(explainer='ale', save=False)
