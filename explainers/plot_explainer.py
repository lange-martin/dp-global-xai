import numpy as np
from matplotlib import pyplot as plt


class FeatureEffectExplainer:
    def __init__(self):
        self.explainer_name = '?'
        self.feature_name = '?'
        self.is_categorical = False
        self.feature_values = np.array([])
        self.x_values = np.array([])
        self.y_values = np.array([])
        self.error = 0

    def get_x_y_values(self):
        return self.x_values, self.y_values

    def interpolate(self, x):
        return np.interp(x, self.x_values, self.y_values)

    def get_feature_distribution(self):
        return self.feature_values

    def draw_plot(self, show=True, save=False, filename=None, comparison_plot=None):
        fig, ax = plt.subplots(figsize=(10, 7))

        fontsize = 32

        #ax.set_title(f'{self.explainer_name} for Feature {self.feature_name}', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.locator_params(axis='x', nbins=4)
        ax.set_xlabel(f'{self.feature_name}'.capitalize(), fontsize=fontsize)
        ax.set_ylabel('Prediction', fontsize=fontsize)

        if self.is_categorical:
            ax.bar(self.x_values, self.y_values, width=0.5, color='#828282') #, yerr=self.error, width=0.5, color='#828282')
        else:
            # add plot to compare
            if comparison_plot is not None:
                orig_x_values, orig_y_values = comparison_plot
                ax.plot(orig_x_values, orig_y_values, color='#828282')

            # self.x_values = np.array([x * 47 - 8 for x in self.x_values])
            color = 'black'
            if 'DP PDP' in self.explainer_name or 'DP ALE' in self.explainer_name:
                color = 'red'
            elif 'Generic' in self.explainer_name:
                color = 'blue'

            ax.plot(self.x_values, self.y_values, linewidth=3, color=color)#, marker='o')
            #ax.fill_between(self.x_values, self.y_values-self.error, self.y_values+self.error, color="#d6b1a3")

        ax_feature_distribution = ax.twinx()
        self._show_rug_plot(ax_feature_distribution)

        # horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="-")

        #ax.set_zorder(ax_feature_distribution.get_zorder() + 1)
        #ax.set_frame_on(False)

        plt.subplots_adjust(left=0.2, right=0.8, bottom=0.15)

        if save:
            if filename is None:
                plt.savefig(f'{self.explainer_name}-{self.feature_name}.png')
            else:
                plt.savefig(filename)

        if show:
            plt.show()
        plt.clf()
        plt.close()

    def _show_rug_plot(self, ax):
        # generate rug plot
        ax.set_ylim(0, 1)
        ax.get_yaxis().set_visible(False)

        # show distribution of feature in the given explanation dataset
        zero_repeated = np.tile(np.array([0]), (self.feature_values.shape[0]))

        verts = list(zip([-1., 1., 1., -1.], [-5., -5., 25., 25.]))

        ax.scatter(self.feature_values, zero_repeated, s=2625, c='black', marker=verts, alpha=100/len(self.feature_values))

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=100, class_num=None, verbose=False):
        return self


class DPFeatureEffectExplainer(FeatureEffectExplainer):
    def __init__(self):
        super().__init__()

        # default values
        self.epsilon_histogram = 1
        self.is_int = False
        self.x_min = 0
        self.x_max = 1
        self.y_min = 0
        self.y_max = 1

        self.x_values_hist = np.array([])
        self.y_values_hist = np.array([])

        self.rng = np.random.default_rng(seed=None)

    def _calc_histogram(self, num_x_values):
        if self.is_categorical:
            categories = np.unique(self.feature_values)
            self.x_values_hist = categories

            if self.y_values_hist is not None:
                self.y_values_hist = np.zeros(categories.shape)

                for i in range(categories.shape[0]):
                    self.y_values_hist[i] = len(np.nonzero(self.feature_values == categories[i])[0])

                self.y_values_hist = self.y_values_hist + \
                                     self.rng.laplace(scale=1 / self.epsilon_histogram, size=len(categories))
        else:
            bin_borders = [self.x_min + (i/num_x_values) * (self.x_max - self.x_min) for i in range(num_x_values + 1)]
            bin_borders = np.array(bin_borders)

            x_values_hist = [(bin_borders[i] + bin_borders[i+1]) / 2 for i in range(num_x_values)]
            x_values_hist = np.array(x_values_hist)
            y_values_hist = np.zeros(num_x_values)

            for i in range(num_x_values):
                indices = np.where((self.feature_values >= bin_borders[i]) & (self.feature_values < bin_borders[i+1]))[0]
                y_values_hist[i] = len(indices)

            # add noise
            y_values_hist = y_values_hist + self.rng.laplace(scale=2 / self.epsilon_histogram, size=num_x_values)

            self.x_values_hist = x_values_hist
            self.y_values_hist = y_values_hist

    def _show_rug_plot(self, ax):
        width = (np.amax(self.x_values_hist) - np.amin(self.x_values_hist)) / len(self.x_values_hist)
        ax.bar(self.x_values_hist, self.y_values_hist, width=width, color="#aec6e8", linewidth=0, alpha=0.5)

        ax.set_ylabel('Number of Records', fontsize=32)
        plt.yticks(fontsize=32)

        ax.set_ylim(0, len(self.feature_values))

    def set_privacy_parameters(self, epsilon, is_int, x_min, x_max, y_min, y_max, seed):
        raise NotImplementedError

    def fit(self, x_train, pred_func, feature_index, is_cat=False, num_x_values=100, class_num=None, verbose=False):
        return self