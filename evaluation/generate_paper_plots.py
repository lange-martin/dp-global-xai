from data_loader import AdultIncome, BikeSharing, HeartDisease
from explainers import generic_private_plot
from explainers.accumulated_local_effects import ale
from explainers.accumulated_local_effects import ale_private
from explainers.partial_dependence_plots import pdp
from explainers.partial_dependence_plots import pdp_private

folder = ''
show_original = False


def draw_multiple_graphs(exp_original, demo_original, demo_specific, demo_generic, dataloader, feature,
                         seeds_and_epsilons, compare_bool=True, save=True, resolution=20):
    if save:
        filename_original = f'{folder}{dataloader.name}-{feature}-{exp_original.explainer_name}.pdf'
        print(filename_original)
    else:
        filename_original = None

    comparison_plot = demo_original(dataloader, feature, filename=filename_original, show=show_original)
    if not compare_bool:
        comparison_plot = None

    for (seed, epsilon) in seeds_and_epsilons:
        if save:
            filename_specific = f'{folder}{dataloader.name}-{feature}-DP-{exp_original.explainer_name}-epsilon{epsilon}-res{resolution}-seed{seed}.pdf'
            print(filename_specific)
            filename_generic = f'{folder}{dataloader.name}-{feature}-Generic DP-{exp_original.explainer_name}-epsilon{epsilon}-res{resolution}-seed{seed}.pdf'
            print(filename_generic)
        else:
            filename_specific = None
            filename_generic = None

        demo_specific(dataloader, feature, seed=seed, epsilon=epsilon, compare=comparison_plot,
                      filename=filename_specific, resolution=resolution)
        demo_generic(dataloader, exp_original, feature, seed=seed, epsilon=epsilon, compare=comparison_plot,
                     filename=filename_generic, resolution=resolution)


def try_all(epsilon=1):
    adult = AdultIncome()
    _, adult_data, cont_features, _, _ = adult.load_data()

    for feature in adult_data.columns[:-1]:
        if feature not in cont_features:
            continue
        draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
                             generic_private_plot.demo_one_feature, adult, feature, [(42, epsilon)], save=False)

    bike = BikeSharing()
    _, bike_data, cont_features, _, _ = bike.load_data()

    for feature in bike_data.columns[:-1]:
        if feature not in cont_features:
            continue
        draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
                             generic_private_plot.demo_one_feature, bike, feature, [(42, epsilon)], save=False)

    heart = HeartDisease()
    _, heart_data, cont_features, _, _ = heart.load_data()

    for feature in heart_data.columns[:-1]:
        if feature not in cont_features:
            continue
        draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
                             generic_private_plot.demo_one_feature, heart, feature, [(42, epsilon)], save=False)


if __name__ == '__main__':
    #try_all(epsilon=0.5)

    #seeds_and_epsilons = [(96606, 0.04), (96606, 0.03), (96606, 0.02)]
    #resolutions = [20, 20, 20]
    #for seed_and_epsilon, res in zip(seeds_and_epsilons, resolutions):
    #    draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
    #                         generic_private_plot.demo_one_feature, AdultIncome(), 'age', [seed_and_epsilon],
    #                         save=False, resolution=res)
    #    #draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #    #                     generic_private_plot.demo_one_feature, AdultIncome(), 'capital-gain', [seed_and_epsilon],
    #    #                     resolution=res)

    #seeds_and_epsilons = [(36230, 10), (29630, 10), (42410, 10), (24675, 10), (39138, 10), (28620, 10), (71148, 10),
    #                      (80362, 10), (72612, 10), (93129, 10)]
    seeds_and_epsilons = [(24675, 10)]
    draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
                         generic_private_plot.demo_one_feature, AdultIncome(), 'capital-gain', seeds_and_epsilons)

    #seeds_and_epsilons = [(32228, 10), (92494, 10), (82110, 10), (50227, 10), (76765, 10), (64873, 10), (89680, 10),
    #                      (25431, 10), (43922, 10), (15605, 10)]
    #draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
    #                     generic_private_plot.demo_one_feature, AdultIncome(), 'education-num', seeds_and_epsilons, save=False)

    #seeds_and_epsilons = [(58546, 0.5), (2739, 2), (32622, 10)]
    #draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
    #                     generic_private_plot.demo_one_feature, BikeSharing(), 'hr', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(81327, 0.5), (28842, 2), (36230, 10)]
    # draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'capital-gain', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(26978, 0.5), (55409, 2), (32228, 10)]
    # draw_multiple_graphs(pdp.PartialDependencePlot(), pdp.demo_one_feature, pdp_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'education-num', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(85062, 0.5), (80197, 2), (96231, 10)]
    # draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'age', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(28182, 0.5), (76217, 2), (41125, 10)]
    # draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, HeartDisease(), 'age', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(81327, 0.5), (28842, 2), (36230, 10)]
    # draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'capital-gain', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(64941, 0.5), (45244, 2), (69474, 10)]
    # draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'capital-loss', seeds_and_epsilons)
    #
    # seeds_and_epsilons = [(50362, 0.5), (49182, 2), (41945, 10)]
    # draw_multiple_graphs(ale.AccumulatedLocalEffects(), ale.demo_one_feature, ale_private.demo_one_feature,
    #                      generic_private_plot.demo_one_feature, AdultIncome(), 'native-country', seeds_and_epsilons)
