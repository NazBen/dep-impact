from collections import Counter

from scipy.stats import logistic

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .utils import get_grid_sample, to_copula_params
from .conservative import ListDependenceResult

sns.set(style="ticks", color_codes=True)

COPULA_NAME = {1: "Gaussian",
               2: "Student",
               3: "Clayton",
               4: "Gumbel",
               5: "Frank",
               6: "Joe",
               13: "Gumbel",
               14: "Gumbel",
               16: "Joe"}


def set_style_paper():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def get_all_quantity(results, q_func=None):
    """
    """
    quantities = []
    for res_name in results:
        if q_func is not None:
            # We change the quantity function
            results[res_name].q_func = q_func
        min_quantity = results[res_name].min_quantity
        quantities.append(min_quantity)
    return quantities


def get_all_min_result(results, q_func=None):
    """
    """
    min_results = []
    for res_name in results:
        if q_func is not None:
            # We change the quantity function
            results[res_name].q_func = q_func
        min_result = results[res_name].min_result
        min_results.append(min_result)
    return min_results


def get_min_result(all_min_results, q_func=None):
    """
    """
    min_result = None
    min_quantity = np.inf
    for result in all_min_results:
        if q_func is not None:
            # We change the quantity function
            result.q_func = q_func
        if result.min_quantity < min_quantity:
            min_result = result.min_result
            min_quantity = result.min_quantity

    return min_result


def get_n_pairs(all_results):
    """Get the number of pairs in each experiments of an dictionary of iterative results.
    """
    n_pairs = []
    for results in all_results:
        n_pairs_result = Counter()
        for res_name in results:
            n_pair = results[res_name].n_pairs
            n_pairs_result[n_pair] += 1
        assert len(n_pairs_result) == 1, "Not the same number of pairs... Weird"
        n_pairs.append(n_pair)
    return n_pairs


def corrfunc_plot(x, y, **kws):
    """


    Source: https://stackoverflow.com/a/30942817/5224576
    """
    r, _ = stats.pearsonr(x, y)
    k, _ = stats.kendalltau(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}\nk = {:.2f}".format(r, k),
                xy=(.1, .8), xycoords=ax.transAxes,
                weight='heavy', fontsize=14)


def matrix_plot_input(result, kde=False, margins=None):
    """
    """
    input_sample = result.input_sample

    if margins:
        sample = np.zeros(input_sample.shape)
        for i, marginal in enumerate(margins):
            for j, ui in enumerate(input_sample[:, i]):
                sample[j, i] = marginal.computeCDF(ui)
    else:
        sample = input_sample

    data = pd.DataFrame(sample)
    plot = sns.PairGrid(data, palette=["red"])
    if kde:
        plot.map_upper(plt.scatter, s=10)
        plot.map_lower(sns.kdeplot, cmap="Blues_d")
    else:
        plot.map_offdiag(plt.scatter, s=10)

    plot.map_diag(sns.distplot, kde=False)
    plot.map_lower(corrfunc_plot)

    if margins:
        plot.set(xlim=(0, 1), ylim=(0, 1))

    return plot


def get_color_range(n):
    colors = mcolors.TABLEAU_COLORS
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]
    delta = int(len(sorted_names)/n)
    return sorted_names[::delta]


def get_hull(x, y, n_bins):
    bins = np.linspace(-5., 5., n_bins)
    bins = logistic.cdf(bins) * (2) - 1
    bins = np.linspace(x.min(), x.max(), n_bins)
    down_hull = np.zeros((n_bins-1, ))
    up_hull = np.zeros((n_bins-1, ))
    x_hull = np.zeros((n_bins-1, ))
    for i in range(n_bins-1):
        down, up = bins[i], bins[i+1]
        bin_ids = (x >= down) & (x < up)
        down_id = y[bin_ids].argmin()
        up_id = y[bin_ids].argmax()
        down_hull[i] = y[bin_ids][down_id]
        up_hull[i] = y[bin_ids][up_id]
        x_hull[i] = x[bin_ids][down_id]

    x_hull = bins[:-1] + (bins[1] - bins[0])/2
    return x_hull, down_hull, up_hull


def plot_quantities(results, ratio=(3.5, 2.5), quantity_name=None, label="",
                    plot_scatter=False, plot_hull=True, n_bins=15):

    if isinstance(results, ListDependenceResult):
        n_plot = 1
        dim = results.input_dim
        kendalls = [results.kendalls]
        quantities = [results.quantities]
        label = [label]
    elif isinstance(results, list):
        n_plot = len(results)
        dim = results[0].input_dim
        kendalls = []
        quantities = []
        for result in results:
            kendalls.append(result.kendalls)
            quantities.append(result.quantities)
            assert result.input_dim == dim, "The dimension should be the same for all plots."

    if dim == 2:
        plot_hull = False
        plot_scatter = True

    fig, axes = plt.subplots(
        dim-1, dim-1, figsize=(dim*ratio[0], dim*ratio[1]), sharex=True, sharey=True)

    colors = get_color_range(n_plot)
    for i_plot in range(n_plot):
        k = 0
        for i in range(dim-1):
            for j in range(i+1):
                if dim == 2:
                    ax = axes
                    sorted_items = np.argsort(kendalls[i_plot].ravel())
                    kendalls[i_plot] = kendalls[i_plot][sorted_items]
                    quantities[i_plot] = quantities[i_plot][sorted_items]
                    linestyle = '-'
                else:
                    ax = axes[i, j]
                    linestyle = ''

                x, y = kendalls[i_plot][:, k], quantities[i_plot]
                if plot_scatter:
                    h = ax.plot(x, y, '.',
                                linestyle=linestyle, label=label[i_plot])
                    color = h[0].get_color()
                else:
                    color = colors[i_plot]

                if plot_hull:
                    x_hull, y_down, y_up = get_hull(x, y, n_bins)
                    h = ax.plot(x_hull, y_down, '--', color=color,
                                label='Down hull '+label[i_plot])
                    color = h[0].get_color()
                    ax.plot(x_hull, y_up, '--', color=color,
                            label='Up hull '+label[i_plot])

                ax.set_xlabel('$\\tau_{%d, %d}$' % (j+1, i+2))
                ax.set_xlim(-1, 1.)

                if j == 0:
                    ax.set_ylabel(quantity_name)
                k += 1

    if dim == 2:
        ax.legend(loc=0)
    else:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, dim-2].legend(handles, labels)

    return fig, axes


def matrix_plot_quantities(results, indep_result=None, grid_result=None,
                           q_func=None, figsize=(9, 7), dep_measure='kendalls',
                           quantity_name='Quantity', with_bootstrap=False):
    """
    """
    if isinstance(results, dict):
        input_dim = list(results.values())[0].input_dim
    else:
        input_dim = results.input_dim

    # Figure
    fig, axes = plt.subplots(input_dim, input_dim,
                             figsize=figsize, sharex=True, sharey=True)
    for res in results:
        t = res.split(', ')[-1:-3:-1]
        i = int(t[1][1])
        j = int(t[0][0])
        ax = axes[i, j]
        if dep_measure == 'dependence-param':
            measure = results[res].dep_params
        else:
            measure = results[res].kendalls
        quantities = results[res].quantities
        ax.plot(measure, quantities, '.')

    if dep_measure == 'dependence-param':
        x_label = 'Dependence Parameter'
    else:
        x_label = 'Kendall tau'
    for i in range(input_dim):
        axes[i, 0].set_ylabel(quantity_name)
        axes[-1, i].set_xlabel(x_label)

    fig.tight_layout()


def plot_iterative_results_2(iter_results, indep_result=None, grid_results=None, q_func=None, diff_with_indep=False, figsize=(8, 4),
                             quantity_name='Quantity', with_bootstrap=False, n_boot=200, ax=None, color='r', with_all_quantities=True):
    """
    """

    # Figure
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Number of iteration
    n_levels = iter_results.iteration+1
    dim = iter_results.dim

    # Number of pairs at each iteration
    n_pairs = range(n_levels)

    if indep_result is not None:
        if not diff_with_indep:
            ax.plot([n_pairs[0], n_pairs[-1]], [indep_result.quantity]*2, '-o',
                    color='k', label='Independence')
        else:
            ax.plot([n_pairs[0], n_pairs[-1]], [0]*2, '--',
                    color='k')

    if grid_results is not None:
        min_grid_result = grid_results.min_result
        ax.plot([n_pairs[0], n_pairs[-1]], [min_grid_result.quantity]*2, '-o',
                color='b', label='Grid-search with $K=%d$' % (grid_results.n_params))

    quantities = []
    min_results_level = []
    selected_families = []

    last_families = iter_results.min_result(-1).families
    for lvl in range(n_levels):
        # All the results
        values = iter_results.min_quantities(lvl)[np.tril_indices(dim, -1)]
        values = values[values != 0.].tolist()
        min_result = iter_results.min_result(lvl)
        quantities.append(values)
        min_results_level.append(min_result)
        pair = iter_results.selected_pairs[lvl][-1]
        selected_families.append(last_families[pair])

    # Get the minimum of each level
    min_quantities = []
    for quant_lvl in quantities:
        min_quant = min(quant_lvl)
        min_quantities.append(min_quant)
        # Remove the minimum from the list of quantities
        # It's repetitve with the othr list
        quant_lvl.remove(min_quant)

    for lvl in range(n_levels):
        # The quantities of this level
        quant_lvl = np.asarray(quantities[lvl])
        # The number of results
        n_res = len(quant_lvl)
        if not diff_with_indep:
            y_quantities = quant_lvl
            y_min_quantity = min_quantities[lvl]
        else:
            y_quantities = quant_lvl - indep_result.quantity
            y_min_quantity = min_quantities[lvl] - indep_result.quantity

        if with_all_quantities:
            if not diff_with_indep:
                ax.plot([n_pairs[lvl]]*n_res, y_quantities, '.', color=color)
            else:
                ax.plot([n_pairs[lvl]]*n_res, y_quantities, '.', color=color)

        if n_pairs[lvl] == n_pairs[-1]:
            ax.plot(n_pairs[lvl], y_min_quantity,
                    'o', color=color)
        else:
            ax.plot(n_pairs[lvl], y_min_quantity, 'o', color=color)
            ax.plot([n_pairs[lvl], n_pairs[lvl+1]],
                    [y_min_quantity]*2, '-', color=color)

    ax.axis('tight')
    ax.set_ylabel(quantity_name, fontsize=12)

    ax.set_xticks(n_pairs)
    ax.set_xticklabels(['$k=%d$' % (lvl) for lvl in range(n_levels)])


def plot_iterative_results(iter_results, indep_result=None, grid_results=None, q_func=None, diff_with_indep=False, figsize=(8, 4),
                           quantity_name='Quantity', with_bootstrap=False, n_boot=200, ax=None):
    """
    """

    # Figure
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Number of iteration
    n_levels = iter_results.iteration+1
    dim = iter_results.dim

    # Colors of the levels and independence
    cmap = plt.get_cmap('jet')

    n_p = 0  # Number of additional plots
    n_p += 1 if indep_result is not None else 0
    n_p += 1 if grid_results is not None else 0
    colors = [cmap(i) for i in np.linspace(0, 1, n_levels+n_p)]

    # Number of pairs at each iteration
    n_pairs = range(n_levels)

    if indep_result is not None:
        if not diff_with_indep:
            ax.plot([n_pairs[0], n_pairs[-1]], [indep_result.quantity]*2, '-o',
                    color=colors[0], label='Independence')
        else:
            ax.plot([n_pairs[0], n_pairs[-1]], [0]*2, '--',
                    color=colors[0])

        if with_bootstrap:
            indep_result.compute_bootstrap()
            boot = indep_result.bootstrap_sample

            up = np.percentile(boot, 99)
            down = np.percentile(boot, 1)
            ax.plot([n_pairs[0], n_pairs[-1]], [up]*2, '-.',
                    color=colors[0], linewidth=0.8)
            ax.plot([n_pairs[0], n_pairs[-1]], [down]*2, '-.',
                    color=colors[0], linewidth=0.8)

    if grid_results is not None:
        min_grid_result = grid_results.min_result
        ax.plot([n_pairs[0], n_pairs[-1]], [min_grid_result.quantity]*2, '-o',
                color=colors[1], label='Grid-search with $K=%d$' % (grid_results.n_params))
        if with_bootstrap:
            min_grid_result.compute_bootstrap()
            boot = min_grid_result.bootstrap_sample
            up = np.percentile(boot, 95)
            down = np.percentile(boot, 5)
            ax.plot([n_pairs[0], n_pairs[-1]], [up]*2, '--',
                    color=colors[1], linewidth=0.8)
            ax.plot([n_pairs[0], n_pairs[-1]], [down]*2, '--',
                    color=colors[1], linewidth=0.8)

    quantities = []
    min_results_level = []
    selected_families = []

    # TODO : bug in iterative algorithm for the saving results of families
    last_families = iter_results.min_result(-1).families
    for lvl in range(n_levels):
        # All the results
        values = iter_results.min_quantities(lvl)[np.tril_indices(dim, -1)]
        values = values[values != 0.].tolist()
        min_result = iter_results.min_result(lvl)
        quantities.append(values)
        min_results_level.append(min_result)
        pair = iter_results.selected_pairs[lvl][-1]
        selected_families.append(last_families[pair])

    # Get the minimum of each level
    min_quantities = []
    for quant_lvl in quantities:
        min_quant = min(quant_lvl)
        min_quantities.append(min_quant)
        # Remove the minimum from the list of quantities
        # It's repetitve with the othr list
        quant_lvl.remove(min_quant)

    for lvl in range(n_levels):
        # The quantities of this level
        quant_lvl = np.asarray(quantities[lvl])
        # The number of results
        n_res = len(quant_lvl)
        if not diff_with_indep:
            y_quantities = quant_lvl
            y_min_quantity = min_quantities[lvl]
        else:
            y_quantities = quant_lvl - indep_result.quantity
            y_min_quantity = min_quantities[lvl] - indep_result.quantity

        if not diff_with_indep:
            ax.plot([n_pairs[lvl]]*n_res, y_quantities,
                    '.', color=colors[lvl+n_p])
        else:
            ax.plot([n_pairs[lvl]]*n_res, y_quantities,
                    '.', color=colors[lvl+n_p])

        if n_pairs[lvl] == n_pairs[-1]:
            ax.plot(n_pairs[lvl], y_min_quantity,
                    'o', color=colors[lvl+n_p])

            if with_bootstrap:
                min_results_level[lvl].compute_bootstrap(n_boot)
                boot = min_results_level[lvl].bootstrap_sample
                up = np.percentile(boot, 95)
                down = np.percentile(boot, 5)
                ax.plot(n_pairs[lvl], up, '.',
                        color=colors[lvl+n_p], linewidth=0.8)
                ax.plot(n_pairs[lvl], down, '.',
                        color=colors[lvl+n_p], linewidth=0.8)
        else:
            ax.plot(n_pairs[lvl], y_min_quantity, 'o', color=colors[lvl+n_p])
            ax.plot([n_pairs[lvl], n_pairs[lvl+1]],
                    [y_min_quantity]*2, '-', color=colors[lvl+n_p])

            if with_bootstrap:
                min_results_level[lvl].compute_bootstrap(n_boot)
                boot = min_results_level[lvl].bootstrap_sample
                up = np.percentile(boot, 95)
                down = np.percentile(boot, 5)
                ax.plot([n_pairs[lvl], n_pairs[lvl+1]], [up]*2, '--',
                        color=colors[lvl+n_p], linewidth=0.8)
                ax.plot([n_pairs[lvl], n_pairs[lvl+1]], [down]*2, '--',
                        color=colors[lvl+n_p], linewidth=0.8)

    ax.axis('tight')
    # ax.set_xlabel('Iterations')
    ax.set_ylabel(quantity_name, fontsize=12)

    selected_pairs = iter_results.selected_pairs
    x_label = []
    for lvl in range(n_levels):
        i, j = selected_pairs[lvl][-1]
        copula_name = COPULA_NAME[selected_families[lvl]]
        label = '$k=%d$\n$X_%d-X_%d$\n%s' % (lvl, j+1, i+1, copula_name)
        x_label.append(label)

    ax.set_xticks(n_pairs)
    ax.set_xticklabels(x_label)
    ax.legend(loc=0)


def compute_influence(obj, K, n, copulas, pair, eps=1.E-4):
    """
    """
    kendalls_fixed = [[]]*2
    bounds = [[eps, 1.-eps]]
    kendalls_fixed[0] = get_grid_sample(bounds, K, 'lhs')
    bounds = [[-1.+eps, -eps]]
    kendalls_fixed[1] = get_grid_sample(bounds, K, 'lhs')

    families = np.zeros((obj.families.shape), dtype=int)
    families[pair[0], pair[1]] = 1
    obj.families = families
    indep_output_sample = obj.independence(n).output_sample
    perfect_output_sample = obj.gridsearch(None, n, 'vertices').output_samples

    output_samples = {}
    for copula in copulas:
        res_out_samples = []
        res_kendalls = []
        for i, num in enumerate(copulas[copula]):
            families[pair[0], pair[1]] = num
            obj.families = families
            converter = [obj._copula_converters[k] for k in obj._pair_ids]
            params = to_copula_params(converter, kendalls_fixed[i])
            output_sample = obj.run_stochastic_models(
                params, n, return_input_sample=False)[0]
            res_out_samples.append(np.asarray(output_sample))

        output_samples[copula] = np.r_[np.concatenate(
            res_out_samples), indep_output_sample.reshape(1, -1), perfect_output_sample].T

    kendalls = np.concatenate(kendalls_fixed).ravel()
    kendalls = np.r_[kendalls, 0.]
    kendalls = np.r_[kendalls, -1., 1.]

    return kendalls, output_samples


def plot_variation(output_samples, kendalls, q_func, plot_area='left', plt_lib='seaborn', figsize=(7, 4), ylabel=None,
                   colors={'Normal': 'b', 'Clayton': 'g', 'Gumbel': 'r', 'Joe': 'm'}, n_boot=5000, ci=99.9):
    """
    """
    set_style_paper()

    if plot_area == 'full':
        taken = np.ones(kendalls.shape, dtype=bool)
    elif plot_area == 'left':
        taken = kendalls <= 0.
    elif plot_area == 'right':
        taken = kendalls >= 0.

    sorting = np.argsort(kendalls[taken])
    fig, ax = plt.subplots(figsize=figsize)

    for copula in output_samples:
        if plt_lib == 'matplotlib':
            quantities = q_func(output_samples[copula].T)
            ax.plot(kendalls[taken][sorting], quantities[taken]
                    [sorting], 'o-', label=copula, markersize=5)
        else:
            sns.tsplot(output_samples[copula][:, taken], time=kendalls[taken],
                       condition=copula, err_style='ci_band', ci=ci, estimator=q_func,
                       n_boot=n_boot, color=colors[copula], ax=ax)

    ax.set_xlabel('Kendall coefficient')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('Output quantity')
    ax.legend(loc=0)
    ax.axis('tight')
    fig.tight_layout()
