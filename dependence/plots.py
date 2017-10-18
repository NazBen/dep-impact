import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from collections import Counter

from .utils import get_grid_sample, to_copula_params

sns.set(style="ticks", color_codes=True)


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
    fig, axes = plt.subplots(input_dim, input_dim, figsize=figsize, sharex=True, sharey=True)
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
    
    
def plot_iterative_results(all_results, indep_result=None, grid_results=None, 
                           q_func=None, figsize=(8, 4), 
                           quantity_name='Quantity', with_bootstrap=False, n_boot=200):
    """
    """
    
    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Number of trees
    n_levels = len(all_results)
    
    # Colors of the levels and independence
    cmap = plt.get_cmap('jet')
    n_p = 0  # Number of additional plots
    n_p += 1 if indep_result is not None else 0
    n_p += 1 if grid_results is not None else 0
    colors = [cmap(i) for i in np.linspace(0, 1, n_levels+n_p)]
    
    # Number of pairs at each iteration
    n_pairs = get_n_pairs(all_results)
    
    if indep_result is not None:
        ax.plot([n_pairs[0], n_pairs[-1]], [indep_result.quantity]*2, '-o', 
                color=colors[0], label='independence')
        
        if with_bootstrap:
            indep_result.compute_bootstrap()
            boot = indep_result.bootstrap_sample
            
            up = np.percentile(boot, 99)
            down = np.percentile(boot, 1)
            ax.plot([n_pairs[0], n_pairs[-1]], [up]*2, '--', 
                color=colors[0], linewidth=0.8)
            ax.plot([n_pairs[0], n_pairs[-1]], [down]*2, '--', 
                color=colors[0], linewidth=0.8)
        
    
    if grid_results is not None:
        min_grid_result = grid_results.min_result
        ax.plot([n_pairs[0], n_pairs[-1]], [min_grid_result.quantity]*2, '-o', 
                color=colors[1], label='grid-search with $K=%d$' % (grid_results.n_params))
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
    for result_by_level in all_results:
        quantities.append(get_all_quantity(result_by_level, q_func=q_func))
        min_results_level.append(get_min_result(result_by_level.values(), q_func=q_func))

    # Get the minimum of each level
    min_quantities = []
    for quant_lvl in quantities:
        min_quant = min(quant_lvl)
        min_quantities.append(min_quant)
        
        # Remove the minimum from the list of quantities
        quant_lvl.remove(min_quant)
    
    for lvl in range(n_levels):
        # The quantities of this level
        quant_lvl = np.asarray(quantities[lvl])
        # The number of results
        n_res = len(quant_lvl)
        ax.plot([n_pairs[lvl]]*n_res, quant_lvl, '.', color=colors[lvl+n_p])
        
    for lvl in range(n_levels):
        if n_pairs[lvl] == n_pairs[-1]:
            ax.plot(n_pairs[lvl], min_quantities[lvl], 'o', color=colors[lvl+n_p])
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
            ax.plot([n_pairs[lvl], n_pairs[lvl+1]], [min_quantities[lvl]]*2, 'o-', color=colors[lvl+n_p])
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
    ax.set_xlabel('Number of considered pairs')
    ax.set_ylabel(quantity_name)
    ax.set_xticks(n_pairs)
    ax.legend(loc=0)
    fig.tight_layout()        


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
            output_sample = obj.run_stochastic_models(params, n, return_input_sample=False)[0]
            res_out_samples.append(np.asarray(output_sample))

        output_samples[copula] = np.r_[np.concatenate(res_out_samples), indep_output_sample.reshape(1, -1), perfect_output_sample].T

    kendalls = np.concatenate(kendalls_fixed).ravel()
    kendalls = np.r_[kendalls, 0.]
    kendalls = np.r_[kendalls, -1., 1.]
    
    return kendalls, output_samples

def plot_variation(output_samples, kendalls, q_func, plot_area='left', plt_lib='seaborn', figsize=(7, 4), ylabel=None, 
    colors={'Normal': 'b','Clayton': 'g', 'Gumbel': 'r', 'Joe': 'm'}, n_boot=5000, ci=99.9):
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
            ax.plot(kendalls[taken][sorting], quantities[taken][sorting], 'o-', label=copula, markersize=5)
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