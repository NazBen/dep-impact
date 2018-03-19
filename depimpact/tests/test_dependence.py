"""Test of the dependence library.

TODO:
    New tests to add:
        - additive gaussian problem
            * grid search with rand, lhs, fixed
            * on several function of interest
            * with bounds on the parameter
            * with kendall or not 
            * same with the iterative
            * with perfect dependencies
        - saving/loading data
"""

import numpy as np
import openturns as ot
from scipy.special import erf, erfinv
from numpy.testing import assert_allclose

from depimpact import ConservativeEstimate
from depimpact.utils import quantile_func, proba_func
from depimpact.tests import func_sum
from depimpact.iterative_vines import iterative_vine_minimize

QUANTILES_PROB = [0.05, 0.01]
PROB_THRESHOLDS = [1., 2.]
DIMENSIONS = range(2, 10)
GRIDS = ['lhs', 'rand', 'vertices']
COPULA = ["NormalCopula", "ClaytonCopula"]
MEASURES = ["dependence-parameter", "kendall-tau"]

def dep_params_list_to_matrix(params, dim):
    """
    """
    if params == 0:
        return np.identity(dim)

    sigma = np.ones((dim, dim))
    k = 0
    for i in range(1, dim):
        for j in range(i):
            sigma[i, j] = params[k]
            sigma[j, i] = params[k]
            k += 1
    return sigma

def true_additive_gaussian_quantile(alpha, dim, sigma, nu=None, const=None):
    """The output quantile of an additive problem with
    gaussian distritution and linearly correlated.

    Parameters
    ----------
    alpha : float
        Quantile probability :math:`\alpha`.

    dim : int
        Problem dimension

    sigma : :class:`~numpy.ndarray`
        Covariance matrix.

    nu : :class:`~numpy.ndarray`, optional (default=None)
        Mean vector. If None, the mean is at zero.

    const : :class:`~numpy.ndarray`, optional (default=None)
        

    Returns
    -------
    quantile : float
        The output quantile.

    We consider the random vector :math:`\mathbf X` with a :math:`d`-dimensional multivariate
    normal distribution such as :math:`\mathbf X \sim \mathcal (\nu, \Sigma)` with 
    :math:`\nu = (\nu_1, \dots, \nu_d)^T` the mean vector and the covariance matrix :math:`\Sigma`,
    such as :math:`\Sigma_{ij} = cov(X_i, X_j)` for :math:`i, j = 1, \dots, d`.

    We define :math:`Y = \mathbf a^T \mathbf X = \sum_{j=1} a_j X_j`. The output variable is 
    also normaly distributed :math:`Y \sim \mathcal N (\mu_Y, \sigma^2_Y)` with mean :math:`\mu_Y=\sum_{j=1}^{d}a_j\mu_j` and
    variance :math:`\sigma^2_Y=\sum_{j=1}^{d}a_j^2\Sigma_{jj}+2\sum_{j=2}^{d}\sum_{k=1}^{j-1}a_ja_k\Sigma_{jk}`

    Thanks to @probabilityislogic for the detailed response at http://stats.stackexchange.com/a/19953.
    """
    if nu == None:
        nu = np.zeros((dim, 1))
    if const == None:
        const = np.ones((1, dim))

    tmp = np.hstack(sigma[i][:i] for i in range(sigma.shape[0]))
    var_y = (const**2 * sigma.diagonal()).sum() + (2*tmp).sum()
    sigma_y = np.sqrt(var_y)
    quantile = sigma_y * np.sqrt(2.) * erfinv(2 * alpha - 1.)
    return quantile


def true_additive_gaussian_probability(x, dim, sigma, nu=None, const=None):
    """
    """
    if nu == None:
        nu = np.zeros((dim, 1))
    if const == None:
        const = np.ones((1, dim))

    tmp = np.hstack(sigma[i][:i] for i in range(sigma.shape[0]))
    var_y = (const**2 * sigma.diagonal()).sum() + (2*tmp).sum()
    sigma_y = np.sqrt(var_y)
    return 0.5 * (1. + erf(x / (sigma_y * np.sqrt(2.))))


def check_dims(obj, dim):
    """
    """
    assert len(obj.margins) == dim
    assert obj.families.shape[0] == dim
    assert obj.vine_structure.shape[0] == dim
    assert obj.bounds_tau.shape[0] == dim
    assert obj.fixed_params.shape[0] == dim
    
    
def test_modification_dimension():
    dim = 2
    families = np.tril(np.ones((dim, dim)), k=-1)
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families)    
    check_dims(impact, dim)
    for dim in range(3, 6):
        impact.margins = [ot.Normal()]*dim
        assert len(impact.margins) == dim
        
        impact.families = np.tril(np.ones((dim, dim)), k=-1)
        check_dims(impact, dim)
        
        # Test Grid results
        impact.gridsearch(
            n_dep_param=10, 
            n_input_sample=10, 
            grid_type='lhs', 
            random_state=0)


def test_modification_families():
    dim = 8
    families = np.tril(np.ones((dim, dim)), k=-1)
    
    ind_pair = [3, 2]
    families[ind_pair[0], ind_pair[1]] = 0
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families)
    check_dims(impact, dim)
        
    # Test Grid results
    impact.gridsearch(
        n_dep_param=10, 
        n_input_sample=10, 
        grid_type='lhs', 
        
        random_state=0)
    
    n_ind_pairs = 2
    ind_pairs = [ind_pair]
    for p in range(n_ind_pairs):
        # Set a family to independence
        condition = True
        while condition:
            i = np.random.randint(1, dim)
            j = np.random.randint(0, i)
            pair = [i, j] 
            if pair not in ind_pairs:
                ind_pairs.append(pair)
                condition = False
        families[pair[0], pair[1]] = 0
        impact.families = families
        pairs_lvl = get_tree_pairs(impact.vine_structure, 0)
        for ind_pair in ind_pairs:
            assert (ind_pair in pairs_lvl) or (list(reversed(ind_pair)) in pairs_lvl) 
        check_dims(impact, dim)
        
        # Test Grid results
        impact.gridsearch(
            n_dep_param=10, 
            n_input_sample=10, 
            grid_type='lhs', 
            random_state=0)
        

def test_modification_fixed_params():
    dim = 10
    families = np.tril(np.ones((dim, dim)), k=-1)
    fixed_params = np.zeros((dim, dim))
    fixed_params[:] = np.nan
    
    fixed_pair = [3, 2]
    fixed_params[fixed_pair[0], fixed_pair[1]] = 0.5
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families,
                                  fixed_params=fixed_params)
    # Test Grid results
    impact.gridsearch(
        n_dep_param=10,
        n_input_sample=10,
        grid_type='lhs',
        random_state=0)
    check_dims(impact, dim)
   
    n_fixed_pair = 2
    fixed_pairs = [fixed_pair]
    for p in range(n_fixed_pair):
        # Set a family to independence
        condition = True
        while condition:
            i = np.random.randint(1, dim)
            j = np.random.randint(0, i)
            pair = [i, j] 
            if pair not in fixed_pairs:
                fixed_pairs.append(pair)
                condition = False
        fixed_params[pair[0], pair[1]] = 0.5
        impact.fixed_params = fixed_params
        pairs_lvl = get_tree_pairs(impact.vine_structure, 0)
        for ind_pair in fixed_pairs:
            assert ((ind_pair in pairs_lvl) or (list(reversed(ind_pair)) in pairs_lvl))
        check_dims(impact, dim)
        
        # Test Grid results
        impact.gridsearch(
            n_dep_param=10, 
            n_input_sample=10, 
            grid_type='lhs', 
            random_state=0)
        
        
def test_modification_bounds_tau():
    dim = 6
    families = np.tril(np.ones((dim, dim)), k=-1)
    bounds_tau = np.zeros((dim, dim))
    bounds_tau[:] = np.nan
    
    bounded_pair = [3, 2]
    bounds_tau[bounded_pair[0], bounded_pair[1]] = -0.5
    bounds_tau[bounded_pair[1], bounded_pair[0]] = 0.5
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families,
                                  bounds_tau=bounds_tau)
    
    # Test Grid results
    impact.gridsearch(
        n_dep_param=10, 
        n_input_sample=10, 
        grid_type='lhs', 
        random_state=0)
    
    check_dims(impact, dim)
   
    n_bounded = 2
    bounded_pairs = [bounded_pair]
    for p in range(n_bounded):
        # Set a family to independence
        condition = True
        while condition:
            i = np.random.randint(1, dim)
            j = np.random.randint(0, i)
            pair = [i, j] 
            if pair not in bounded_pairs:
                bounded_pairs.append(pair)
                condition = False
        bounds_tau[pair[0], pair[1]] = -0.5
        bounds_tau[pair[1], pair[0]] = 0.5
        
        impact.bounds_tau = bounds_tau
        check_dims(impact, dim)
        
        # Test Grid results
        impact.gridsearch(
            n_dep_param=10, 
            n_input_sample=10, 
            grid_type='lhs', 
            random_state=0)
        
        
def test_modification_multiple():
    dim = 6
    families = np.tril(np.ones((dim, dim)), k=-1)
    
    ind_pair = [1, 0]
    families[ind_pair[0], ind_pair[1]] = 0
    
    bounds_tau = np.zeros((dim, dim))
    bounds_tau[:] = np.nan
    
    bounded_pair = [3, 2]
    bounds_tau[bounded_pair[0], bounded_pair[1]] = -0.5
    bounds_tau[bounded_pair[1], bounded_pair[0]] = 0.5
    
    fixed_params = np.zeros((dim, dim))
    fixed_params[:] = np.nan
    
    fixed_pair = [2, 1]
    fixed_params[fixed_pair[0], fixed_pair[1]] = 0.5   
    
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families,
                                  bounds_tau=bounds_tau,
                                  fixed_params=fixed_params)
    
    # Test Grid results
    impact.gridsearch(
        n_dep_param=10, 
        n_input_sample=10, 
        grid_type='lhs', 
        random_state=0)
    
    check_dims(impact, dim)
   
    n_bounded = 2
    bounded_pairs = [bounded_pair]
    for p in range(n_bounded):
        # Set a family to independence
        condition = True
        while condition:
            i = np.random.randint(1, dim)
            j = np.random.randint(0, i)
            pair = [i, j] 
            if pair not in bounded_pairs:
                bounded_pairs.append(pair)
                condition = False
        bounds_tau[pair[0], pair[1]] = -0.5
        bounds_tau[pair[1], pair[0]] = 0.5
        
        impact.bounds_tau = bounds_tau
        check_dims(impact, dim)
        
        # Test Grid results
        impact.gridsearch(
            n_dep_param=10, 
            n_input_sample=10, 
            grid_type='lhs', 
            random_state=0)
        
        
def test_iterative():
    dim = 6
    alpha = 0.05
    families = np.tril(np.ones((dim, dim)), k=-1)
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families)
    algorithm_parameters = {
        "n_input_sample": 1000,
        "n_dep_param_init": None,
        "max_n_pairs": 3,
        "grid_type": 'vertices',
        "q_func": quantile_func(alpha),
        "n_add_pairs": 1,
        "n_remove_pairs": 0,
        "adapt_vine_structure": True,
        "with_bootstrap": False,
        "verbose": True,
        "iterative_save": False,
        "iterative_load": False,
        "load_input_samples": False,
        "keep_input_samples": False
        }
    
    iterative_vine_minimize(estimate_object=impact, **algorithm_parameters)

        
def get_tree_pairs(structure, lvl):
    """
    """
    dim = structure.shape[0]
    pairs = []
    for l in range(dim-1-lvl):
        i = structure[l, l] - 1
        j = structure[-1-lvl, l] - 1
        pairs.append([i, j])
    return pairs

def test_bidim_additive_gaussian_gridsearch():
    dim = 2
    n_params = 50
    n_input_sample = 10000

    for alpha, threshold in zip(QUANTILES_PROB, PROB_THRESHOLDS):
        for grid in GRIDS:
            # Only Gaussian families
            families = np.tril(np.ones((dim, dim)), k=1)

            impact = ConservativeEstimate(model_func=func_sum,
                                          margins=[ot.Normal()]*dim,
                                          families=families)

            # Grid results
            grid_results = impact.gridsearch(
                n_dep_param=n_params, 
                n_input_sample=n_input_sample, 
                grid_type=grid, 
                random_state=0)

            # Theorical results
            true_quantiles = np.zeros((grid_results.n_params, ))
            true_probabilities = np.zeros((grid_results.n_params, ))
            for k in range(grid_results.n_params):
                sigma = dep_params_list_to_matrix(grid_results.dep_params[k, :], dim)
                true_quantiles[k] = true_additive_gaussian_quantile(alpha, dim, sigma)
                true_probabilities[k] = true_additive_gaussian_probability(threshold, dim, sigma)

            # Quantile results
            grid_results.q_func = quantile_func(alpha)
            empirical_quantiles = grid_results.quantities
            
            assert_allclose(empirical_quantiles, true_quantiles, rtol=1e-01,
                            err_msg="Quantile estimation failed for alpha={0}, dim={1}, grid: {2}"\
                            .format(alpha, dim, grid))

            # Probability results
            grid_results.q_func = proba_func(threshold)
            empirical_probabilities = 1. - grid_results.quantities

            assert_allclose(empirical_probabilities, true_probabilities, rtol=1e-01,
                            err_msg="Probability estimation failed for threshold = {0}, dim = {1}, grid: {2}"\
                            .format(alpha, dim, grid))
                        
def test_independence():
    n_input_sample = 10000
    for alpha, threshold in zip(QUANTILES_PROB, PROB_THRESHOLDS):
        for dim in DIMENSIONS:
            # Only Gaussian families
            families = np.tril(np.ones((dim, dim)), k=1)
    
            impact = ConservativeEstimate(model_func=func_sum,
                                          margins=[ot.Normal()]*dim,
                                          families=families)
    
            indep_result = impact.independence(n_input_sample=n_input_sample)
    
            sigma = dep_params_list_to_matrix(0., dim)
            true_quantile = true_additive_gaussian_quantile(alpha, dim, sigma)
            true_probability = true_additive_gaussian_probability(threshold, dim, sigma)
    
            # Quantile results
            indep_result.q_func = quantile_func(alpha)
            empirical_quantile = indep_result.quantity
            
            assert_allclose(empirical_quantile, true_quantile, rtol=1e-01,
                            err_msg="Quantile estimation failed for alpha={0}, dim={1}"\
                            .format(alpha, dim))
    
            # Probability results
            indep_result.q_func = proba_func(threshold)
            empirical_probability = 1. - indep_result.quantity
    
            assert_allclose(empirical_probability, true_probability, rtol=1e-01,
                            err_msg="Probability estimation failed for threshold = {0}, dim = {1}"\
                            .format(alpha, dim))
            
        
def test_vines():
    n_input_sample = 1000
    dim = 3
    n_params = 200
    grid_type = 'lhs'
    families = np.tril(np.ones((dim, dim)), k=1)
    
    impact = ConservativeEstimate(model_func=func_sum,
                                  margins=[ot.Normal()]*dim,
                                  families=families)

    # Grid results
    grid_results = impact.gridsearch(
        n_dep_param=n_params, 
        n_input_sample=n_input_sample, 
        grid_type=grid_type, 
        random_state=0)