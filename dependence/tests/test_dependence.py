from scipy.special import erf, erfinv
from numpy.testing import assert_allclose
import openturns as ot
import numpy as np
from itertools import combinations

from dependence import ImpactOfDependence

def add_function(x):
    """
    """
    return x.sum(axis=1)

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
    tmp = np.hstack(sigma[i][:i] for i in xrange(sigma.shape[0]))
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
    tmp = np.hstack(sigma[i][:i] for i in xrange(sigma.shape[0]))
    var_y = (const**2 * sigma.diagonal()).sum() + (2*tmp).sum()
    sigma_y = np.sqrt(var_y)
    return 0.5 * (1. + erf(x / (sigma_y * np.sqrt(2.))))

ALPHAS = [0.05, 0.01]
THRESHOLDS = [1., 2.]
DIMENSIONS = range(2, 4)
LIST_COPULA = ["NormalCopula", "ClaytonCopula"]
LIST_MEASURES = ["PearsonRho", "KendallTau"]

def dep_params_list_to_matrix(params, dim):
    sigma = np.ones((dim, dim))
    k = 0
    for i in range(1, dim):
        for j in range(i):
            sigma[i, j] = params[k]
            sigma[j, i] = params[k]
            k += 1
    return sigma

def test_aaa():
    for alpha, threshold in zip(ALPHAS, THRESHOLDS):
        for dim in DIMENSIONS:
            impact = ImpactOfDependence(add_function, [ot.Normal()]*dim, np.ones((dim, dim)), copula_type='normal')
            impact.run(n_dep_param=50, n_input_sample=10000, grid='rand', seed=0)

            empirical_quantiles = impact.compute_quantiles(alpha).quantity.ravel()
            empirical_probabilities = impact.compute_probability(threshold, operator='<').quantity.ravel()
            true_quantile = np.zeros((impact.n_dep_params_, ))
            true_probability = np.zeros((impact.n_dep_params_, ))
            for k in range(impact.n_dep_params_):
                sigma = dep_params_list_to_matrix(impact.params_[k, :], dim)
                true_quantile[k] = true_additive_gaussian_quantile(alpha, dim, sigma)
                true_probability[k] = true_additive_gaussian_probability(threshold, dim, sigma)

            assert_allclose(empirical_quantiles, true_quantile, rtol=1e-01,
                            err_msg="Failed with alpha = {0}, dim = {1}"\
                            .format(alpha, dim))
            assert_allclose(empirical_probabilities, true_probability, rtol=1e-01,
                            err_msg="Failed with threshold = {0}, dim = {1}"\
                            .format(threshold, dim))

def test_additive_gaussian_emprical_estimation():
    """
    Check if the estimated output quantiles and probabilities of an additive 
    gaussian model are correct.
    """
    for alpha, threshold in zip(ALPHAS, THRESHOLDS):
        for dim in DIMENSIONS:
            impact = ImpactOfDependence(add_function, [ot.Normal()] * dim, 
                                        copula_name='NormalCopula')
            impact.run(n_dep_param=100, n_input_sample=100000, 
                       fixed_grid=False, dep_measure='PearsonRho', seed=0)
        
            true_quant = true_quantile(alpha, dim, impact._params)
            quant_res = impact.compute_quantiles(alpha)
            assert_allclose(quant_res.quantity, true_quant, rtol=1e-01,
                            err_msg="Failed with alpha = {0}, dim = {1}"\
                            .format(alpha, dim))
                            
            true_proba = true_probability(threshold, dim, impact._params)
            proba_res = impact.compute_probability(threshold, operator='lower')
            assert_allclose(proba_res.quantity, true_proba, rtol=1e-01,
                            err_msg="Failed with threshold = {0}, dim = {1}"\
                            .format(threshold, dim))

def test_draw():
    dim = 2
    copula_name = 'ClaytonCopula'
    measure = 'KendallTau'
    alpha = 0.05
    threshold = 2.
    impact = ImpactOfDependence(add_function, [ot.Normal()] * dim, 
                                copula_name=copula_name)
    impact.run(n_dep_param=100, n_input_sample=10000, fixed_grid=True, 
               dep_measure=measure, seed=0)

    quant_result = impact.compute_quantiles(alpha)
    quant_result.draw('KendallTau')
    
    proba_result = impact.compute_probability(threshold)
    proba_result.draw('KendallTau')

def test_fixed_grid():
    dim = 2
    alpha = 0.05
    threshold = 2.
    for copula_name in LIST_COPULA:
        for measure in LIST_MEASURES:
            if copula_name == "ClaytonCopula" and measure == "PearsonRho":
                pass
            else:
                impact = ImpactOfDependence(add_function, [ot.Normal()] * dim, 
                                            copula_name=copula_name)
                impact.run(n_dep_param=100, n_input_sample=10000, fixed_grid=True, 
                           dep_measure=measure, seed=0)

                quant_result = impact.compute_quantiles(alpha)
                quant_result.draw(measure)
                proba_result = impact.compute_probability(threshold)
                proba_result.draw(measure)

def test_dim_measure():
    alpha = 0.05
    threshold = 2.
    copula_name = "NormalCopula"
    for dim in DIMENSIONS:
        for measure in LIST_MEASURES:
            impact = ImpactOfDependence(add_function, [ot.Normal()] * dim, 
                                        copula_name=copula_name)
            impact.run(n_dep_param=100, n_input_sample=10000, fixed_grid=False, 
                       dep_measure=measure, seed=0)

            true_quant = true_quantile(alpha, dim, impact._params)
            quant_res = impact.compute_quantiles(alpha)
            assert_allclose(quant_res.quantity, true_quant, rtol=1e-01,
                            err_msg="Failed with alpha = {0}, dim = {1}"\
                            .format(alpha, dim))

            true_proba = true_probability(threshold, dim, impact._params)
            proba_res = impact.compute_probability(threshold, operator='lower')
            assert_allclose(proba_res.quantity, true_proba, rtol=1e-01,
                            err_msg="Failed with threshold = {0}, dim = {1}"\
                            .format(threshold, dim))

def test_custom_corr_vars():
    dim = 4
    alpha = 0.05
    threshold = 2.
    copula_name = "NormalCopula"

    # We increases the number of correlated variables
    all_corr_vars = []
    for i, corr in enumerate(list(combinations(range(dim), 2))):
        if i > 0:
            tmp = all_corr_vars[i - 1] + [corr]
            all_corr_vars.append(tmp)
        else:
            all_corr_vars.append([corr])

    for corr_vars in all_corr_vars:
        for measure in LIST_MEASURES:
            impact = ImpactOfDependence(add_function, [ot.Normal()] * dim,
                                        copula_name=copula_name)
            impact.set_correlated_variables(corr_vars)
            assert impact._n_corr_vars == len(corr_vars), "Not good"
            
            impact.run(n_dep_param=100, n_input_sample=10000, fixed_grid=False, 
                       dep_measure=measure, seed=0)
            
            true_quant = true_quantile(alpha, dim, impact._params)
            quant_res = impact.compute_quantiles(alpha)
            
            assert_allclose(quant_res.quantity, true_quant, rtol=1e-01,
                            err_msg="Failed with alpha = {0}, dim = {1}"\
                            .format(alpha, dim))

            true_proba = true_probability(threshold, dim, impact._params)
            proba_res = impact.compute_probability(threshold, operator='lower')
            assert_allclose(proba_res.quantity, true_proba, rtol=1e-01,
                            err_msg="Failed with threshold = {0}, dim = {1}"\
                            .format(threshold, dim))

def test_saving_loading():
    dim = 3
    alpha = 0.05
    threshold = 2.
    measure = "KendallTau"
    margins = [ot.Weibull(), ot.Normal(), ot.Normal()]

    families = np.zeros((dim, dim), dtype=int)
    families[1, 0] = 1
    families[2, 0] = 0
    families[2, 1] = 1

    impact = ImpactOfDependence(model_func=add_function, margins=margins, families=families)
    impact.run(n_dep_param=10, n_input_sample=500, fixed_grid=True, 
               dep_measure=measure, seed=0)
    impact.save_data()
    impact2 = ImpactOfDependence.from_structured_data()

    np.testing.assert_allclose(impact2._output_sample, impact._output_sample)


def test_last():
    dim = 3
    alpha = 0.05
    threshold = 2.
    measure = "KendallTau"
    margins = [ot.Weibull(), ot.Normal(), ot.Normal()]

    families = np.zeros((dim, dim), dtype=int)
    families[1, 0] = 0
    families[2, 0] = 0
    families[2, 1] = 26

    impact = ImpactOfDependence(model_func=add_function, margins=margins, families=families)

    impact.run(n_dep_param=10, n_input_sample=10000, fixed_grid=True, 
                dep_measure=measure, seed=0)

    quant_res = impact.compute_quantiles(alpha)
    id_min = quant_res.quantity.argmax()
    impact.draw_matrix_plot(id_min, copula_space=True)

    
def test_bounds():
    dim = 4
    alpha = 0.05
    threshold = 2.
    measure = "KendallTau"
    margins = [ot.Normal()]*dim
    families = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(i):
            families[i, j] = 1
    impact = ImpactOfDependence(model_func=add_function, margins=margins, families=families)

    impact.minmax_run(10000, eps=1.E-4)
    quant_res = impact.compute_quantiles(alpha)

    id_min = quant_res.quantity.argmin()
    
    
def test_hdf():
    dim = 3
    alpha = 0.05
    threshold = 2.
    measure = "KendallTau"
    margins = [ot.Weibull(), ot.Normal(), ot.Normal()]

    families = np.zeros((dim, dim), dtype=int)
    families[1, 0] = 1
    families[2, 0] = 0
    families[2, 1] = 0
  
    impact = ImpactOfDependence(model_func=add_function, margins=margins, families=families)

    impact.run(n_dep_param=20, n_input_sample=100, grid='fixed')

    filename = impact.save_data_hdf()
    impact_load = ImpactOfDependence.from_hdf(filename)
    impact.compute_quantiles(alpha).draw()
    impact_load.compute_quantiles(alpha).draw()

# TODO: add a test for the saving and loading of a DOE sample

def test_constraints():
    dim = 4
    n = 100
    K = 10
    margins = [ot.Normal()]*dim
    families = np.zeros((dim, dim), dtype=int)
    families[1, 0] = 1
    families[2, 0] = 4
    families[2, 1] = 1

    fixed_params = np.zeros((dim, dim), dtype=float)
    fixed_params[1, 0] = None
    fixed_params[2, 0] = 2.27
    fixed_params[2, 1] = None

    bounds_tau = np.zeros((dim, dim), dtype=float)
    bounds_tau[:] = None
    bounds_tau[1, 0] = 0.
    bounds_tau[0, 1] = None
    bounds_tau[2, 1] = None
    bounds_tau[1, 2] = 0.
    alpha = 0.1
  
    impact = ImpactOfDependence(model_func=add_function, 
                                margins=margins, 
                                families=families,
                                fixed_params=fixed_params,
                                bounds_tau=bounds_tau)
    
    impact.run(n_dep_param=K, n_input_sample=n, grid='fixed', seed=0)
    quantile = impact.compute_quantiles(alpha)

    impact.draw_matrix_plot()
    print quantile.cond_params
    print quantile.quantity

if __name__ == '__main__':
    test_aaa()

