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
    
    
def true_quantile(alpha, dim, params):
    sigma = np.sqrt(dim + 2*params.sum(axis=1))
    return sigma*np.sqrt(2.) * erfinv(2*alpha - 1.)
        
    
def true_probability(x, dim, params):
    sigma = np.sqrt(dim + 2*params.sum(axis=1))    
    return 0.5*(1. + erf(x / (sigma*np.sqrt(2.))))
    

ALPHAS = [0.05, 0.01]
THRESHOLDS = [1., 2.]
DIMENSIONS = range(2, 5)
LIST_COPULA = ["NormalCopula", "ClaytonCopula"]
LIST_MEASURES = ["PearsonRho", "KendallTau"]

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
            tmp = all_corr_vars[i-1] + [corr]
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


#proba_result = impact.compute_probability(threshold)
#proba_result.draw(measure)