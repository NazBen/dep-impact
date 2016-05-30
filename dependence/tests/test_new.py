import numpy as np
import openturns as ot

from dependence import ImpactOfDependence

def add_function(x):
    """
    """
    return x.sum(axis=1)

# Creation of the random variable
dim = 2
marginals = [ot.Normal()] * dim
n_dep_param = 100
alpha = 0.05
threshold = 2.
n_sample_per_param = 10000
fixed_grid = True
copulas = ['NormalCopula', 'ClaytonCopula']
copulas = ['ClaytonCopula']
for copula_name in copulas:
    impact = ImpactOfDependence(add_function, marginals, copula_name)
    impact.run(n_dep_param, n_input_sample=n_sample_per_param, fixed_grid=fixed_grid)

    quantile_result = impact.compute_quantiles(alpha)
    proba_result = impact.compute_probability(threshold)
    quantile_result.draw(savefig="./dependence/tests/quant" + copula_name)
    proba_result.draw(savefig="./dependence/tests/proba" + copula_name)