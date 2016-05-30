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

impact = ImpactOfDependence(add_function, marginals)
impact.run(n_dep_param, n_input_sample=10000)

alpha = 0.05
threshold = 0.
quantile_result = impact.compute_quantiles(alpha)
proba_result = impact.compute_probability(0.)
print quantile_result.quantity

quantile_result.draw(savefig=True)