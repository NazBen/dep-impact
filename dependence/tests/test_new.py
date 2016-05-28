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
impact.run(n_dep_param, n_input_sample=10)

impact.compute_quantity("quantile", (0.05, 1))

print impact._output_quantity