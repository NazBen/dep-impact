import numpy as np
import openturns as ot
from dependence import ImpactOfDependence

def add_function(x):
    """
    """
    return x.sum(axis=1)

# Creation of the random variable
dim = 2
copula_name = "InverseClaytonCopula"
marginals = [ot.Normal()] * dim

if copula_name == "NormalCopula":
    copula = ot.NormalCopula(dim)
elif copula_name == "InverseClaytonCopula":
    copula = ot.ClaytonCopula(dim)
    copula.setName(copula_name)
# Variable object
var = ot.ComposedDistribution(marginals, copula)

# Set the correlated variables
corr_vars = [[0, 1]]
n_corr_vars = len(corr_vars)

# Parameters
n_rho_dim = 50  # Number of correlation values per dimension
n_obs_sample = 10000  # Observation per rho
rho_dim = dim * (dim - 1) / 2
sample_size = (n_rho_dim ** rho_dim + 1) * n_obs_sample
alpha = 0.05  # Quantile probability

fixed_grid = False  # Fixed design sampling
estimation_method = 1  # Used method
dep_measure = "PearsonRho"
n_output = 1
out_names = ["A", "B"]
input_names = ["H", "L", "K"]
out_names = []
input_names = []

impact = ImpactOfDependence(add_function, var, corr_vars)

impact.run(sample_size, fixed_grid, n_obs_sample=n_obs_sample,
            dep_meas=dep_measure, from_init_sample=False)

threshold = 1.
c_level = 0.01
impact.compute_quantity("probability", (threshold, c_level))
impact.draw_quantity("Probability", savefig=True)