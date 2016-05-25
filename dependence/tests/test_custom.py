import numpy as np
import openturns as ot
from dependence import ImpactOfDependence

def add_function(x):
    return x.sum(axis=1)

def levy_function(x, phase=1.):
    x = np.asarray(x)

    w = 1 + (x - 1.) / 4.

    if x.shape[0] == x.size:
        w1 = w[0]
        wi = w[:-1]
        wd = w[-1]
        ax = 0
    else:
        w1 = w[:, 0]
        wi = w[:, :-1]
        wd = w[:, -1]
        ax = 1

    w1 += phase  # Modification of the function
    output = np.sin(np.pi * w1) ** 2
    output += ((wi - 1.)**2 * (1. + 10. *
                                np.sin(np.pi * wi + 1.)**2)).sum(axis=ax)
    output += (wd - 1.) ** 2 * (1. + np.sin(2 * np.pi * wd) ** 2)
    return output

# Creation of the random variable
dim = 3  # Input dimension
copula_name = "NormalCopula"  # Name of the used copula
marginals = [ot.Normal()] * dim  # Marginals

# TODO : find a way to create a real InverseClaytonCopula
if copula_name == "NormalCopula":
    copula = ot.NormalCopula(dim)
elif copula_name == "InverseClaytonCopula":
    copula = ot.ClaytonCopula(dim)
    copula.setName(copula_name)

# Variable object
var = ot.ComposedDistribution(marginals, copula)

# Set the correlated variables
corr_vars = [[0, 2]]
corr_vars = [[0, 1], [0, 2]]
n_corr_vars = len(corr_vars)

# Parameters
n_rho_dim = 20  # Number of correlation values per dimension
n_obs_sample = 1000  # Observation per rho
rho_dim = dim * (dim - 1) / 2
sample_size = (n_rho_dim ** n_corr_vars + 1) * n_obs_sample
#    sample_size = 100000 # Number of sample
alpha = 0.05  # Quantile probability

fixed_grid = False  # Fixed design sampling
estimation_method = 1  # Used method
measure = "PearsonRho"
n_output = 1
out_names = ["A", "B"]
input_names = ["H", "L", "K"]
out_names = []
input_names = []

def used_function(x):
    out = levy_function(x)
    if n_output > 1:
        output = np.asarray([out * (i + 1) for i in range(n_output)]).T
    else:
        output = out
    return output

impact = ImpactOfDependence(used_function, var, corr_vars)

impact.run(sample_size, fixed_grid, n_obs_sample=n_obs_sample,
            dep_meas=measure, from_init_sample=False)

threshold = 5.
c_level = 0.01
impact.compute_quantity("probability", (threshold, c_level))
impact.draw_quantity("Probability")
#    impact.compute_quantiles(alpha, estimation_method)
#    impact.compute_probability(2.)
#    print impact._probability
#    print impact._probability_interval

#    id_min_quant = impact._quantiles.min()
#rho_in = impact._params[id_min_quant]
#impact.draw_design_space(rho_in, display_quantile_value=alpha)
#    impact.draw_quantiles(alpha, estimation_method, n_rho_dim,
#                          dep_meas=measure, saveFig=False)

#    impact.draw_design_space(rho_in, input_names=input_names,
#                             output_name=out_names[0])
#    impact.save_all_data()
#    impact.save_structured_all_data(input_names, out_names)
#    del impact
