# -*- coding: utf-8 -*-
import numpy as np
import openturns as ot
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import nlopt
import random
sys.path.append("/netdata/D58174/gdrive/These/Scripts/library/randomForest")
from quantForest import QuantileForest
np.random.seed(0)

COPULA_LIST = ["Normal", "Clayton", "Gumbel"]


class Conversion:
    class NormalCopula:
        @staticmethod
        def fromPearsonToKendall(rho):
            return 2. / np.pi * np.arcsin(rho)

        @staticmethod
        def fromKendallToPearson(tau):
            return np.sin(np.pi / 2. * tau)

    class ClaytonCopula:
        @staticmethod
        def fromPearsonToKendall(rho):
            return rho / (rho + 2.)

        @staticmethod
        def fromKendallToPearson(tau):
            return 2. * tau / (1. - tau)


def get_random_rho(size, dim, rho_min=-1., rho_max=1.):
    if dim == 1:
        list_rho = np.asarray(ot.Uniform(rho_min, rho_max).getSample(size))
    else:  # TODO : make it available in d dim
        list_rho = np.zeros((size, dim))
        for i in range(size):
            rho1 = random.uniform(rho_min, rho_max)
            rho2 = random.uniform(rho_min, rho_max)
            l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
            u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
            rho3 = random.uniform(l_bound, u_bound)
            list_rho[i, :] = [rho1, rho2, rho3]

    return list_rho


def get_list_rho3(rho1, rho2, n):
    """
    """
    l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
    u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
    list_rho3 = np.linspace(l_bound, u_bound, n+1, endpoint=False)[1:]
    return list_rho3


def get_grid_rho(n_sample, dim=3, rho_min=-1., rho_max=1., all_sample=True):
    """
    """
    if dim == 1:
        return np.linspace(rho_min, rho_max, n_sample + 1,
                           endpoint=False)[1:]
    else:
        if all_sample:
            n = int(np.floor(n_sample**(1./dim)))
        else:
            n = n_sample
        v_rho_1 = [np.linspace(rho_min, rho_max, n + 1,
                               endpoint=False)[1:]]*(dim - 1)
        grid_rho_1 = np.meshgrid(*v_rho_1)
        list_rho_1 = np.vstack(grid_rho_1).reshape(dim - 1, -1).T

        list_rho = np.zeros((n**dim, dim))
        for i, rho_1 in enumerate(list_rho_1):
            a1 = rho_1[0]
            a2 = rho_1[1]
            list_rho_2 = get_list_rho3(a1, a2, n)
            for j, rho_2 in enumerate(list_rho_2):
                tmp = rho_1.tolist()
                tmp.append(rho_2)
                list_rho[n*i+j, :] = tmp

        return list_rho


class ImpactOfDependence:
    def __init__(self, model_function=None, variable=None):
        """
        """
        if model_function:
            self.setModelFunction(model_function)
        if variable:
            self.setInputVariables(variable)

        self._rhoMin, self._rhoMax = -1., 1.
        self._tauMin, self._tauMax = -1., 1.

    @classmethod
    def from_data(cls, data_sample, dim):
        """

        """
        obj = cls()
        corr_dim = dim*(dim-1)/2
        obj._corr_dim = corr_dim
        obj._list_param = data_sample[:, :corr_dim]
        obj._n_sample = obj._list_param.shape[0]
        obj._input_sample = data_sample[:, corr_dim:corr_dim+dim]
        obj._output_sample = data_sample[:, -1:]
        obj._params = pd.DataFrame(obj._list_param).drop_duplicates().values
        obj._n_param = obj._params.shape[0]
        obj._n_obs_sample = obj._n_sample / obj._n_param
        return obj

    def run(self, n_sample, fixed_grid=True, dep_meas="KendallTau",
            n_obs_sample=1):
        """
        """
        self._create_sample(n_sample, fixed_grid, dep_meas, n_obs_sample)
        self._output_sample = self._modelFunction(self._input_sample)

    def _create_sample(self, n_sample, fixed_grid, dep_measure, n_obs_sample):
        """
            Create the observations for differents dependence parameters
        """
        dim = self._input_dim  # Dimension of input variables
        corr_dim = self._corr_dim  # Dimension of correlation parameters

        # We need the same number of observation for each sample
        n_sample -= n_sample % n_obs_sample  # We take off the rest
        n_param = n_sample / n_obs_sample  # Number of correlation parameters

        # We convert the dependence measure to the copula parameter
        if dep_measure == "KendallTau":
            # TODO : Work on it
            tauMin, tauMax = self._tauMin, self._tauMax

            if fixed_grid:  # Fixed grid
                grid_tau = get_grid_tau(n_param, corr_dim, tauMin, tauMax)
                listTau = np.vstack(grid_tau).reshape(corr_dim, -1).T
            else:  # Random grid
                tmp = [ot.Uniform(tauMin, tauMax)]*corr_dim
                tmpVar = ot.ComposedDistribution(tmp)
                listTau = np.array(tmpVar.getSample(n_param)).ravel()

            if self._copula_name == "NormalCopula":
                list_param = Conversion.\
                    NormalCopula.fromKendallToPearson(listTau)

            elif self._copula_name == "InverseClaytonCopula":
                list_param = Conversion.\
                    ClaytonCopula.fromKendallToPearson(listTau)

        elif dep_measure == "PearsonRho":
            rho_min, rho_max = self._rhoMin, self._rhoMax

            if fixed_grid:  # Fixed grid
                list_rho = get_grid_rho(n_param, corr_dim, rho_min, rho_max)
                # Once again, we change the number of param to have a grid
                # with the same number of parameters in each dim
                n_param = list_rho.shape[0]
                # TODO : The total number of param may certainly be different
                # than the initial one. Find a way to adapt it...
                n_sample = n_param * n_obs_sample
            else:  # Random grid
                list_rho = get_random_rho(n_param, corr_dim, rho_min, rho_max)

            if self._copula_name == "NormalCopula":
                list_param = list_rho

            elif self._copula_name == "InverseClaytonCopula":
                # TODO : check that shit
                list_param = Conversion.\
                    ClaytonCopula.fromPearsonToKendall(list_rho)
        else:
            raise Exception("Other dependence measures not yet implemented")

        self._n_param = n_param
        self._n_sample = n_sample
        self._n_obs_sample = n_obs_sample
        self._params = list_param
        self._input_sample = np.empty((n_sample, dim))  # Input sample
        self._list_param = np.empty((n_sample, corr_dim))  # Input Corr sample

        # We loop for each copula param and create observations for each
        for i, param in enumerate(list_param):  # For each copula parameter
            tmp = self._get_sample(param, n_obs_sample)

            # TODO : find a way to create a "real" class InverseClayton...
            if self._copula_name == "InverseClaytonCopula":
                tmp[:, 0] = -tmp[:, 0]

            # We save the input sample
            self._input_sample[n_obs_sample*i:n_obs_sample*(i+1), :] = tmp

            # As well for the parameters (for dependence measure)
            self._list_param[n_obs_sample*i:n_obs_sample*(i+1), :] = \
                list_param[i]

    def buildForest(self, n_jobs=8):
        """
        """
        self._quantForest = QuantileForest(self._list_param,
                                           self._output_sample, n_jobs=n_jobs)

    def compute_quantiles(self, alpha, estimation_method, n_sample=None):
        """
        """
        if estimation_method == 1:
            out_sample = self._output_sample.reshape((self._n_param,
                                                      self._n_obs_sample))
            self._quantiles = np.percentile(out_sample, alpha*100., axis=1)
        elif estimation_method == 2:
            self.buildForest()
        else:
            raise Exception("Not done yet")

        self._estimation_method = estimation_method

    def saveInputData(self, path=".", fname="inputSampleCop", ftype=".csv"):
        """
        """
        full_fname = path + '/' + fname + ftype
        np.savetxt(full_fname, self._list_param)

    def save_all_data(self, path=".", fname="full_data", ftype=".csv"):
        """

        """
        # TODO : think about it
        full_fname = path + '/' + fname + ftype
        out = np.c_[self._list_param, self._input_sample, self._output_sample]
        np.savetxt(full_fname, out)

    def _get_sample(self, param, n_obs):
        """
        must be the copula parameter
        """
        # TODO: change it when the new ot released is out
        if self._copula_name == "NormalCopula":
            if self._corr_dim == 1:
                self._corr_matrix[0, 1] = param
            elif self._corr_dim == 3:
                self._corr_matrix[0, 1] = param[0]
                self._corr_matrix[0, 2] = param[1]
                self._corr_matrix[1, 2] = param[2]
            self._copula = ot.NormalCopula(self._corr_matrix)
        else:
            self._copula.setParametersCollection([param])

        self._input_variables.setCopula(self._copula)
        return np.array(self._input_variables.getSample(n_obs))

    def _empQuantileFunction(self, sample, alpha):
        """
        TODO : think about any other method to compute quantiles
        """
        return np.percentile(sample, alpha*100)

    def _computeEmpiricalQuantile(self, param, alpha, num_obs):
        """
        must be copula parameter
        """
        input_sample = self._get_sample(param, num_obs)
        output_sample = self._modelFunction(input_sample)
        return self._empQuantileFunction(output_sample, alpha)

    def draw_quantiles(self, alpha, n_per_dim=10, dep_meas="KendallTau",
                       with_sample=False, figsize=(10, 6), saveFig=False,
                       color_map="jet"):
        """

        """
        assert self._corr_dim in [1, 3], "Cannot draw quantiles"
        if dep_meas == "KendallTau":
            param_name = "\\tau"
            param_min, param_max = self._tauMin, self._tauMax
            grid_func = get_grid_tau

        elif dep_meas == "PearsonRho":
            param_name = "\\rho"
            param_min, param_max = self._rhoMin, self._rhoMax
            grid_func = get_grid_rho
        else:
            raise("Undefined param")

        if self._estimation_method == 1:
            listParam = self._params
            quantiles = self._quantiles

        elif self._estimation_method == 2:
            listParam = grid_func(n_per_dim, self._corr_dim, param_min,
                                  param_max, all_sample=False)
            quantiles = self._quantForest.computeQuantile(listParam, alpha)

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if self._corr_dim == 1:  # If correlation dimension is 1
            ax = fig.add_subplot(111)  # Creat the ax object
            ax.plot(listParam, quantiles, 'b',
                    label="Conditional %.2f %% quantiles" % (alpha),
                    linewidth=2)
            if with_sample:
                ax.plot(self._list_param, self._output_sample, 'k.')
            ax.set_xlabel("$%s_{12}$" % (param_name), fontsize=14)
            ax.set_ylabel("Quantile")
            ax.legend(loc="best")
        elif self._corr_dim == 3:  # If correlation dimension is 3
            color_scale = quantiles
            cm = plt.get_cmap(color_map)
            cNorm = matplotlib.colors.Normalize(vmin=min(color_scale),
                                                vmax=max(color_scale))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            x, y, z = listParam[:, 0], listParam[:, 1], listParam[:, 2]

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=scalarMap.to_rgba(color_scale), s=40)
            scalarMap.set_array(color_scale)
            fig.colorbar(scalarMap)

            ax.set_xlabel("$%s_{12}$" % (param_name), fontsize=14)
            ax.set_ylabel("$%s_{13}$" % (param_name), fontsize=14)
            ax.set_zlabel("$%s_{23}$" % (param_name), fontsize=14)

        title = r"$\hat q_{\alpha, n}(%s)$ with $\alpha=%.1e$, $n = %d$" % \
            (param_name, alpha, self._n_sample)
        ax.set_title(title, fontsize=18)
        ax.axis("tight")
        fig.tight_layout()
        if saveFig:
            if type(saveFig) is str:
                fname = saveFig + '/'
            else:
                fname = "./"
            fname += "condQuantiles"
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")

    def _nloptObjFunc(self, x, grad):
        """
        """
        return -self._quantFunc(x)

    def optimize(self, method, alpha, x0, maxEvaluation=100000,
                 n_obs_sample=None, tol=1.E-8):
        """

        """
        self._quantFunc = self.getQuantFunc(method, alpha, maxEvaluation,
                                            n_obs_sample)

        dim = self._input_dim
        rhoDim = dim*(dim-1)/2
        nloptMethod = nlopt.GN_DIRECT  # Object of the method
        opt = nlopt.opt(nloptMethod, rhoDim)
        opt.set_min_objective(self._nloptObjFunc)
        opt.set_stopval(tol)
        opt.set_ftol_rel(tol)
        opt.set_ftol_abs(tol)
        opt.set_xtol_rel(tol)
        opt.set_xtol_abs(tol)
        opt.set_lower_bounds([-.9])
        opt.set_upper_bounds([0.9])
        opt.set_maxeval(maxEvaluation)
        self._optimalPoint = opt.optimize(x0)
        return self._optimalPoint

# =============================================================================
# Setters
# =============================================================================
    def setModelFunction(self, modelFunction):
        """
        """
        assert callable(modelFunction), "The model function is not callable"
        self._modelFunction = modelFunction

    def setInputVariables(self, variables):
        """
        """
        self._input_dim = variables.getDimension()
        self._copula = variables.getCopula()
        self._copula_name = self._copula.getName()
        self._input_variables = variables
        self._corr_dim = self._input_dim*(self._input_dim-1)/2
        if self._copula_name == "NormalCopula":
            self._corr_matrix = ot.CorrelationMatrix(self._input_dim)


# =============================================================================
#
# =============================================================================

if __name__ == "__main__":
    def add_function(x):
        return x.sum(axis=1)

    def levy_function(x, phase=1.):
        x = np.asarray(x)

        w = 1 + (x - 1.)/4.

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
        output = np.sin(np.pi*w1)**2
        output += ((wi-1.)**2*(1.+10.*np.sin(np.pi*wi+1.)**2)).sum(axis=ax)
        output += (wd - 1.)**2 * (1. + np.sin(2*np.pi*wd)**2)
        return output

    # Creation of the random variable
    dim = 2  # Input dimension
    copula_name = "NormalCopula"  # Name of the used copula
    marginals = [ot.Normal()]*dim  # Marginals

    # TODO : find a way to create a real InverseClaytonCopula
    if copula_name == "NormalCopula":
        copula = ot.NormalCopula(dim)
    elif copula_name == "InverseClaytonCopula":
        copula = ot.ClaytonCopula(dim)
        copula.setName(copula_name)

    # Variable object
    var = ot.ComposedDistribution(marginals, copula)

    # Parameters
    n_rho_dim = 10  # Number of correlation values per dimension
    n_obs_sample = 10  # Observation per rho
    rho_dim = dim * (dim-1)/2
    sample_size = (n_rho_dim**rho_dim + 1)*n_obs_sample
#    sample_size = 100000  # Number of sample
    alpha = 0.01  # Quantile probability
    fixed_grid = True  # Fixed design sampling
    estimation_method = 2  # Used method
    measure = "PearsonRho"

    impact = ImpactOfDependence(levy_function, var)
    impact.run(sample_size, fixed_grid, n_obs_sample=n_obs_sample,
               dep_meas=measure)

    impact.save_all_data()
    del impact

    data = np.genfromtxt("full_data.csv")
    impact_load = ImpactOfDependence.from_data(data, dim)
    impact_load.compute_quantiles(alpha, 1)
    impact_load.draw_quantiles(alpha, dep_meas=measure)

    impact_load.compute_quantiles(alpha, 2)
    impact_load.draw_quantiles(alpha, dep_meas=measure)
