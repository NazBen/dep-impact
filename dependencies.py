# -*- coding: utf-8 -*-
import numpy as np
import openturns as ot
import sys
import matplotlib.pyplot as plt
import nlopt
import random
from randomForest.quantForest import QuantileForest
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


def get_a3(a1, a2, n):
    """
    """
    l_bound = a1*a2 - np.sqrt((1-a1**2)*(1-a2**2))
    u_bound = a1*a2 + np.sqrt((1-a1**2)*(1-a2**2))
    list_a3 = np.linspace(l_bound, u_bound, n+1, endpoint=False)[1:]
    return list_a3


def get_grid_rho(sample_size, dim=3, rho_min=-1., rho_max=1.):
    """
    """
    if dim == 1:
        return np.linspace(rho_min, rho_max, sample_size + 1,
                           endpoint=False)[1:]
    else:
        n = int(np.floor(sample_size**(1./dim)))
        v_rho_1 = [np.linspace(rho_min, rho_max, n + 1,
                               endpoint=False)[1:]]*(dim - 1)
        grid_rho_1 = np.meshgrid(*v_rho_1)
        list_rho_1 = np.vstack(grid_rho_1).reshape(dim - 1, -1).T

        list_rho = np.zeros((n**dim, dim))
        for i, rho_1 in enumerate(list_rho_1):
            a1 = rho_1[0]
            a2 = rho_1[1]
            list_rho_2 = get_a3(a1, a2, n)
            for j, rho_2 in enumerate(list_rho_2):
                tmp = rho_1.tolist()
                tmp.append(rho_2)
                list_rho[n*i+j, :] = tmp

        return list_rho


class ImpactOfDependence:
    def __init__(self, modelFunction, variables):
        """
        """
        self.setModelFunction(modelFunction)  # The model function
        self.setInputVariables(variables)  # The input variables
        self._rhoMin, self._rhoMax = -1., 1.
        self._tauMin, self._tauMax = -1., 1.

    def _createSample(self, numSample, fixed_grid, dep_measure, eps,
                      obsPerSample):
        """
            Create the observations for differents dependence parameters
        """
        dim = self._input_dim
        corr_dim = dim * (dim - 1) / 2

        # Number of different value of parameters
        numSample -= numSample % obsPerSample
        n_param = numSample / obsPerSample
        print n_param
#        n = int(np.floor(n_param ** (1. / corr_dim)))

#        numSample -= numSample % obsPerSample
        # Adapt the number of sample
#        numSample = n**corr_dim

        self._corr_dim = corr_dim  # Dimension of correlation parameters

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

            if self._copulaName == "NormalCopula":
                list_param = Conversion.\
                    NormalCopula.fromKendallToPearson(listTau)

            elif self._copulaName == "InverseClaytonCopula":
                list_param = Conversion.\
                    ClaytonCopula.fromKendallToPearson(listTau)

        elif dep_measure == "PearsonRho":
            rho_min, rho_max = self._rhoMin, self._rhoMax

            if fixed_grid:  # Fixed grid
                list_rho = get_grid_rho(n_param, corr_dim, rho_min, rho_max)
                n_param = list_rho.shape[0]
            else:  # Random grid
                list_rho = get_random_rho(n_param, corr_dim, rho_min, rho_max)

            if self._copulaName == "NormalCopula":
                list_param = list_rho

            elif self._copulaName == "InverseClaytonCopula":
                # TODO : check that shit
                list_param = Conversion.\
                    ClaytonCopula.fromPearsonToKendall(list_rho)
        else:
            raise Exception("Other dependence measures not yet implemented")

        numSample = n_param * obsPerSample
        self._inputSample = np.empty((numSample, dim))  # Input sample
        self._listParam = np.empty((numSample, corr_dim))  # Input Corr sample

        # We loop for each copula param and create observations for each
        for i, param in enumerate(list_param):  # For each copula parameter
            tmp = self._getSample(param, obsPerSample)

            # TODO : find a way to create a "real" class InverseClayton...
            if self._copulaName == "InverseClaytonCopula":
                tmp[:, 0] = -tmp[:, 0]

            # We save the input sample
            self._inputSample[obsPerSample*i:obsPerSample*(i+1), :] = tmp

            # As well for the parameters (for dependence measure)
            self._listParam[obsPerSample*i:obsPerSample*(i+1), :] = list_param[i]

    def buildForest(self):
        """
        """
        self._quantForest = QuantileForest(self._listParam, self._outputSample,
                                           n_jobs=8)

    def run(self,
            numSample,
            fixedGrid=True,
            depMeas="KendallTau",
            eps=1.E-3,
            obsPerSample=1):
        """
        """
        self._createSample(numSample, fixedGrid, depMeas, eps, obsPerSample)
        self._outputSample = self._modelFunction(self._inputSample)

    def saveInputData(self, path=".", fname="inputSampleCop", fileType=".csv"):
        """
        """
        fileName = path + '/' + fname + fileType
        np.savetxt(fileName, self._listParam)

    def _getSample(self, param, numObs):
        """
        must be the copula parameter
        """
        # TODO: change it when the new ot released is out
        if self._copulaName == "NormalCopula":
            if self._corr_dim == 1:
                self._corrMatrix[0, 1] = param
            elif self._corr_dim == 3:
                self._corrMatrix[0, 1] = param[0]
                self._corrMatrix[0, 2] = param[1]
                self._corrMatrix[1, 2] = param[2]
            self._copula = ot.NormalCopula(self._corrMatrix)
        else:
            self._copula.setParametersCollection([param])

        self._inputVariables.setCopula(self._copula)
        return np.array(self._inputVariables.getSample(numObs))

    def _empQuantileFunction(self, sample, alpha):
        """
        TODO : think about any other method to compute quantiles
        """
        return np.percentile(sample, alpha*100)

    def _computeEmpiricalQuantile(self, param, alpha, numObs):
        """
        must be copula parameter
        """
        inputSample = self._getSample(param, numObs)
        outputSample = self._modelFunction(inputSample)
        return self._empQuantileFunction(outputSample, alpha)

    def drawImpactOnQuantile(self, alpha, numPoints=100, depMeas="KendallTau",
                             with_sample=False, figsize=(10, 6),
                             saveFig=False):
        """

        """
        if depMeas == "KendallTau":
            listParam = np.linspace(self._tauMin, self._tauMax, numPoints)
            xlabel = "Kendall $\\tau$"
            quantiles = self._quantForest.computeQuantile(listParam, alpha)

        elif depMeas == "PearsonRho":
            listParam = np.linspace(self._rhoMin, self._rhoMax, numPoints)
            xlabel = "Pearson $\\rho$"
            quantiles = self._quantForest.computeQuantile(listParam, alpha)
        else:
            raise("Undefined param")

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(listParam, quantiles, 'b',
                label="Conditional %.2f %% quantiles" % (alpha), linewidth=2)

        if with_sample:
            ax.plot(self._listParam, self._outputSample, 'k.')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Quantile")
        ax.legend(loc="best")
        fig.tight_layout()
        if saveFig:
            if type(saveFig) is str:
                fname = saveFig + '/'
            else:
                fname = "./"
            fname += "condQuantiles"
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")

    def getQuantFunc(self, method, alpha, maxEvaluation, obsPerSample):
        """

        """
        if method == 1:
            if obsPerSample is None:
                obsPerSample = 10

            def quantFunc(param):
                if type(param) not in [int, float]:
                    if len(param) == self._corr_dim:
                        param = param[0]
                return self._computeEmpiricalQuantile(param, alpha, obsPerSample)

        elif method == 2:
            if obsPerSample is None:
                obsPerSample = 1

            if not hasattr(self, '_quantForest'):  # Forest is already created
                self.buildForest(maxEvaluation)

            def quantFunc(param):
                return self._quantForest.computeQuantile(param, alpha)

        return quantFunc

    def _nloptObjFunc(self, x, grad):
        """
        """
        return -self._quantFunc(x)

    def optimize(self, method, alpha, x0, maxEvaluation=100000,
                 obsPerSample=None, tol=1.E-8):
        """

        """
        self._quantFunc = self.getQuantFunc(method, alpha, maxEvaluation,
                                            obsPerSample)

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
        self._copulaName = self._copula.getName()
        self._inputVariables = variables
        if self._copulaName == "NormalCopula":
            self._corrMatrix = ot.CorrelationMatrix(self._input_dim)


# =============================================================================
#
# =============================================================================
print "oui"
if __name__ == "__main__":
    def function(x):
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
    dim = 3  # Problem dimension
    copulaName = "NormalCopula"  # Name of the used copula
    marginals = [ot.Normal()]*dim  # Marginals

    # TODO : find a way to create a real InverseClaytonCopula
    if copulaName == "NormalCopula":
        copula = ot.NormalCopula(dim)
    elif copulaName == "InverseClaytonCopula":
        copula = ot.ClaytonCopula(dim)
        copula.setName(copulaName)

    var = ot.ComposedDistribution(marginals, copula)

    # Parameters
    numSample = 64  # Number of sample
    alpha = 0.01  # Quantile probability
    fixedGrid = True  # Fixed design sampling
    estimationMethod = 2  # Used method
    obsPerSample = 1

    impact = ImpactOfDependence(function, var)
    impact.run(numSample, fixedGrid, obsPerSample=obsPerSample,
               depMeas="PearsonRho")
    impact.buildForest()
#    impact.drawImpactOnQuantile(alpha, depMeas="PearsonRho")
    #print impact.optimize(estimationMethod, alpha, [-0.5])
