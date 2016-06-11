import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
vinecopula = importr('VineCopula')

def get_pos(dim, k):
    ll = np.cumsum(range(dim-1, 0, -1))
    i = np.where(k < ll)[0][0]
    j = dim - (ll[i] - k)
    return i, j

class Conversion(object):
    """
    Static class to convert dependence parameters
    """    
    def __init__(self, families):
        self.families = families
        
    @property
    def families(self):
        return self._families
        
    @families.setter
    def families(self, value):
        """
        """
        self._families = value
        self._input_dim = value.shape[0]

    def __call__(self, measure_param, dep_measure):
        return self.to_copula_parameter(measure_param, dep_measure)
        
    def to_copula_parameter(self, measure_param, dep_measure):
        """Convert the dependence_measure to the copula parameter.
        """
        if dep_measure == "KendallTau":
            n_sample, corr_dim = measure_param.shape
            copula_param = np.zeros((n_sample, corr_dim))
            for i, params in enumerate(measure_param):
                for j, param in enumerate(params):
                    k1, k2 = get_pos(self._input_dim, j)
                    print self._families.T[k1, k2], param
                    print vinecopula.BiCopTau2Par(self._families.T[k1, k2], param)[0]
                    copula_param[i, :] = vinecopula.BiCopTau2Par(self._families.T[k1, k2], param)[0]

            print copula_param
            copula_param = self._copula.fromKendallToParam(measure_param)
                    
        elif dep_measure == "PearsonRho":
            copula_param = self._copula.fromPearsonToParam(measure_param)
        else:
            raise ValueError("Unknow Dependence Measure")

        return copula_param
        
    def to_Kendall(self, measure_param):
        """Convert the dependence_measure to the copula parameter.
        """
        return self._copula.fromParamToKendall(measure_param)
        
                
    def to_Pearson(self, measure_param):
        """Convert the dependence_measure to the copula parameter.
        """
        return self._copula.fromParamToPearson(measure_param)
        
    class NormalCopula:
        """
        For Normal copula
        """
        @staticmethod
        def fromParamToKendall(param):
            """
            From Pearson correlation parameter to Kendal dependence parameter.
            """
            return 2. / np.pi * np.arcsin(param)

        @staticmethod
        def fromKendallToParam(tau):
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return np.sin(np.pi / 2. * tau)            
            
        @staticmethod
        def fromPearsonToParam(rho):
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return rho
            
                        
        @staticmethod
        def fromParamToPearson(rho):
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return rho
            
    class ClaytonCopula:
        """
        For Clayton copula
        """
        @staticmethod
        def fromParamToKendall(param):
            """
            From Pearson correlation parameter to Kendal dependence parameter.
            """
            return param / (param + 2.)

        @staticmethod
        def fromKendallToParam(tau):            
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return 2. * tau / (1. - tau)
            
        @staticmethod
        def fromPearsonToParam(rho):
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            raise NotImplementedError("Cannot convert Pearson to copula parameter for Clayton Copula")