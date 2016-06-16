import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

from .vinecopula import check_matrix, check_family

vinecopula = importr('VineCopula')

def get_param1_interval(copula):
    """
    """
    assert isinstance(copula, (int, str)), \
        TypeError("Input must be int or str")

    if copula in [1, 'Gaussian', 2, 't']:
        return -1., 1.
    elif copula in [3, 'Clayton']:
        return 0., np.inf
    elif copula in [4, 'Gumbel']:
        return 1., np.inf
    elif copula in [5, 'Frank']:
        return -np.inf, np.inf
    else:
        raise NotImplementedError("Not implemented yet.")

def get_param2_interval(copula):
    """
    """
    assert isinstance(copula, (int, str)), \
        TypeError("Input must be int or str")

    if copula in [2, 't']:
        return 1.e-6, np.inf
    else:
        raise NotImplementedError("Not implemented yet.")

def get_tau_interval(copula):
    assert isinstance(copula, (int, str)), \
        TypeError("Input must be int or str")

    if copula in [1, 'Gaussian', 2, 't']:
        return -1., 1.
    elif copula in [3, 'Clayton']:
        return 0., 1.
    elif copula in [4, 'Gumbel']:
        return 0., 1.
    elif copula in [5, 'Frank']:
        return 0., 1.
    else:
        raise NotImplementedError("Not implemented yet.")

class Conversion(object):
    """
    Static class to convert dependence parameters.
    """    
    def __init__(self, family):
        self.family = family

    def to_copula_parameter(self, measure_param, dep_measure):
        """Convert the dependence_measure to the copula parameter.
        """
        if isinstance(measure_param, np.ndarray):
            n_sample = measure_param.shape[0]
            assert n_sample == measure_param.size, \
                AttributeError("It must be a vector")
        elif isinstance(measure_param, float):
            n_sample = 1
        else:
            raise TypeError("Wrong type for measure_param")
        
        if dep_measure == "KendallTau":
            r_params = numpy2ri(measure_param)
            copula_param = np.asarray(vinecopula.BiCopTau2Par(self._family, r_params))
                    
        elif dep_measure == "PearsonRho":
            copula_param = self._copula.fromPearsonToParam(measure_param)
        else:
            raise ValueError("Unknow Dependence Measure")

        return copula_param
    
    def to_Kendall(self, params):
        """Convert the dependence_measure to the copula parameter.
        """
        r_params = numpy2ri(params)
        copula_param = np.asarray(vinecopula.BiCopPar2Tau(self._family, r_params))
        return copula_param
        
                
    def to_Pearson(self, measure_param):
        """Convert the dependence_measure to the copula parameter.
        """
        return self._copula.fromParamToPearson(measure_param)
        
    @property
    def family(self):
        return self._family
        
    @family.setter
    def family(self, value):
        """
        """
        if isinstance(value, (int, float)):
            np.testing.assert_equal(value, int(value))
            self._family = value
            self._family_name = vinecopula.BiCopName(value, False)[0]            
        elif isinstance(value, str):
            self._family = int(vinecopula.BiCopName(value, False)[0])
            self._family_name = value
        else:
            raise TypeError("Unkow Type for family")

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
       
         
def get_pos(dim, k):
    ll = np.cumsum(range(dim-1, 0, -1))
    i = np.where(k < ll)[0][0]
    j = dim - (ll[i] - k)
    return i, j