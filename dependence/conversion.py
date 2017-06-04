"""Conversion of copula.

This class makes the conversion between copula parameters and dependence measures
"""
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

VINECOPULA = importr('VineCopula')

def get_param1_interval(copula):
    """
    """
    assert isinstance(copula, (int, str)), \
        TypeError("Input must be int or str")
    if isinstance(copula, str):
        copula = int(VINECOPULA.BiCopName(copula, False)[0])

    if copula in [1, 2]:
        return -1., 1.
    elif copula in [3, 13]:
        return 0., np.inf
    elif copula in [23, 33]:
        return 0., -np.inf,
    elif copula in [4, 14, 24, 34]:
        return 1., np.inf
    elif copula in [5]:
        return -np.inf, np.inf
    elif copula in [6, 26, 36]:
        return 1., np.inf
    else:
        raise NotImplementedError("Not implemented yet.")

def get_param2_interval(copula):
    """
    """
    assert isinstance(copula, (np.integer, str)), \
        TypeError("Input must be int or str")

    if copula in [2, 't']:
        return 1.e-6, np.inf
    else:
        raise NotImplementedError("Not implemented yet.")

def get_tau_interval(copula):
    assert isinstance(copula, (np.integer, str)), \
        TypeError("Input must be int or str. Not: ", type(copula))
    if isinstance(copula, str):
        copula = int(VINECOPULA.BiCopName(copula, False)[0])

    if copula in [1, 2]:
        return -0.99, 0.99
    elif copula in [3, 13, 4, 14, 5, 6, 16]:
        return 0., 0.99
    elif copula in [23, 24, 26, 33, 34, 36]:  # Rotated copulas
        return -0.99, 0.
    else:
        raise NotImplementedError("Not implemented yet.")

class Conversion(object):
    """
    Static class to convert dependence parameters.
    """    
    def __init__(self, family):
        self.family = family

    def to_copula_parameter(self, measure_param, dep_measure='kendall-tau'):
        """Convert the dependence_measure to the copula parameter.
        """
        if isinstance(measure_param, list):
            measure_param = np.asarray(measure_param)

        if isinstance(measure_param, np.ndarray):
            n_sample = measure_param.shape[0]
            assert n_sample == measure_param.size, \
                AttributeError("It must be a vector")
        elif isinstance(measure_param, float):
            n_sample = 1
            measure_param = np.asarray([measure_param])
        else:
            raise TypeError("Wrong type for measure_param")
        
        if dep_measure == "kendall-tau":
            r_kendall = numpy2ri(measure_param)
            copula_param = np.asarray(VINECOPULA.BiCopTau2Par(self._family, r_kendall))
        elif dep_measure == "pearson-rho":
            copula_param = self._copula.fromPearsonToParam(measure_param)
        else:
            raise ValueError("Unknow Dependence Measure")

        if copula_param.size == 1:
            return copula_param.item()
        else:
            return copula_param
    
    def to_kendall(self, params):
        """Convert the dependence_measure to the copula parameter.
        """
        if isinstance(params, np.ndarray):
            r_params = numpy2ri(params)
        elif isinstance(params, float):
            r_params = params
        else:
            raise TypeError("Wrong type for params. Got: ", type(params))
        copula_param = np.asarray(VINECOPULA.BiCopPar2Tau(self._family, r_params))
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
        if isinstance(value, (int, float, np.integer)):
            np.testing.assert_equal(value, int(value))
            self._family = int(value)
            self._family_name = VINECOPULA.BiCopName(int(value), False)[0]            
        elif isinstance(value, str):
            self._family = int(VINECOPULA.BiCopName(value, False)[0])
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