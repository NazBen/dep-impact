import numpy as np

class Conversion(object):
    """
    Static class to convert dependence parameters
    """    
    def __init__(self, copula_name):
        self.copula = copula_name
        
    @property
    def copula(self):
        return self._copula
        
    @copula.setter
    def copula(self, name):
        assert isinstance(name, str), \
            TypeError("The copula name must be a string")
            
        if name == "NormalCopula":
            copula = Conversion.NormalCopula
        elif name in ['ClaytonCopula', 'InverseClaytonCopula']:
            copula = Conversion.ClaytonCopula
        else:
            raise AttributeError("Not implemented or unknow copula")
                    
        self._copula = copula

    def __call__(self, measure_param, dep_measure):
        return self.to_copula_parameter(measure_param, dep_measure)
        
    def to_copula_parameter(self, measure_param, dep_measure):
        """Convert the dependence_measure to the copula parameter.
        """
        if dep_measure == "KendallTau":
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