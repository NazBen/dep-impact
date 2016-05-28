import numpy as np

class Conversion:
    """
    Static class to convert dependence parameters
    """

    @staticmethod
    def to_copula_parameter(copula_name, measure_param, dep_measure):
        """Convert the dependence_measure to the copula parameter.
        """
        if dep_measure == "KendallTau":
            if copula_name == "NormalCopula":
                copula_param = Conversion.\
                    NormalCopula.fromKendallToPearson(measure_param)

            elif copula_name == "InverseClaytonCopula":
                copula_param = Conversion.\
                    ClaytonCopula.fromKendallToPearson(measure_param)
        elif dep_measure == "PearsonRho":
            if copula_name == "NormalCopula":
                copula_param = measure_param
            elif copula_name == "InverseClaytonCopula":
                copula_param = Conversion.\
                    ClaytonCopula.fromPearsonToKendall(measure_param)
        else:
            raise ValueError("Unknow Dependence Measure")

        return copula_param
    class NormalCopula:
        """
        For Normal copula
        """
        @staticmethod
        def fromPearsonToKendall(rho):
            """
            From Pearson correlation parameter to Kendal dependence parameter.
            """
            return 2. / np.pi * np.arcsin(rho)

        @staticmethod
        def fromKendallToPearson(tau):
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return np.sin(np.pi / 2. * tau)

    class ClaytonCopula:
        """
        For Clayton copula
        """
        @staticmethod
        def fromPearsonToKendall(rho):
            """
            From Pearson correlation parameter to Kendal dependence parameter.
            """
            return rho / (rho + 2.)

        @staticmethod
        def fromKendallToPearson(tau):            
            """
            From Kendal dependence parameter to Pearson correlation parameter.
            """
            return 2. * tau / (1. - tau)