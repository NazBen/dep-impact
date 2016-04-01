import numpy as np

class Conversion:
    """
    Static class to convert dependence parameters
    """
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