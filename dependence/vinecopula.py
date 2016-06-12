import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri
vinecopula = importr('VineCopula')

def check_matrix(value):
    assert isinstance(value, np.ndarray), \
        TypeError('Variable must be a numpy array.')
    assert value.ndim == 2, \
        AttributeError('Matrix should be of dimension 2.')
    assert value.shape[0] == value.shape[1], \
        AttributeError('Matrix should be squared.')

def check_family(matrix):
    d = matrix.shape[0]
    for i in range(d):
        for j in range(i):
            if isinstance(matrix[i, j], str):
                matrix[i, j] = int(vinecopula.BiCopName(matrix[i, j], False)[0])
            elif isinstance(matrix[i, j], int):
                pass
            else:
                raise ValueError("Uncorrect Family matrix")

    return matrix
class VineCopula(object):
    """Vine Copula Class."""

    def __init__(self, structure, family, param1, param2):
        self.structure = structure
        self.family = family
        self.param1 = param1
        self.param2 = param2
        self.build_vine()
        ri.initr()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        check_matrix(value)
        self._dim = value.shape[0]
        self._structure = value

    @property
    def family(self):
        return self._family

    @family.setter
    def family(self, value):
        check_matrix(value)
        check_family(value)
        assert value.shape[0] == self._dim, \
            AttributeError('Family matrix should be of dimension == %d' % (self._dim))
        self._family = value

    @property
    def param1(self):
        return self._param1

    @param1.setter
    def param1(self, value):
        check_matrix(value)
        assert value.shape[0] == self._dim, \
            AttributeError('Family matrix should be of dimension == %d' % (self._dim))
        self._param1 = value
        
    @property
    def param2(self):
        return self._param2

    @param2.setter
    def param2(self, value):
        check_matrix(value)
        assert value.shape[0] == self._dim, \
            AttributeError('Family matrix should be of dimension == %d' % (self._dim))
        self._param2 = value

    def build_vine(self):
        """
        """
        r_structure = numpy2ri(self.structure)
        r_family =  numpy2ri(self.family)
        r_par =  numpy2ri(self.param1)
        r_par2 =  numpy2ri(self.param2)
        self._rvine = vinecopula.RVineMatrix(r_structure, r_family, r_par, r_par2)

    def get_sample(self, n):
        """
        """
        assert isinstance(n, int), \
            TypeError("Sample size must be an integer.")
        assert n > 0, \
            TypeError("Sample size must be positive.")
        return np.asarray(vinecopula.RVineSim(n, self._rvine))