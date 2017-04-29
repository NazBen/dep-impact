import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

VINECOPULA = importr('VineCopula')

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
                matrix[i, j] = int(VINECOPULA.BiCopName(matrix[i, j], False)[0])
            elif isinstance(matrix[i, j], int):
                pass
            else:
                raise ValueError("Uncorrect Family matrix")

    return matrix


class VineCopula(object):
    """Vine Copula Class."""

    def __init__(self, structure, family, param1, param2=None):
        self.structure = structure
        self.family = family
        self.param1 = param1
        self.param2 = param2
        ri.initr()
        self.build_vine()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        check_matrix(value)
        self._dim = value.shape[0]
        self._structure = value
        self._to_rebuild = True

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
        self._to_rebuild = True

    @property
    def param1(self):
        return self._param1

    @param1.setter
    def param1(self, value):
        check_matrix(value)
        assert value.shape[0] == self._dim, \
            AttributeError('Family matrix should be of dimension == %d' % (self._dim))
        self._param1 = value
        self._to_rebuild = True

    @property
    def param2(self):
        return self._param2

    @param2.setter
    def param2(self, value):
        if value is None:
            value = np.zeros((self._dim, self._dim))
        check_matrix(value)
        assert value.shape[0] == self._dim, \
            AttributeError('Family matrix should be of dimension == %d' % (self._dim))
        self._param2 = value
        self._to_rebuild = True

    def build_vine(self):
        """
        """
        r_structure = numpy2ri(self.structure)
        r_family = numpy2ri(self.family)
        r_par = numpy2ri(self.param1)
        r_par2 = numpy2ri(self.param2)
        self._rvine = VINECOPULA.RVineMatrix(r_structure, r_family, r_par, r_par2)
        self._to_rebuild = False

    def get_sample(self, n_obs):
        """
        """
        assert isinstance(n_obs, int), \
            TypeError("Sample size must be an integer.")
        assert n_obs > 0, \
            ValueError("Sample size must be positive.")
        if self._to_rebuild:
            self.build_vine()
        return np.asarray(VINECOPULA.RVineSim(n_obs, self._rvine))
