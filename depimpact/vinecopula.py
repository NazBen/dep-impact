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
            elif isinstance(matrix[i, j], np.integer):
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
        r_family = numpy2ri(permute_params(self.family, self.structure))
        r_par = numpy2ri(permute_params(self.param1, self.structure))
        r_par2 = numpy2ri(permute_params(self.param2, self.structure))
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

    def loglikelihood(self, sample):
        """
        """
        data = numpy2ri(sample)
        return np.asarray(VINECOPULA.RVineLogLik(data, self._rvine, separate=True, calculate_V=False)[0])

    def grad_loglikelihood(self, sample):
        """
        """
        data = numpy2ri(sample)
        return np.asarray(VINECOPULA.RVineGrad(data, self._rvine)[0])**2


def permute_params(params, structure):
    """Permute the parameters of the initiat parameter matrix to fit to the R-vine structure.

    Parameters
    ----------
    params : array,
        The matrix of parameters.
    structure : array,
        The R-vine structure array.


    Returns
    -------
    permuted_params : array,
        The permuter matrix of parameters.
    """
    dim = params.shape[0]
    permuted_params = np.zeros(params.shape)
    for i in range(dim):
        for j in range(i+1, dim):
            if structure[i, i] > structure[j, i]:
                coords = structure[i, i]-1, structure[j, i]-1
            else:
                coords = structure[j, i]-1, structure[i, i]-1
            permuted_params[j, i] = params[coords]
        
    return permuted_params