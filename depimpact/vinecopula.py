import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

VINECOPULA = importr('VineCopula')
ROTATED_FAMILIES = [3, 4, 6, 13, 14, 16]


class VineCopula(object):
    """Vine Copula class.

    Parameters:
    ----------
    structure : array
        The vine structure.
    family : array
        The pair-copula families.
    param1 : array
        The first parameters of each pair-copula of the vine structure.
    param2 : array, optional (default=None)
        The second parameters of each pair-copula of the vine structure.
    """

    def __init__(self, structure, family, param1, param2=None):
        self.structure = structure
        self.family = family
        self.param1 = param1
        self.param2 = param2
        ri.initr()
        self.build_vine()

    @property
    def structure(self):
        """The vine structure.
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        structure = check_structure(structure)
        self._dim = structure.shape[0]
        self._structure = structure
        self._to_rebuild = True

    @property
    def family(self):
        """The pair-copula families.
        """
        return self._family

    @family.setter
    def family(self, families):
        families = check_family(families)
        assert families.shape[0] == self._dim, 'Wrong matrix dimension'
        self._family = families
        self._family_changed = families
        self._to_rebuild = True

    @property
    def param1(self):
        """The first parameters of the pair-copula.
        """
        return self._param1

    @param1.setter
    def param1(self, param1):
        param1 = check_matrix(param1)
        param1 = check_triangular(param1, k=1)
        dim = self._dim
        assert param1.shape[0] == dim, 'Wrong matrix dimension'

        families = self._family.copy()
        negative_ids = param1 < 0.
        indep_ids = param1 == 0.
        non_gaussian_ids = np.isin(families, ROTATED_FAMILIES)
        elements = np.where(negative_ids & non_gaussian_ids)
        families[elements] += 20
        elements = np.where(indep_ids & non_gaussian_ids)
        families[elements] = 1

        self._family_changed = families
        self._param1 = param1
        self._to_rebuild = True

    @property
    def param2(self):
        """The second parameters of each pair-copula.
        """
        return self._param2

    @param2.setter
    def param2(self, param2):
        if param2 is None:
            param2 = np.zeros((self._dim, self._dim))
        param2 = check_matrix(param2)
        param2 = check_triangular(param2, k=1)
        assert param2.shape[0] == self._dim, 'Wrong matrix dimension'
        self._param2 = param2
        self._to_rebuild = True

    def build_vine(self):
        """After being initialized, the vine copula is created.
        """
        r_structure = numpy2ri(self._structure)
        r_family = numpy2ri(permute_params(
            self._family_changed, self._structure))
        r_par = numpy2ri(permute_params(self._param1, self._structure))
        r_par2 = numpy2ri(permute_params(self._param2, self._structure))
        self._rvine = VINECOPULA.RVineMatrix(
            r_structure, r_family, r_par, r_par2)
        self._to_rebuild = False

    def get_sample(self, n_obs):
        """Sample observations from the vine copula object.

        Parameters
        ----------
        n_obs : int
            The sample size.

        Returns
        -------
        sample : array
            The sample from the vine copula.
        """
        assert isinstance(n_obs, int), "Sample size is not integer"
        assert n_obs > 0, "Sample size must be positive."
        if self._to_rebuild:
            self.build_vine()
        sample = np.asarray(VINECOPULA.RVineSim(n_obs, self._rvine))
        return sample

    def get_pdf(self, x):
        """Computes the probability density function.

        Parameters
        ----------
        x : float or array
            The input values.

        Returns
        -------
        pdf : float or array
            The density values.
        """
        x = numpy2ri(x)
        pdf = np.asarray(VINECOPULA.RVineLogLik(x, self._rvine)[0])
        return pdf

    def loglikelihood(self, x):
        """Computes the loglikelihood function.

        Parameters
        ----------
        x : float or array
            The input values.

        Returns
        -------
        lilelihood : float or array
            The loglikelihood.
        """
        x = numpy2ri(x)
        lilelihood = np.asarray(VINECOPULA.RVineLogLik(
            x, self._rvine, separate=True, calculate_V=False)[0])
        return lilelihood

    def grad_loglikelihood(self, x):
        """Computes the gradient of the loglikelihood function.

        Parameters
        ----------
        x : float or array
            The input values.

        Returns
        -------
        grad : float or array
            The gradient of the loglikelihood function.
        """
        x = numpy2ri(x)
        grad = np.asarray(VINECOPULA.RVineGrad(x, self._rvine)[0])**2
        return grad


def check_matrix(matrix):
    """Check the validity of a matrix for the vine copula.

    Parameters:
    ----------
    matrix : array,
        A matrix.
    Returns
    -------
    matrix : array,
        The corrected matrix.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)
    assert matrix.ndim == 2, 'Matrix should be of dimension 2.'
    assert matrix.shape[0] == matrix.shape[1], 'Matrix should be squared.'
    return matrix


def check_triangular(matrix, k=1):
    """Checks if a matrix is triangular.

    Parameters:
    ----------
    matrix : array
        The triangular matrix.
    k : int, (default=1)
        The distance with the diagonal.

    Returns
    -------
    matrix : array
        The corrected triangular matrix.
    """
    up = np.triu(matrix, k=k)
    down = np.tril(matrix, k=-k)
    is_up = (up == matrix).all()
    is_down = (down == matrix).all()
    assert is_up or is_down, "The matrix should be triangular"
    # By convention, we want lower triangular matrices.
    if is_up and matrix.all() > 0:
        print(matrix)
        print("Matrix transposed")
        matrix = matrix.T
    return matrix


def check_family(matrix):
    """Check the validity of a family matrix for the vine copula.

    Parameters:
    ----------
    matrix : array
        The pair-copula families.

    Returns
    -------
    matrix : array
        The corrected matrix.
    """
    # TODO: check if the families are in the list of copulas
    matrix = check_matrix(matrix)
    matrix = check_triangular(matrix, k=1)
    dim = matrix.shape[0]
    for i in range(dim):
        for j in range(i):
            if isinstance(matrix[i, j], str):
                matrix[i, j] = int(
                    VINECOPULA.BiCopName(matrix[i, j], False)[0])
            elif isinstance(matrix[i, j], np.integer):
                pass
    matrix = matrix.astype(int)
    return matrix


def check_structure(structure):
    """Check a vine structure matrix.

    Parameters:
    ----------
    matrix : array
        The vine structure array.

    Returns
    -------
    matrix : array
        The corrected matrix.
    """
    structure = check_matrix(structure)
    structure = check_triangular(structure, k=0)
    # TODO: add the checking of a vine structure definition
    return structure


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
