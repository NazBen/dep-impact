import numpy as np
import rpy2.rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects.numpy2ri import numpy2ri

R_VINECOPULA = importr('VineCopula')
AVAILABLE_COPULAS = [1, 2, 3, 4, 5, 6, 13, 14, 16]
ROTATED_FAMILIES = [3, 4, 6, 7, 13, 14, 16, 17]
NON_GAUSSIAN_FAMILIES = list(set(AVAILABLE_COPULAS) - set([1, 2]))

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

        # Rotate copulas
        negative_ids = param1 < 0.
        rotate_ids = np.isin(families, ROTATED_FAMILIES)
        elements = np.where(negative_ids & rotate_ids)
        families[elements] += 20

        # Independent
        indep_ids = param1 == 0.
        non_gaussian_ids = np.isin(families, NON_GAUSSIAN_FAMILIES)
        elements = np.where(indep_ids & non_gaussian_ids)
        families[elements] = 0

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
            param2 = np.tril(np.ones((self._dim, self._dim))*2., k=-1)
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
        self._rvine = R_VINECOPULA.RVineMatrix(
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
        sample = np.asarray(R_VINECOPULA.RVineSim(n_obs, self._rvine))
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
        lilelihood = np.asarray(R_VINECOPULA.RVineLogLik(
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


class Conversion(object):
    """
    Static class to convert dependence parameters.

    Parameters
    ----------
    family : int or str,
        The copula family which can be :
        - 1 : Normal,
        - 2 : Student,
        - 3 : Clayton,
        - 4 : Gumbel,
        - 5 : Frank,
        - 6 : Joe,
        - 13 : survival Clayton,
        - 14 : survival Gumbel,
        - 16 : survival Joe.
    """

    def __init__(self, family):
        self.family = family

    def to_parameter(self, values, dep_measure='kendall'):
        """Convert the dependence_measure to the copula parameter.

        Parameters:
        ----------
        values : float or list/array of float
            The dependence measure vlaue to convert.
        dep_measure : str, optional (default='kendall')
            The dependence measure to convert.

        Returns
        -------
        params : float or list of float
            The copula parameter(s) associated to the values.

        """
        check_values(values)
        if dep_measure == "kendall":
            if self.family in NON_GAUSSIAN_FAMILIES:
                params = convert_values(
                    kendall_to_parameter, self.family, values)
            else:
                params = kendall_to_parameter(self.family, values)
        elif dep_measure == "pearson":
            raise NotImplementedError('Maybe in the next versions...')
        elif dep_measure == "spearman":
            raise NotImplementedError('Maybe in the next versions...')
        else:
            raise ValueError("Unknow Dependence Measure")
        return params

    def to_kendall(self, params):
        """Converts the copula parameter to the kendall's tau dependence measure.      

        Parameters
        ----------
        values : float or list/array of float
            The copula parameters.

        Returns
        -------
        kendalls : float or array of float
            The kendall's coefficients.
        """
        check_values(params)
        if self.family in NON_GAUSSIAN_FAMILIES:
            kendalls = convert_values(
                parameter_to_kendall, self.family, params)
        else:
            kendalls = parameter_to_kendall(self.family, params)
        return kendalls

    def to_pearson(self, params):
        """Convert the copula parameter to the pearson correlation value.

        Parameters
        ----------
        values : float or list/array of float
            The copula parameters.

        Returns
        -------
        pearsons : float or array of float
            The pearson correlation values.
        """
        raise NotImplementedError('Not yet...')

    @property
    def family(self):
        """The copula family.
        """
        return self._family

    @family.setter
    def family(self, value):
        if isinstance(value, (int, np.integer)):
            np.testing.assert_equal(value, int(value))
            self._family = int(value)
            self._family_name = R_VINECOPULA.BiCopName(int(value), False)[0]
        elif isinstance(value, str):
            self._family = int(R_VINECOPULA.BiCopName(value, False)[0])
            self._family_name = value
        else:
            raise TypeError("Unkow family {}".format(value))


def get_tau_interval(family, eps=0.02):
    assert isinstance(family, (np.integer, str)), \
        TypeError("Input must be int or str, given:", type(family))
    if isinstance(family, str):
        family = int(R_VINECOPULA.BiCopName(family, False)[0])

    # TODO:
    # the frank copula does not have an explicit transformation
    # for the kendall tau. There is some problems at the bounds
    if family == 5:
        return -0.97, 0.97
    if family in AVAILABLE_COPULAS:
        return -1+eps, 1.-eps
    else:
        raise NotImplementedError("Not implemented yet.")


def to_ri(values):
    """Convert values to rpy interface.

    Parameters:
    ----------
    values : float or array
        The numerical values.

    Returns
    -------
    values
        The converted values.
    """
    if isinstance(values, np.ndarray):
        return numpy2ri(values)
    elif isinstance(values, float):
        return values
    else:
        raise TypeError('Unknow type for values: {}'.format(values))


def convert_values(convert_func, family, values):
    """Convert values from copula parameter to dependence measure, or inverse.

    Parameters:
    ----------
    convert_func : callable
        A function that converts the values.
    family : int
        The copula family.
    values : float or array
        The value to convert.
    Returns
    -------
    converted_params : float or array
        The converted values.
    """
    assert callable(convert_func), "The function is not callable."
    assert isinstance(family, int), "The family is not an integer."

    delta_rot = 20 if family in ROTATED_FAMILIES else 0
    # Single value or array
    if isinstance(values, float):
        if values == 0:
            converted_params = 0  # Independence
        else:
            # If < 0, we rotate the copula
            rot = delta_rot if values < 0 else 0
            converted_params = convert_func(family + rot, values)
    else:
        # Different conversion depending on the sign of the values
        converted_params = np.zeros(values.shape)
        up_id = values > 0  # Normal
        down_id = values < 0  # To rotate
        null_id = values == 0  # Independence
        if up_id.any():
            converted_params[up_id] = convert_func(family, values[up_id])
        if down_id.any():
            converted_params[down_id] = convert_func(
                family + delta_rot, values[down_id])
        if null_id.any():
            converted_params[null_id] = 0.
    return converted_params


def kendall_to_parameter(family, kendalls):
    """For a given copula family, converts a kendall coefficient to 
    the associated copula paramter value.

    Parameters:
    ----------
    family : int
        The copula family.
    kendalls : float or array
        The kendall coefficients.

    Returns
    -------
    params : float or array
        The copula parameter values.
    """
    params = base_conv_to(R_VINECOPULA.BiCopTau2Par, family, kendalls)
    return params


def parameter_to_kendall(family, params):
    """For a given copula family, converts a copula parameter 
    to the kendall coefficient value.

    Parameters:
    ----------
    family : int
        The copula family.
    params : float or array
        The kendall coefficients.

    Returns
    -------
    kendalls : float or array
        The kendall coefficent values.
    """
    kendalls = base_conv_to(R_VINECOPULA.BiCopPar2Tau, family, params)
    return kendalls


def base_conv_to(conv_func, family, values):
    """Base parameter convertion function for a given copula family.

    Parameters:
    ----------
    conv_func : callable,
        The converting function.
    family : int
        The copula family.
    kendalls : float or array
        The kendall coefficients.

    Returns
    -------
    converted_values : float or array
        The converted values.
    """
    values = to_ri(values)
    converted_values = np.asarray(conv_func(family, values))
    if converted_values.size == 1:
        # Because rpy2 returns a list even for a float
        converted_values = converted_values.item()
    return converted_values


def check_values(values):
    """Checks the validity of the values 

    Parameters:
    ----------
    values : float, list or array
        The values to convert.
    Raises
    ------
    TypeError
        If the values are not correct    
    """

    if isinstance(values, list):
        values = np.asarray(values)

    if isinstance(values, np.ndarray):
        assert values.shape[0] == values.size, "It must be a vector"
    elif isinstance(values, float):
        pass
    else:
        raise TypeError("Wrong type of values: {}".format(type(values)))


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
                    R_VINECOPULA.BiCopName(matrix[i, j], False)[0])
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
