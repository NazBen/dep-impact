import numpy as np
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr

from .vinecopula import ROTATED_FAMILIES

R_VINECOPULA = importr('VineCopula')


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
            if self.family in ROTATED_FAMILIES:
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
        if self.family in ROTATED_FAMILIES:
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


def get_tau_interval(family, eps=0.01):
    assert isinstance(family, (np.integer, str)), \
        TypeError("Input must be int or str, given:", type(family))
    if isinstance(family, str):
        family = int(R_VINECOPULA.BiCopName(family, False)[0])

    if family in [1, 2, 3, 4, 5, 6, 13, 14, 16]:
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

    # Single value or array
    if isinstance(values, float):
        if values == 0:
            converted_params = 0  # Independence
        else:
            # If < 0, we rotate the copula
            rot = 20 if values < 0 else 0
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
                family + 20, values[down_id])
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
