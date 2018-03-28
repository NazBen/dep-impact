import numpy as np
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr

R_VINECOPULA = importr('VineCopula')

def get_param1_interval(copula):
    """
    """
    assert isinstance(copula, (int, str)), \
        TypeError("Input must be int or str")
    if isinstance(copula, str):
        copula = int(R_VINECOPULA.BiCopName(copula, False)[0])

    if copula in [1, 2]:
        return -1., 1.
    elif copula in [3, 13]:
        return 0., np.inf
    elif copula in [23, 33]:
        return -np.inf, 0.
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


def get_tau_interval(family, eps=0.02):
    assert isinstance(family, (np.integer, str)), \
        TypeError("Input must be int or str, given:", type(family))
    if isinstance(family, str):
        family = int(R_VINECOPULA.BiCopName(family, False)[0])

    if family in [1, 2, 3, 4, 6, 13, 14, 16]:
        return -1+eps, 1.-eps
    else:
        raise NotImplementedError("Not implemented yet.")


def to_ri(values):
    if isinstance(values, np.ndarray):
        values = numpy2ri(values)
    return values

def convert_to(convert_func, family, values):    
    if isinstance(values, float):
        if values == 0:
            params = 0
        else:
            rot = 0 if values > 0 else 20
            params = convert_func(family + rot, values)
    else:
        params = np.zeros(values.shape)
        up_id = values > 0
        down_id = values < 0
        null_id = values == 0
        if up_id.any():
            params[up_id] = convert_func(family, values[up_id])
        if down_id.any():
            params[down_id] = convert_func(family+20, values[down_id])    
        if null_id.any():
            params[null_id] = 0.
    return params


def kendall_to_parameter(family, values):
    values = to_ri(values)
    params = np.asarray(R_VINECOPULA.BiCopTau2Par(family, values))
    if params.size == 1:
        params = params.item()
    return params


def parameter_to_kendall(family, values):
    values = to_ri(values)
    kendalls =  np.asarray(R_VINECOPULA.BiCopPar2Tau(family, values))
    if kendalls.size == 1:
        kendalls = kendalls.item()
    return kendalls


class Conversion(object):
    """
    Static class to convert dependence parameters.
    
    Parameters
    ----------
    family : int or str,
        The copula family.
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
            if self._family in [1, 2]:
                params = kendall_to_parameter(self._family, values)
            else:
                params = convert_to(kendall_to_parameter, self._family, values)
        elif dep_measure == "pearson":
            raise NotImplementedError('Maybe in the next versions...')
        elif dep_measure == "spearman":
            raise NotImplementedError('Maybe in the next versions...')
        else:
            raise ValueError("Unknow Dependence Measure")
        return params
    

    def to_kendall(self, values):
        """Convert the dependence_measure to the copula parameter.      
        
        Parameters
        ----------       
        
        Returns
        -------
        """
        check_values(values)
        if self._family in [1, 2]:
            kendalls = parameter_to_kendall(self._family, values)
        else:
            kendalls = convert_to(parameter_to_kendall, self._family, values)
        return kendalls

    def to_pearson(self, measure_param):
        """Convert the dependence_measure to the copula parameter.
                
        Parameters
        ----------
        
        
        Returns
        -------
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


def check_values(values):
    if isinstance(values, list):
        values = np.asarray(values)

    if isinstance(values, np.ndarray):
        assert values.shape[0] == values.size, "It must be a vector"
    elif isinstance(values, float):
        pass
    else:
        raise TypeError("Wrong type of values: {}".format(type(values)))
