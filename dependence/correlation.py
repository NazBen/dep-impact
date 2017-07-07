import numpy as np
import random
import openturns as ot
from pyDOE import lhs

from .conversion import Conversion

"""
TODO :
    - Make the classes in Cython for better performances??
    - Change the functions to create a sample for custom number of correlation inputs
    and not conditionally to the parameters dimension.
"""

class SemiDefiniteMatrixError(Exception):
    """
    Specific exception for matrix which are not semi-definite positives
    """
    pass


def create_corr_matrix(rho, dim=None, library="openturns"):
    """
    Create correlation matrices from a list of input parameters
    """
    rho = np.asarray(rho) # We convert the parameter in numpy array

    # Check if the correlation parameter is correct
    if (rho >= 1.).any() or (rho <= -1.).any():
        raise ValueError("Correlation parameters not in [-1., 1.]")

    # Dimenion of correlation parameter
    corr_dim = len(rho)

    # The dimension problem is computed if not specified
    if not dim:
        root = np.roots([1., -1., -2.*corr_dim])[0]
        dim = int(root)

    # Initialize the matrix
    if library == "openturns":
        corr_matrix = ot.CorrelationMatrix(dim)
    elif library == "numpy":
        corr_matrix = np.identity(dim, dtype=np.float)
    else:
        raise ValueError("Unknow value for library")

    k = 0
    for i in range(dim):
        for j in range(i+1, dim):
            corr_matrix[i, j] = rho[k]
            if library == "numpy":
                corr_matrix[j, i] = rho[k]
            k += 1
    
    # Check if the matrix is Positive Definite
    error = SemiDefiniteMatrixError("The matrix is not semi-positive definite")
    if library == "openturns":
        if corr_matrix.isPositiveDefinite():
            raise error
    elif library == "numpy":
        if (np.linalg.eigvals(corr_matrix) <= 0.).any():
            raise error

    return corr_matrix


def check_params(rho, dim=None):
    """
    Check is the matrix of a given 
    """
    rho = np.asarray(rho) # We convert the parameter in numpy array

    # Check if the correlation parameter is correct
    if (rho >= 1.).any() or (rho <= -1.).any():
        raise ValueError("Correlation parameters not in [-1., 1.]")

    # Dimenion of correlation parameter
    corr_dim = len(rho)

    # The dimension problem is computed if not specified
    if not dim:
        root = np.roots([1., -1., -2.*corr_dim])[0]
        dim = int(root)

    # Initialize the matrix
    corr_matrix = ot.CorrelationMatrix(dim)
    
    k = 0
    for i in range(dim):
        for j in range(i+1, dim):
            corr_matrix[i, j] = rho[k]
            k += 1
    
    return corr_matrix.isPositiveDefinite()

def create_random_correlation_param_previous(dim, n=1, sampling="monte-carlo"):
    """
    Using acceptation reject...
    """
    # Dimenion of correlation parameter
    corr_dim = dim * (dim - 1) / 2

    # Array of correlation parameters
    list_rho = np.zeros((n, corr_dim), dtype=np.float)

    for i in range(n): # For each parameter
        condition = True
        # Stop when the matrix is definit semi positive
        while condition:
            if sampling == "monte-carlo":
                rho = np.random.uniform(-1., 1., corr_dim)
            elif sampling == "lhs":
                rho = (lhs(corr_dim, samples=1)*2. - 1.).ravel()
            else:
                raise ValueError("Unknow sampling strategy")
            if check_params(rho, dim):
                condition = False
        list_rho[i, :] = rho
        
    if n == 1:
        return list_rho.ravel()
    else:
        return list_rho
        
def create_random_kendall_tau(corr_variables, n=1, sampling="monte-carlo"):
    """
    Using acceptation reject...

    corr_variables: the matrix to defined the variables that are correlated
    """
    dim = corr_variables.shape[0]
    corr_dim = dim * (dim - 1) / 2
    corr_vars = []  # Correlated variables
    k = 0
    for i in range(1, dim):
        for j in range(i):
            # If the variables are correlated,
            # we add the correlation ID in the list
            if corr_variables[i, j]:
                corr_vars.append(k)
            k += 1
    n_corr_vars = len(corr_vars)

    # Array of correlation parameters
    list_tau = np.zeros((n, corr_dim))

    for i in range(n): # For each parameter
        condition = True
        # Stop when the matrix is definite semi positive
        while condition:
            if sampling == "monte-carlo":
                u = np.random.uniform(-1., 1., n_corr_vars)
            elif sampling == "lhs":
                u = (lhs(n_corr_vars, samples=1)*2. - 1.).ravel()
            else:
                raise ValueError("Unknow sampling strategy")

            u = Conversion.NormalCopula.fromKendallToParam(u)

            if n_corr_vars == corr_dim:
                rho = u
            else:
                rho = np.zeros(corr_dim)
                rho[corr_vars] = u

            if check_params(rho, dim):
                condition = False

        list_tau[i, :] = Conversion.NormalCopula.fromParamToKendall(rho)

    if n == 1:
        return list_tau.ravel()
    else:
        return list_tau

def create_random_correlation_param(corr_variables, n=1, sampling="monte-carlo"):
    """
    Using acceptation reject...

    corr_variables: the matrix to defined the variables that are correlated
    """

    # Dimension problem
    dim = corr_variables.shape[0]    
    # Number of correlation parameters
    corr_dim = dim * (dim - 1) / 2
    # Correlated variables
    corr_vars = []
    k = 0
    for i in range(dim):
        for j in range(i+1, dim):
            # If the variables are correlated,
            # we add the correlation ID in the list
            if corr_variables[i, j]:
                corr_vars.append(k)
            k += 1
    n_corr_vars = len(corr_vars)
    # Array of correlation parameters
    list_rho = np.zeros((n, corr_dim), dtype=np.float)

    for i in range(n): # For each parameter
        condition = True
        # Stop when the matrix is definit semi positive
        while condition:
            if sampling == "monte-carlo":
                u = np.random.uniform(-1., 1., n_corr_vars)
            elif sampling == "lhs":
                u = (lhs(n_corr_vars, samples=1)*2. - 1.).ravel()
            else:
                raise ValueError("Unknow sampling strategy")

            if n_corr_vars == corr_dim:
                rho = u
            else:
                rho = np.zeros(corr_dim)
                rho[corr_vars] = u

            if check_params(rho, dim):
                condition = False
        list_rho[i, :] = rho
        
    if n == 1:
        return list_rho.ravel()
    else:
        return list_rho


def get_random_rho_3d(size, dim, rho_min=-1., rho_max=1.):
    """
    Works in 1 and 3 dimension
    """
    if dim == 1:
        list_rho = np.asarray(ot.Uniform(rho_min, rho_max).getSample(size))
    else:  # TODO : make it available in d dim
        list_rho = np.zeros((size, dim))
        for i in range(size):
            rho1 = random.uniform(rho_min, rho_max)
            rho2 = random.uniform(rho_min, rho_max)
            l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
            u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
            rho3 = random.uniform(l_bound, u_bound)
            list_rho[i, :] = [rho1, rho2, rho3]

    return list_rho


def get_list_rho3(rho1, rho2, n):
    """
    """
    l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
    u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
    list_rho3 = np.linspace(l_bound, u_bound, n+1, endpoint=False)[1:]
    return list_rho3


def get_grid_rho(corr_variables, n):
    """
    TODO: think about how to build a sample when not all the variables
    Get the grid of rho parameters
    """
    # Dimension problem
    dim = corr_variables.shape[0]
    # Number of correlation parameters
    corr_dim = dim * (dim - 1) / 2
    # Correlated variables
    corr_vars = []
    k = 0
    for i in range(dim):
        for j in range(i+1, dim):
            # If the variables are correlated,
            # we add the correlation ID in the list
            if corr_variables[i, j]:
                corr_vars.append(k)
            k += 1

    n_corr_vars = len(corr_vars)
    # Array of correlation parameters
    list_rho = np.zeros((n, corr_dim), dtype=np.float)

    if n_corr_vars == 1:  # Easy
        grid = np.linspace(-1., 1., n + 1, endpoint=False)[1:].reshape(n, 1)
    else:  # Not that easy...
        raise NotImplementedError('Not implemented for dim > 1')

    list_rho[:, corr_vars] = grid
    
    return list_rho


def get_grid_rho_prev(n_sample, dim=3, corr_dim=None, all_sample=True):
    """
    TODO: think about how to build a sample when not all the variables are correlated.
    Get the grid of rho parameters
    """
    if not corr_dim:
        corr_dim = dim * (dim - 1) / 2
    if all_sample:
        n = int(np.floor(n_sample**(1./corr_dim)))
    else:
        n = n_sample

    if corr_dim == 1:  # Easy
        return np.linspace(-1., 1., n + 1, endpoint=False)[1:]
    else:  # Not that easy...
        v_rho = [np.linspace(-1., 1., n+1, endpoint=False)[1:]]*(corr_dim - 1)
        grid_rho = np.meshgrid(*v_rho)
        list_rho = np.vstack(grid_rho).reshape(corr_dim - 1, -1).T

        list_rho = np.zeros((n**corr_dim, corr_dim))
        for i, rho_1 in enumerate(list_rho_1):
            a1 = rho_1[0]
            a2 = rho_1[1]
            list_rho_2 = get_list_rho3(a1, a2, n)
            for j, rho_2 in enumerate(list_rho_2):
                tmp = rho_1.tolist()
                tmp.append(rho_2)
                list_rho[n*i+j, :] = tmp

        return list_rho
