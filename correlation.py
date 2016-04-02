import numpy as np
import random
import openturns as ot
from pyDOE import lhs

"""
TODO :
    - Make the classes in Cython for better performances??
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

def create_random_correlation_param(dim, n=1, sampling="monte-carlo"):
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


if __name__ == "__main__":
    dim = 4
    n = 1
    print create_random_correlation_param(dim, n)

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


def get_grid_rho(n_sample, dim=3, rho_min=-1., rho_max=1., all_sample=True):
    """
    """
    if dim == 1:
        return np.linspace(rho_min, rho_max, n_sample + 1,
                           endpoint=False)[1:]
    else:
        if all_sample:
            n = int(np.floor(n_sample**(1./dim)))
        else:
            n = n_sample
        v_rho_1 = [np.linspace(rho_min, rho_max, n + 1,
                               endpoint=False)[1:]]*(dim - 1)
        grid_rho_1 = np.meshgrid(*v_rho_1)
        list_rho_1 = np.vstack(grid_rho_1).reshape(dim - 1, -1).T

        list_rho = np.zeros((n**dim, dim))
        for i, rho_1 in enumerate(list_rho_1):
            a1 = rho_1[0]
            a2 = rho_1[1]
            list_rho_2 = get_list_rho3(a1, a2, n)
            for j, rho_2 in enumerate(list_rho_2):
                tmp = rho_1.tolist()
                tmp.append(rho_2)
                list_rho[n*i+j, :] = tmp

        return list_rho