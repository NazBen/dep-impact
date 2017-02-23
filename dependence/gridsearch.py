import numpy as np

from skopt.utils import create_result
from sklearn.utils import check_random_state
from skopt.space import Space
from scipy.optimize import OptimizeResult

def quantile_func(alpha):
    """
    """
    def func(x, axis):
        return np.percentile(x, alpha*100., axis=axis)
    return func

def proba_func(threshold):
    """
    """
    def func(x, axis):
        return (x >= threshold).sum(axis=axis)
    return func

def gridsearch_minimize(func, dimensions, grid_size, n_calls, q_func='mean', grid_type='lhs', random_state=None, quantile=0.05):
    """
    """
    rng = check_random_state(random_state)

    # Create the grid
    space = Space(dimensions)
    Xi = space.rvs(grid_size)

    # Evaluate the sample
    out_samples = np.asarray(map(func, Xi*n_calls)).reshape(n_calls, grid_size).T

    if callable(q_func):
        yi = q_func(out_samples, axis=1)
    elif q_func == 'mean':
        yi = out_samples.mean(axis=1)
    elif q_func == 'quantile':
        yi = np.percentile(out_samples, alpha*100.)


    res = OptimizeResult()
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = yi
    res.x_iters = Xi
    res.space = space
    res.random_state = rng
    return res


class GridSearch(object):
    """
    """
    def __init__(self):
        pass