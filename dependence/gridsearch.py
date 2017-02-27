import numpy as np

from skopt.utils import create_result
from sklearn.utils import check_random_state
from skopt.space import Space as sk_Space
from scipy.optimize import OptimizeResult
from sklearn.utils.fixes import sp_version
import pyDOE

class Space(sk_Space):
    """
    """
    def rvs(self, n_samples=1, random_state=None, sampling='rand', 
            sampling_criterion='centermaximin'):
        """Draw random samples.
    
        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.
    
        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.
    
        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.
    
        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """
        rng = check_random_state(random_state)
    
        if sampling == 'rand':
            # Draw
            columns = []
        
            for dim in self.dimensions:
                if sp_version < (0, 16):
                    columns.append(dim.rvs(n_samples=n_samples))
                else:
                    columns.append(dim.rvs(n_samples=n_samples, random_state=rng))
        
            # Transpose
            rows = []
        
            for i in range(n_samples):
                r = []
                for j in range(self.n_dims):
                    r.append(columns[j][i])
        
                rows.append(r)
        elif sampling == 'lhs':
            # Draw
            sample = pyDOE.lhs(self.n_dims, samples=n_samples, criterion=sampling_criterion)
            tmp = np.zeros((n_samples, self.n_dims))
            for k, dim in enumerate(self.dimensions):
                tmp[:, k] = sample[:, k]*(dim.high - dim.low) + dim.low
            rows = tmp.tolist()
        return rows

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

def gridsearch_minimize(func, dimensions, grid_size, n_calls, q_func=np.mean, 
                        grid_type='rand', random_state=None):
    """
    """
    rng = check_random_state(random_state)

    # Create the grid
    space = Space(dimensions)
    Xi = space.rvs(grid_size, sampling=grid_type)

    # Evaluate the sample
    out_samples = np.asarray(map(func, Xi*n_calls)).reshape(n_calls, grid_size).T

    if callable(q_func):
        yi = q_func(out_samples, axis=1)

    # Set the outputs
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