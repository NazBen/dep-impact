import numpy as np

from sklearn.utils import check_random_state
from skopt.space import Space as sk_Space
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
        n_samples : int, optional (default=1)
            Number of samples to be drawn from the space.
    
        random_state : int, RandomState or None, optional (default=None)
            Set random state to something other than None for reproducible
            results.
    
        Returns
        -------
        points : list of lists, shape=(n_points, n_dims)
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
    """To associate an alpha to an empirical quantile function.
    
    Parameters
    ----------
    alpha : float
        The probability of the target quantile. The value must be between 0 and 1.
            
    Returns
    -------
    q_func : callable
        The quantile function.
            
    """
    def q_func(x, axis=1):
        return np.percentile(x, alpha*100., axis=axis)
    return q_func

def proba_func(threshold):
    """To associate an alpha to an empirical distribution function.
    
    Parameters
    ----------
    threshold : float
        The threshold of the target probability.
            
    Returns
    -------
    p_func : callable
        The probability function.
            
    """
    def p_func(x, axis=1):
        return (x >= threshold).sum(axis=axis)
    return p_func