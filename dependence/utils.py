import os
import operator
import numpy as np
import openturns as ot
from itertools import product
from pyDOE import lhs
from sklearn.utils import check_random_state
from skopt.space import Space as sk_Space
from sklearn.utils.fixes import sp_version

OPERATORS = {">=": operator.ge,
            ">": operator.gt}


def get_grid_sample(dimensions, n_sample, grid_type):
    """Sample inside a fixed design space.
    
    Parameters
    ----------
    dimensions : array,
        The bounds of the space for each dimensions.
    n_sample: int,
        The number of observations inside the space.
    grid_type: str,
        The type of sampling.

    """
    # We create the grid
    space = Space(dimensions)
    sample = space.rvs(n_sample, sampling=grid_type)
    return sample


class Space(sk_Space):
    """
    """
    def rvs(self, n_samples=1, sampling='rand', 
            lhs_sampling_criterion='centermaximin', random_state=None):
        """Draw random samples.
    
        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.
    
        Parameters
        ----------
        n_samples : int or None, optional (default=1)
            Number of samples to be drawn from the space. If None and 
            sampling is 'vertices', then all the space vertices are taken.

        sampling : str, optional (default='rand')
            The sampling strategy, which can be:
            - 'rand' : Random sampling
            - 'lhs' : Latin Hypercube Sampling is done
            - 'vertices' : Sampling over the vertices of the space.

        lhs_sampling_criterion : str, optional (default='centermaximin')
            The sampling criterion for the LHS sampling.

        random_state : int, RandomState or None, optional (default=None)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        points : list of lists, shape=(n_points, n_dims)
           Points sampled from the space.
        """
        rng = check_random_state(random_state)
        assert isinstance(sampling, str), \
            TypeError("sampling must be a string")

        if sampling == 'rand':
            # Random sampling
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
            # LHS sampling
            sample = lhs(self.n_dims, samples=n_samples, criterion=lhs_sampling_criterion)
            tmp = np.zeros((n_samples, self.n_dims))
            # Assert the bounds
            for k, dim in enumerate(self.dimensions):
                tmp[:, k] = sample[:, k]*(dim.high - dim.low) + dim.low
            rows = tmp.tolist()
        elif sampling == 'vertices':
            # Sample on the vertices of the space.
            n_pair = len(self.dimensions)
            bounds = list(product([-1., 1., 0.], repeat=n_pair))
            if n_samples is None:
                bounds.remove((0.,)*n_pair) # remove indepencence
            bounds = np.asarray(bounds)
            n_bounds = len(bounds)

            if n_samples is None:
                # We take all the vertices
                n_samples = n_bounds
                sample = bounds
            else:
                # Random sampling over the vertices
                n_samples = min(n_samples, n_bounds)
                id_taken = np.random.choice(n_bounds, size=n_samples, replace=False)
                sample = bounds[sorted(id_taken), :]

            # Assert the bounds
            for p in range(n_pair):
                sample_p = sample[:, p]
                sample_p[sample_p == -1.] = self.dimensions[p].low
                sample_p[sample_p == 1.] = self.dimensions[p].high

            rows = sample.tolist()
            
        elif sampling == 'fixed':
            raise NotImplementedError("Maybe I'll do it...")
        else:
            raise NameError("Sampling type does not exist.")
        return rows


def list_to_matrix(values, dim):
    """Transform a list of values in a lower triangular matrix.

    Parameters
    ----------
    param : list, array
        The list of values
    dim : int,
        The shape of the matrix

    Returns
    -------
    matrix : array
        The lower triangular matrix
    """
    matrix = np.zeros((dim, dim))
    k = 0
    for i in range(1, dim):
        for j in range(i):
            matrix[i, j] = values[k]
            k += 1

    return matrix


def matrix_to_list(matrix, return_ids=False, return_coord=False, op_char='>'):
    """Convert a lower triangular martix into a list of its values.

    Parameters
    ----------
    matrix : array
        A square lower triangular matrix.
    return_ids : bool, optional (default=False)
        If true, the index of the list are returns.
    return_coord : bool, optional (default=False)
        If true, the coordinates in the matrix are returns.
    op_char : str, optional (default='>')
        If '>', only the positive values of the matrix are taken into account.

    Returns
    -------
    values : list
        The list of values in the matrix. If only_positive is True, then only the positive values
        are returns.
    """
    op_func = OPERATORS[op_char]
    values = []
    ids = []
    coord = []
    dim = matrix.shape[0]
    k = 0
    for i in range(1, dim):
        for j in range(i):
            if op_func(matrix[i, j], 0):
                values.append(matrix[i, j])
                ids.append(k)
                coord.append([i, j])
            k += 1

    if return_ids and return_coord:
        return values, ids, coord
    elif return_ids:
        return values, ids
    elif return_coord:
        return values, coord
    else:
        return values


def bootstrap(data, num_samples, statistic):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
    
    Inspired from: http://people.duke.edu/~ccc14/pcfb/analysis.html"""
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, axis=1))
    return stat


def to_kendalls(converters, params):
    """Convert the copula parameters to kendall's tau.

    Parameters
    ----------
    converter s: list of VineCopula converter
        The converters from the copula parameter to the kendall tau for the given families.
    params : list or array
        The parameters of each copula converter.

    Returns
    -------
    kendalls : list
        The kendalls tau of the given parameters of each copula
    """
    if isinstance(params, list):
        params = np.asarray(params)
    elif isinstance(params, float):
        params = np.asarray([params])

    n_params, n_pairs = params.shape
    kendalls = np.zeros(params.shape)
    for k in range(n_pairs):
        kendalls[:, k] = converters[k].to_kendall(params[:, k])

    # If there is only one parameter, no need to return the list
    if kendalls.size == 1:
        kendalls = kendalls.item()
    return kendalls

def to_copula_params(converters, kendalls):
    """Convert the kendall's tau to the copula parameters.

    Parameters
    ----------
    converters : list of VineCopula converters
        The converters from the kendall tau to the copula parameter of the given families.
    kendalls : list or array
        The kendall's tau values of each pairs.
    Returns
    -------
    params : array
        The copula parameters.
    """
    if isinstance(kendalls, list):
        kendalls = np.asarray(kendalls)
    elif isinstance(kendalls, float):
        kendalls = np.asarray([kendalls])

    n_params, n_pairs = kendalls.shape
    params = np.zeros(kendalls.shape)
    for k in range(n_pairs):
        params[:, k] = converters[k].to_copula_parameter(kendalls[:, k], dep_measure='kendall-tau')

    # If there is only one parameter, no need to return the list
    if params.size == 1:
        params = params.item()
    return params


def margins_to_dict(margins):
    """Convert the margin's informations into a dictionary.

    Parameters
    ----------
    margins : the list of OpenTurns distributions
        The marginal distributions of the input variables.

    Returns
    -------
    margin_dict : dict
        The dictionary with the information of each marginal.
    """
    margin_dict = {}
    for i, marginal in enumerate(margins):
        margin_dict[i] = {}
        name = marginal.getName()
        params = list(marginal.getParameter())
        if name == 'TruncatedDistribution':
            margin_dict[i]['Type'] = 'Truncated'
            in_marginal = marginal.getDistribution()
            margin_dict[i]['Truncated Parameters'] = params
            name = in_marginal.getName()
            params = list(in_marginal.getParameter())
        else:        
            margin_dict[i]['Type'] = 'Standard'
            
        margin_dict[i]['Marginal Family'] = name
        margin_dict[i]['Marginal Parameters'] = params    
    return margin_dict

def dict_to_margins(margin_dict):
    """Convert a dictionary with margins informations into a list of distributions.
    
    Parameters
    ----------
    margin_dict : dict
        A dictionary of information on the margins
    
    Returns
    -------
    margins
    """
    margins = []
    for i in sorted(margin_dict.keys()):
        marginal = getattr(ot, margin_dict[i]['Marginal Family'])(*margin_dict[i]['Marginal Parameters'])
        if margin_dict[i]['Type'] == 'TruncatedDistribution':
            params = margin_dict[i]['Bounds']        
            marginal = ot.TruncatedDistribution(marginal, *params)
        margins.append(marginal)
    
    return margins


def save_dependence_grid(dirname, kendalls, bounds_tau, grid_type):
    """Save a grid of kendall's into a csv ifile.

    The grid is always saved in Kendall's Tau measures.

    Parameters
    ----------
    dirname : str
        The directory path.

    kendalls : list or array
        The kendall's tau of each pair.

    bounds_tau : list or array
        The bounds on the kendall's tau.

    grid_type : str
        The ype of grid.

    Returns
    -------
    grid_filename : str
        The grid filename
    """
    kendalls = np.asarray(kendalls)
    n_param, n_pairs = kendalls.shape
    
    # The sample variable to save
    sample = np.zeros((n_param, n_pairs))
    for k in range(n_pairs):
        tau_min, tau_max = bounds_tau[k]
        sample[:, k] = (kendalls[:, k] - tau_min) / (tau_max - tau_min)
        
    k = 0
    do_save = True
    name = '%s_p_%d_n_%d_%d.csv' % (grid_type, n_pairs, n_param, k)
    
    grid_filename = os.path.join(dirname, name)
    # If this file already exists
    while os.path.exists(grid_filename):
        existing_sample = np.loadtxt(grid_filename).reshape(n_param, -1)
        # We check if the build sample and the existing one are equivalents
        if np.allclose(existing_sample, sample):
            do_save = False
            print('The DOE already exist in %s' % (name))
            break
        k += 1
        name = '%s_p_%d_n_%d_%d.csv' % (grid_type, n_pairs, n_param, k)
        grid_filename = os.path.join(dirname, name)
        
    # It is saved
    if do_save:
        np.savetxt(grid_filename, sample)
        print("Grid saved at %s" % (grid_filename))

    return grid_filename


def load_dependence_grid(dirname, n_pairs, n_params, bounds_tau, grid_type, use_grid=None):
    """Load a grid of parameters

    Parameters
    ----------
    dirname : str
        The directory path.

    n_params : int
        The grid dimension (the number of dependent pairs).

    n_params : int
        The grid size of the sample.

    bounds_tau : list or array
        The bounds on the kendall's tau.

    grid_type : str
        The ype of grid.

    use_grid : int, str or None, optional (default=None)
        If a particular grid should be used.

    Returns
    -------
    kendalls : array
        The kendall's tau of each dependent pairs.
    filename : str
        The name of the loaded grid.
    """
    if isinstance(use_grid, str):
        filename = use_grid
        name = os.path.basename(filename)
    elif isinstance(use_grid, (int, bool)):
        k = int(use_grid)
        name = '%s_p_%d_n_%d_%d.csv' % (grid_type, n_pairs, n_params, k)
        filename = os.path.join(dirname, name)
    else:
        raise AttributeError('Unknow use_grid')

    assert os.path.exists(filename), 'Grid file %s does not exists' % name
    print('loading file %s' % name)
    sample = np.loadtxt(filename).reshape(n_params, n_pairs)
    assert n_params == sample.shape[0], 'Wrong grid size'
    assert n_pairs == sample.shape[1], 'Wrong dimension'

    kendalls = np.zeros((n_params, n_pairs))
    for k in range(n_pairs):
        tau_min, tau_max = bounds_tau[k]
        kendalls[:, k] = sample[:, k]*(tau_max - tau_min) + tau_min
        
    return kendalls, filename


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