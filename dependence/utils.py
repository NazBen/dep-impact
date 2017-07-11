import os
import operator
import numpy as np
import openturns as ot
from itertools import product, permutations
import copy
from pyDOE import lhs
from sklearn.utils import check_random_state
from skopt.space import Space as sk_Space
from sklearn.utils.fixes import sp_version

OPERATORS = {">=": operator.ge,
            ">": operator.gt,
            "==": operator.eq}

MAX_N_PAIR_VERTICES = 13


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
        
    Returns
    -------
    sample : array,
        The sample from the given space and the given grid.
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
            
            if n_pair > MAX_N_PAIR_VERTICES:
                if n_samples is None:
                    raise MemoryError('Too much pairs to create a vertices grid.')
                
                # Iterative add instead of sampling in a big bounds matrix
                sample = []
                for i in range(n_samples):
                    obs = np.random.choice([-1., 1., 0.], size=n_pair).tolist()
                    if obs not in sample:
                        sample.append(obs)
                        
                sample = np.asarray(sample)
            else:
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
        return (x >= threshold).mean(axis=axis)
    return p_func


def asymptotic_error_quantile(n, q_density, q_alpha):
    """
    """
    return np.sqrt(q_alpha * (1. - q_alpha) / (n * q_density**2))


def asymptotic_error_proba(n, proba):
    """
    """    
    return np.sqrt(proba * (1. - proba) / n)


def get_pairs_by_levels(dim, forced_pairs_ids, verbose=False):
    """Split a list of pairs in levels.
    
    Parameters
    ----------
    dim : int,
        The number of variables.
    forced_pairs_ids : list of int,
        The list of pair ids.
    verbose : bool,
        If True, informations about the procedure are printed.
        
    Returns
    -------
    pairs_by_levels : list of list,
        The list of listed pairs.
    """
    n_pairs = int(dim*(dim-1)/2)
    n_forced_pairs = len(forced_pairs_ids)
    assert len(np.unique(forced_pairs_ids)) == n_forced_pairs, \
        "The list should contain unique values"
    assert n_forced_pairs <= n_pairs, "Too many pairs: %d > %d" % (n_forced_pairs > n_pairs)
    assert max(forced_pairs_ids) < n_pairs, "Wrong pair id: %d" % (max(forced_pairs_ids))
    
    forced_pairs = get_pairs(dim, forced_pairs_ids, with_plus=True)
    print(forced_pairs)
    all_pairs = np.asarray(np.tril_indices(dim, k=-1)).T
    print(all_pairs)
    print(np.setdiff1d(all_pairs, forced_pairs, assume_unique=True))

    remaining_pairs_ids = list(range(0, n_pairs))
    for pair_id in forced_pairs_ids:
        remaining_pairs_ids.remove(pair_id)
    remaining_pairs = get_pairs(dim, remaining_pairs_ids, with_plus=True)
    
    if verbose:
        print('Vine dimension: %d' % dim)
        print('Conditioning information:')
        
    k0 = 0
    n_conditionned = range(dim - 1, 0, -1)
    pairs_by_levels = []
    for i in range(dim-1):
        k = n_conditionned[i]
        k1 = min(n_forced_pairs, k0+k)
        if verbose:
            print("\t%d pairs with %d conditionned variables" % (k, i))
            print("Pairs: {0}".format(forced_pairs[k0:k0+k].tolist()))
        if forced_pairs[k0:k0+k].tolist():
            pairs_by_levels.append(forced_pairs[k0:k0+k].tolist())
        k0 = k1
    if verbose:
        print("Concerned pairs: {0}".format(forced_pairs.tolist()))
        print("Remaining pairs: {0}".format(remaining_pairs.tolist()))

    idx = 1
    init_pairs_by_levels = copy.deepcopy(pairs_by_levels)  # copy
    while not np.all([check_node_loop(pairs_level) for pairs_level in pairs_by_levels]):
        pairs_by_levels = copy.deepcopy(pairs_by_levels)
        n_levels = len(pairs_by_levels)
        lvl = 0
        while lvl < n_levels:
            pairs_level = pairs_by_levels[lvl]
            if not check_node_loop(pairs_level):
                # A new level is created
                if lvl == n_levels - 1:
                    pairs_by_levels.append([pairs_level.pop(-idx)])
                    n_levels += 1
                else:
                    # The 1st pair of the next level is replaced by the last of the previous
                    pairs_by_levels[lvl+1].insert(0, pairs_level.pop(-idx))
                    pairs_by_levels[lvl].append(pairs_by_levels[lvl+1].pop(1))
            lvl += 1
        idx += 1
    return pairs_by_levels


def add_pair(structure, pair, index, lvl):
    """Adds a pair in a Vine structure in a certain place and for a specific conditionement.
    
    Parameters
    ----------
    structure : array,
        The R-vine structure array.
    pair : list,
        A pair of variables that we want to add.
    index : int, 
        The position of the pair of variable.
    lvl : int,
        The tree level.

    Returns
    -------
    structure : array,
        The updated R-vine structure.
    """
    dim = structure.shape[0]
    if lvl == 0: # If it's the unconditiononal variables
        assert structure[index, index] == 0, \
            "There is already a variable at [%d, %d]" % (index, index)
        assert structure[dim-1, index] == 0, \
            "There is already a variable at [%d, %d]" % (dim-1, index)
        structure[index, index] = pair[0]
        structure[dim-1, index] = pair[1]
    else:
        assert structure[index, index] == pair[0], \
            "Diagonal element of the structure should be the same as the first variable of the pair: %d != %d" % (structure[index, index], pair[0])
        assert structure[dim-1-lvl, index] == 0, \
            "There is already a variable at [%d, %d]" % (dim-1-lvl, index)
        structure[dim-1-lvl, index] = pair[1]
    return structure


def add_pairs(structure, pairs, lvl, verbose=False):
    """Adds pairs in a structure for a selected level of conditionement.

    Parameters
    ----------
    structure : array,
        The R-vine structure array.
    pairs : list,
        The list of pairs of variables.
    lvl : int,
        The tree level for which we want the variables to be in.
    verbose : bool,
        If True, informations about the procedure are printed.
    """
    dim = structure.shape[0]
    n_pairs = len(pairs)

    assert n_pairs < dim-lvl, "Not enough place to fill the pairs : %d > %d" % (n_pairs, dim-lvl)
    
    # Let's find a way to place these pairs inside the structure
    n_slots = dim - 1 - lvl
    possibilities = list(permutations(range(n_slots), r=n_pairs))
    success = False
    for possibility in possibilities:
        try:
            # Add the pair in the possible order
            structure_modif = np.copy(structure)
            for i in range(n_pairs):
                structure_modif = add_pair(structure_modif, pairs[i], possibility[i], lvl)
            
            if check_redundancy(structure_modif):
                success = True
                break
        except AssertionError:
            pass

    if not success and verbose:
        print('Did not succeded to fill the structure with the given pairs')

    # If it's the 1st level, the last row of last column must be filled
    if (lvl == 0) and (n_pairs == dim-1):
        structure_modif[dim-1, dim-1] = np.setdiff1d(range(dim+1), np.diag(structure_modif))[0]
    return structure_modif


def check_redundancy(structure):
    """Check if there is no redundancy of the diagonal variables.
    """
    dim = structure.shape[0]
    diag = np.diag(structure)
    for i in range(dim-1):
        # Check if it does not appears later in the matrix
        if diag[i] != 0.:
            if diag[i] in structure[:, i+1:]:
                return False
    return True


def rotate_pairs(init_pairs, rotations):
    """Rotate the pairs according to some rotations.
    """
    n_pairs = len(init_pairs)
    assert len(rotations) == n_pairs, \
        "The number of rotations is different to the number of pairs %d != %d" % (len(rotations), n_pairs)
    assert not np.setdiff1d(np.unique(rotations), [1, -1]), \
        "The rotations list should only be composed of -1 and 1."
    pairs = []
    for i in range(n_pairs):
        if rotations[i] == -1:
            pairs.append(list(reversed(init_pairs[i])))
        else:
            pairs.append(init_pairs[i])
    return pairs


def get_possible_structures(dim, pairs_by_levels, verbose=False):
    """
    """
    # For each levels
    good_structures = []
    for lvl, pairs_level in enumerate(pairs_by_levels):
        n_pairs_level = len(pairs_level) # Number of pairs in the level
        
        # The possible combinations
        combinations = list(product([1, -1], repeat=n_pairs_level))
        
        # Now lets get the possible pair combinations for this level
        for k, comb_k in enumerate(combinations):
            # Rotate the pair to the actual combination
            pairs_k = rotate_pairs(pairs_level, comb_k)
            if lvl == 0:
                # Create the associated vine structure
                structure = np.zeros((dim, dim), dtype=int)
                structure = add_pairs(structure, pairs_k, lvl, verbose=verbose)
                if check_redundancy(structure):
#                    structure[dim-2, dim-3] = np.setdiff1d(range(1, dim+1), np.diag(structure)[:dim-2].tolist() + [structure[dim-1, dim-3]])[0]
                    good_structures.append(structure)
            else:
                for structure in good_structures:
                    try:
                        new_structure = add_pairs(structure, pairs_k, lvl, verbose=verbose)
                    except:
                        print("Can't add the pairs {0} in the current structure...".format(pairs_k))
                    if check_redundancy(new_structure):
                        structure = new_structure
            
    
    remain_structures = []
    for structure in good_structures:
        tmp = fill_structure(structure)
        if is_vine_structure(tmp):
            structure = tmp
            remain_structures.append(tmp)
            if verbose:
                print('good:\n{0}'.format(structure))

    return remain_structures


def check_structure_shape(structure):
    """Check if the structure shape is correct.
    """
    assert structure.shape[0] == structure.shape[1], "Structure matrix should be squared"
    assert np.triu(structure, k=1).sum() == 0, "Matrix should be lower triangular"
    
    
def check_natural_order(structure):
    """Check if a parent node is included in a child node.
    """
    d = structure.shape[0]
    for i in range(d-1):
        i = 1
        for j in range(i+1):
            parent = [[structure[j, j], structure[i+1, j]], [structure[i+2:d, j].tolist()]]
            col = structure[:, j]
            parent_elements = col[np.setdiff1d(np.arange(j, d), range(j+1, i+1))]

            i_c = i + 1
            if len(parent_elements) > 2:
                n_child = 0
                for j_c in range(i_c+1):
                    possible_child = [[structure[j_c, j_c], structure[i_c+1, j_c]], [structure[i_c+2:d, j_c].tolist()]]
                    col = structure[:, j_c]
                    possible_child_elements = col[np.setdiff1d(np.arange(j_c, d), range(j_c+1, i_c+1))]
                    if len(np.intersect1d(possible_child_elements, parent_elements)) == d-i-1:
                        n_child += 1
                if n_child < 2:
                    return False

    return True
    

def is_vine_structure(matrix):
    """Check if the given matrix is a Vine Structure matrix
    """
    dim = matrix.shape[0]
    diag = np.diag(matrix)
    check_structure_shape(matrix)
    assert matrix.max() == dim, "Maximum should be the dimension: %d != %d" % (matrix.max(), dim)
    assert matrix.min() == 0, "Minimum should be 0: %d != %d" % (matrix.min(), dim)
    assert len(np.unique(diag)) == dim, "Element should be uniques on the diagonal: %d != %d" % (len(np.unique(diag)), dim)
    for i in range(dim):
        column_i = matrix[i:, i]
        assert len(np.unique(column_i)) == dim - i, "Element should be unique for each column: %d != %d" % ( len(np.unique(column_i)), dim - i)
        for node in diag[:i]:
            assert node not in column_i, "Previous main node should not exist after"
            
    if check_natural_order(matrix):
        return True
    else:
        return False

    
def check_node_loop(pairs, n_p=3):
    """Check if not too many variables are connected in a single tree
    """
    for perm_pairs in list(permutations(pairs, r=n_p)):
        if len(np.unique(perm_pairs)) <= n_p:
            return False
    return True


def fill_structure(structure):
    """Fill the structure with the remaining variables
    """
#    print structure
    dim = structure.shape[0]
#    structure[dim-2, dim-3] = np.setdiff1d(range(1, dim+1), np.diag(structure)[:dim-2].tolist() + [structure[dim-1, dim-3]])[0]
    
    diag = np.unique(np.diag(structure)).tolist()
    if 0 in diag:
        diag.remove(0)
    remaining_vals = np.setdiff1d(range(1, dim+1), diag)
    
    n_remaining_vals = len(remaining_vals)
    
    diag_possibility = list(permutations(remaining_vals, n_remaining_vals))[0]

    
    for k, i in enumerate(range(dim - n_remaining_vals, dim)):
        structure[i, i] = diag_possibility[k]
    
    structure[dim-1, dim-2] = structure[dim-1, dim-1]
    
    for i in range(dim - n_remaining_vals, dim-2):
        structure[dim-1, i] = structure[i+1, i+1]
        
    lvl = 1
    for j in range(dim-2, 0, -1):
        for i in range(j):
            col_i = structure[:, i]
            tmp = col_i[np.setdiff1d(np.arange(i, dim), range(j+1, i+1))]
            var_col_i = tmp[tmp != 0]
            for i_c in range(i+1, j+1):
                col_ic = structure[:, i_c]
                tmp = col_ic[np.setdiff1d(np.arange(i_c, dim), range(j+1, i_c+1))]
                var_col_i_c = tmp[tmp != 0]
                intersection = np.intersect1d(var_col_i, var_col_i_c)
                if len(intersection) == lvl:
                    tt = var_col_i_c.tolist()
                    for inter in intersection:
                        tt.remove(inter)
                    structure[j, i] = tt[0]
                    break
        lvl += 1

    prev_ind = []
    for i in range(dim):
        values_i = structure[i:, i]
        used_values_i = values_i[values_i != 0].tolist() + prev_ind
        remaining_i = list(range(1, dim + 1))
        for var in used_values_i:
            if (var in remaining_i):
                remaining_i.remove(var)
        values_i[values_i == 0] = remaining_i
        prev_ind.append(values_i[0])
    return structure


def get_pair_id(dim, pair, with_plus=True):
    """ Get the pair id from a given index.    
    
    Parameters
    ----------
    dim : int,
        The dimension.
    pair : tuple of int,
        The pair of variable.
    with_plus : bool,
        If True, the variable numbers start at 1 instead of 0.
        
    Return
    ------
    pair_id : int,
        The pair index.
    """
    pairs = np.asarray(np.tril_indices(dim, k=-1)).T
    if with_plus:
        pair[0] += 1
        pair[1] += 1
        
    pair_id = np.where((pairs == pair).all(axis=1))[0][0]
    return pair_id


def get_pair(dim, index, with_plus=True):
    """ Get the pair of variables from a given index.
    
    Parameters
    ----------
    dim : int,
        The dimension.
    index : int,
        The ID of the pair.
    with_plus : bool,
        If True, the variable numbers start at 1 instead of 0.
        
    Return
    ------
    pair : tuple of int,
        The pair of variables.
    """    
    pairs = np.asarray(np.tril_indices(dim, k=-1)).T
    pair = pairs[index]
    if with_plus:
        pair += 1
    pair = tuple(pair)
    
    return pair


def get_pairs(dim, pair_ids, with_plus=True):
    """Get the associated pairs of variables from given IDs.
    
    Parameters
    ----------
    dim : int,
        The dimension.
    pair_ids : list of int,
        The IDs of the pairs.
    with_plus : bool,
        If True, the variable numbers start at 1 instead of 0.
    
    Return
    ------
    pairs : list of tupple,
        The list of pairs.    
    """
    pairs = []
    for pair_id in pair_ids:
        pairs.append(get_pair(dim, pair_id, with_plus))
        
    pairs = list(map(tuple, pairs))
    return pairs