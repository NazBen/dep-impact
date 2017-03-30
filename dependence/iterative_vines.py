import numpy as np
import copy
import itertools

from .dependence import ListDependenceResult

GRIDS = ['lhs', 'rand']

def iterative_vine_minimize(estimate_object, n_input_sample, n_dep_param_init, p_max, grid_type='lhs', 
                            q_func=np.var, n_add_pairs=1, n_remove_pairs=1, 
                            with_bootstrap=False, re_use_params=False, verbose=False):
    """
    """

    quant_estimate = copy.copy(estimate_object)
    max_n_pairs = quant_estimate.corr_dim
    assert grid_type in GRIDS, "Unknow Grid type {0}".format(grid_type)
    assert 0 < p_max < max_n_pairs, "Maximum number of pairs must be positive"
    assert 1 <= n_add_pairs < max_n_pairs, "Must add at least one pair at each iteration"
    assert 0 <= n_remove_pairs < max_n_pairs, "This cannot be negative"
    assert callable(q_func), "Quantity function must be callable"
    
    init_family = quant_estimate.families
    dim = quant_estimate.input_dim
    
    families = np.zeros((dim, dim))
    selected_pairs = []
    removed_pairs = []
    worst_quantities = []
    all_params = np.zeros((0, max_n_pairs))
    
    cost = 0
    p = 0
    n_dep_param = n_dep_param_init
    while p < p_max:
        all_results = ListDependenceResult()
        min_quantity = {}
        pair_id = 0
        for i in range(1, dim):
            for j in range(i):
                if ((i, j) not in selected_pairs) and ((i, j) not in removed_pairs):
                    # Family matrix for this iteration
                    tmp_families = np.copy(families)
                    tmp_families[i, j] = init_family[i, j]
                    
                    # Family matrix is changed
                    quant_estimate.families = tmp_families
                    pairs_iter = selected_pairs + [(i, j)]
                    pairs_iter_id = [get_pair_id(dim, pair, with_plus=False) for pair in pairs_iter]
                    pairs_by_levels = get_pairs_by_levels(dim, pairs_iter_id)
                    
                    quant_estimate.vine_structure = get_possible_structures(dim, pairs_by_levels)[1]

#                    print pairs_iter
                    print quant_estimate.vine_structure
                    print quant_estimate.families
                                  
                    # Lets get the results for this family structure
                    results = quant_estimate.gridsearch_minimize(n_dep_param=n_dep_param,
                                                                 n_input_sample=n_input_sample, 
                                                                 grid_type=grid_type, 
                                                                 q_func=q_func,
                                                                 done_results=all_results)

                    all_params = np.r_[all_params, results.full_dep_params]
                    
                    # How much does it costs
                    cost += results.n_evals

                    if re_use_params:
                        # Save the results
                        all_results.extend(results)

                    # Save the minimum
                    if not with_bootstrap:
                        min_quantity[i, j] = results.min_quantity
                    else:
                        assert isinstance(with_bootstrap, int), "Must be a number"
                        n_bootstrap = with_bootstrap
                        results.compute_bootstraps(n_bootstrap)
                        print(results.bootstrap_samples.mean(axis=1))
                        min_quantity[i, j] = results[results.bootstrap_samples.mean(axis=1).argmin()]

                    if verbose:
                        print('Worst quantile of {0} at {1}'.format(selected_pairs + [(i, j)], min_quantity[i, j]))
                pair_id += 1
        # Get the min from the iterations
        sorted_quantities = sorted(min_quantity.items(), key=lambda x: x[1])
        if (n_remove_pairs > 0) and (n_remove_pairs < len(sorted_quantities)-1):
            removed_pairs.extend([pair[0] for pair in sorted_quantities[-n_remove_pairs:]])

        selected_pair = sorted_quantities[0][0]
        for pair in sorted_quantities[:n_add_pairs]:
            i, j = pair[0][0], pair[0][1]
            families[i, j] = init_family[i, j]
        selected_pairs.extend([pair[0] for pair in sorted_quantities[:n_add_pairs]])
        worst_quantities.append(min_quantity[selected_pair])
        if verbose:
            print('p=%d, worst quantile at %.2f, cost = %d' % (p+1, min_quantity[selected_pair], cost))

        p += n_add_pairs
        if n_dep_param is not None:
            n_dep_param = n_dep_param_init*int(np.sqrt(p+1))

    return worst_quantities, selected_pairs, removed_pairs


def check_structure_shape(structure):
    """Check if the structure shape is correct.
    """
    assert structure.shape[0] == structure.shape[1], "Structure matrix should be squared"
    assert np.triu(structure, k=1).sum() == 0, "Matrix should be lower triangular"

def get_pair(dim, index, with_plus=True):
    """ Get the pair of variables from a given index.
    """
    k = 0
    for i in range(1, dim):
        for j in range(i):
            if k == index:
                if with_plus:
                    return [i+1, j+1]
                else:
                    return [i, j]
            k+=1

def get_pair_id(dim, pair, with_plus=True):
    """ Get the pair of variables from a given index.
    """
    k = 0
    for i in range(1, dim):
        for j in range(i):
            if with_plus:
                if (pair[0] == i+1) & (pair[1] == j+1):
                    return k
            else:
                if (pair[0] == i) & (pair[1] == j):
                    return k
            k+=1

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


def add_pair(structure, pair, index, lvl):
    """Adds a pair in a Vine structure in a certain place and for a specific conditionement.
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
            "First element should be the same as the first variable of the pair"
        assert structure[dim-1-lvl, index] == 0, \
            "There is already a variable at [%d, %d]" % (dim-1, index)
        structure[dim-1-lvl, index] = pair[1]
    return structure


def add_pairs(structure, pairs, lvl, verbose=False):
    """Add pairs in a structure for a selected level of conditionement.
    """
    dim = structure.shape[0]
    n_pairs = len(pairs)
    assert n_pairs < dim - lvl, "Not enough place to fill the pairs"
    n_slots = dim - 1 - lvl
    possibilities = list(itertools.permutations(range(n_slots), r=n_pairs))
    success = False
    init_structure = np.copy(structure)
    for possibility in possibilities:
        try:
            # Add the pair in the possible order
            structure = np.copy(init_structure)
            for i in range(n_pairs):
                structure = add_pair(structure, pairs[i], possibility[i], lvl)
            if check_redundancy(structure):
                success = True
                break
        except AssertionError:
            pass

    if not success and verbose:
        print('Did not succeded to fill the structure with the given pairs')

    # If it's the 1st level, the last row of last column must be filled
    if (lvl == 0) and (n_pairs == dim-1):
        structure[dim-1, dim-1] = np.setdiff1d(range(dim+1), np.diag(structure))[0]
    return structure


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
    
    diag_possibility = list(itertools.permutations(remaining_vals, n_remaining_vals))[0]

    
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
        remaining_i = range(1, dim + 1)
        for var in used_values_i:
            if (var in remaining_i):
                remaining_i.remove(var)
        values_i[values_i == 0] = remaining_i
        prev_ind.append(values_i[0])
    return structure

    
def check_node_loop(pairs, n_p=3):
    """Check if not too many variables are connected in a single tree
    """
    for perm_pairs in list(itertools.permutations(pairs, r=n_p)):
        if len(np.unique(perm_pairs)) <= n_p:
            return False
    return True


def get_pairs_by_levels(dim, forced_pairs_ids, verbose=False):
    """Given a list of sorted pairs, this gives the pairs for the different levels o    f
    conditionement.
    """
    n_pairs = dim*(dim-1)/2
    n_forced_pairs = len(forced_pairs_ids)
    assert len(np.unique(forced_pairs_ids)) == n_forced_pairs, "Unique values should be puted"
    assert n_forced_pairs <= n_pairs, "Not OK!"
    assert max(forced_pairs_ids) < n_pairs, "Not OK!"
    
    n_conditionned = range(dim - 1, 0, -1)
    forced_pairs = np.asarray([get_pair(dim, pair_id) for pair_id in forced_pairs_ids])
    remaining_pairs_ids = range(0, n_pairs)
    for pair_id in forced_pairs_ids:
        remaining_pairs_ids.remove(pair_id)
    remaining_pairs = np.asarray([get_pair(dim, pair_id) for pair_id in remaining_pairs_ids])
    
    if verbose:
        print('Vine dimension: %d' % dim)
        print('Conditioning information:')
    k0 = 0
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
    
def get_possible_structures(dim, pairs_by_levels, verbose=False):
    """
    """
    # For each levels
    good_structures = []
    for lvl, pairs_level in enumerate(pairs_by_levels):
        n_pairs_level = len(pairs_level) # Number of pairs in the level
        
        # The possible combinations
        combinations = list(itertools.product([1, -1], repeat=n_pairs_level))
        
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