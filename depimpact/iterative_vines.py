import copy
import os

import numpy as np

from .conservative import ListDependenceResult
from .utils import get_pair_id, get_pairs_by_levels, get_possible_structures

GRIDS = ['lhs', 'rand', 'vertices']
LIB_PARAMS = ['iterative_save', 'iterative_load', 'input_names',
              'output_names', 'keep_input_samples', 'load_input_samples',
              'use_grid', 'save_grid', 'grid_path', 'n_pairs_start']

# TODO: add the function as a method in ConservativeEstimate


def create_dir(folder):
    folder = os.path.abspath(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_result_dir(condition, default_dir='./iterative_result'):
    if condition is True:
        condition = create_dir(default_dir)
    elif condition is False:
        condition = None
    elif isinstance(condition, str):
        condition = create_dir(condition)
    else:
        raise TypeError(
            "Wrong type for iterative_save: {}".format(type(condition)))
    return condition


def get_grid_size_func(n_dep_param_init):
    # Grid-size evolution
    if callable(n_dep_param_init):
        n_param_iter = n_dep_param_init
    elif n_dep_param_init is None:
        # If vertices
        def n_param_iter(x): return None
    else:
        def n_param_iter(k): return int(n_dep_param_init*(k+1)**2)
    return n_param_iter


# TODO: do something with the large number of parameters
def iterative_vine_minimize(
        estimator,
        n_input_sample=1000,
        n_dep_param_init=20,
        max_n_pairs=5,
        grid_type='lhs',
        q_func=np.var,
        n_add_pairs=1,
        n_remove_pairs=0,
        adapt_vine_structure=True,
        delta=0.1,
        list_families=None,
        with_bootstrap=False,
        verbose=False,
        **kwargs):
    """Iteratively minimises the output quantity of interest.

    Parameters
    ----------


    Returns
    -------
    """
    quant_estimate = copy.copy(estimator)
    corr_dim = quant_estimate.corr_dim
    dim = quant_estimate.input_dim

    # Check the function parameters
    assert grid_type in GRIDS, "Unknow grid type {0}".format(grid_type)
    assert 0 < max_n_pairs <= corr_dim, "max_n_pairs not in (0, corr_dim]"
    assert 1 <= n_add_pairs <= corr_dim, "n_add_pairs not in [1, corr_dim]"
    assert 0 <= n_remove_pairs < corr_dim, "n_remove_pairs not in [0, corr_dim)"
    assert callable(q_func), "Quantity function is not callable"

    # Check if the given parameters are known
    for lib_param in kwargs:
        assert lib_param in LIB_PARAMS, "Unknow parameter %s" % (lib_param)

    # Iterative save of the results
    iterative_save = False
    if 'iterative_save' in kwargs:
        iterative_save = kwargs['iterative_save']
        if iterative_save and (n_input_sample == 0):
            iterative_save = False
        save_dir = get_result_dir(iterative_save)

    # Iterative load of the results
    iterative_load = False
    if 'iterative_load' in kwargs:
        iterative_load = kwargs['iterative_load']
        load_dir = get_result_dir(iterative_load)

    input_names = []
    if 'input_names' in kwargs:
        input_names = kwargs['input_names']

    output_names = []
    if 'output_names' in kwargs:
        output_names = kwargs['output_names']

    keep_input_samples = True
    if 'keep_input_samples' in kwargs:
        keep_input_samples = kwargs['keep_input_samples']

    load_input_samples = True
    if 'load_input_samples' in kwargs:
        load_input_samples = kwargs['load_input_samples']

    use_grid = None
    if 'use_grid' in kwargs:
        use_grid = kwargs['use_grid']

    save_grid = None
    if 'save_grid' in kwargs:
        save_grid = kwargs['save_grid']

    n_pairs_start = 0
    if 'n_pairs_start' in kwargs:
        n_pairs_start = kwargs['n_pairs_start']

    # TODO: do something smarter :/
    # The list of families to test
    if list_families is None:
        list_families = []
        multiple_families = False
    else:
        assert isinstance(
            list_families, list), "This should be a list of copula families"
        multiple_families = True

    n_param_iter = get_grid_size_func(n_dep_param_init)
    n_dep_param = n_param_iter(0)

    # The results
    iterative_result = IterativeDependenceResults(dim)

    # Initial configurations
    init_family = quant_estimate.families.copy()
    init_bounds_tau = quant_estimate.bounds_tau.copy()

    # New empty configurations
    families = np.zeros((dim, dim), dtype=int)
    bounds_tau = np.zeros((dim, dim))
    bounds_tau[:] = None

    # The candidate pairs
    candidate_pairs = np.asarray(np.tril_indices(dim, k=-1)).T.tolist()

    # Fixed and independent pairs
    fixed_params = quant_estimate.fixed_params.copy()
    indep_pairs = quant_estimate._indep_pairs[:]
    fixed_pairs = quant_estimate._fixed_pairs[:]
    other_pairs = indep_pairs + fixed_pairs

    # Remove fixed pairs from the list and add in the family matrix
    for pair in fixed_pairs:
        candidate_pairs.remove(pair)
        families[pair[0], pair[1]] = init_family[pair[0], pair[1]]

    # Remove independent pairs from candidate
    for pair in indep_pairs:
        candidate_pairs.remove(pair)

    # The candidate pairs
    candidate_pairs = np.asarray(np.tril_indices(dim, k=-1)).T.tolist()

    selected_pairs = []
    # Algorithm Loop
    n_evals = 0
    n_pairs = 1
    iteration = 0
    min_quant_iter = []
    stop_conditions = False

    while not stop_conditions:
        min_quantities_iter = {}
        best_families = np.zeros((dim, dim), dtype=int)

        # The candidates
        for i, j in candidate_pairs:
            # Update the bound matrix
            tmp_bounds_tau = bounds_tau.copy()
            tmp_bounds_tau[i, j] = init_bounds_tau[i, j]
            tmp_bounds_tau[j, i] = init_bounds_tau[j, i]

            # The algorithm can iterate through several families for each candidate
            if not multiple_families:
                list_families = [init_family[i, j]]

            min_quantities_families = {}
            # For each copula family
            for cop_id in list_families:
                # update the family matrix
                tmp_families = families.copy()
                tmp_families[i, j] = cop_id

                # Adapt the vine structure matrix
                if adapt_vine_structure:
                    pairs_iter = other_pairs + selected_pairs + [(i, j)]
                    pairs_iter_id = [get_pair_id(
                        dim, pair, with_plus=False) for pair in pairs_iter]
                    pairs_by_levels = get_pairs_by_levels(dim, pairs_iter_id)
                    quant_estimate.vine_structure = get_possible_structures(dim, pairs_by_levels)[
                        0]

                # Family matrix is changed
                quant_estimate.bounds_tau = tmp_bounds_tau
                quant_estimate.families = tmp_families
                quant_estimate.fixed_params = fixed_params

                # Lets get the results for this family structure
                if n_input_sample > 0 and n_pairs >= n_pairs_start:
                    results = quant_estimate.gridsearch(n_dep_param=n_dep_param,
                                                        n_input_sample=n_input_sample,
                                                        grid_type=grid_type,
                                                        keep_input_samples=keep_input_samples,
                                                        load_grid=use_grid,
                                                        save_grid=save_grid,
                                                        use_sto_func=True)
                    results.q_func = q_func

                # TODO: correct save and loading
                # Save iteratively
                if iterative_save or iterative_load:
                    cop_str = "_".join([str(l)
                                        for l in quant_estimate._family_list])
                    vine_str = "_".join(
                        [str(l) for l in quant_estimate._vine_structure_list])
                    filename = "%s/%s" % (load_dir, grid_type)
                    if n_dep_param is None:
                        filename += "_K_None"
                    else:
                        filename += "_K_%d" % (n_dep_param)
                    filename += "_cop_%s_vine_%s.hdf5" % (cop_str, vine_str)

                if iterative_save and n_pairs >= n_pairs_start:
                    results.to_hdf(filename, input_names, output_names,
                                   with_input_sample=keep_input_samples)

                if iterative_load:
                    name, extension = os.path.splitext(filename)
                    condition = os.path.exists(filename)
                    k = 0
                    while condition:
                        try:
                            load_result = ListDependenceResult.from_hdf(
                                filename, with_input_sample=load_input_samples, q_func=q_func)
                            # TODO: create a function to check the configurations of two results
                            # TODO: is the testing necessary? If the saving worked, the loading should be ok.
                            np.testing.assert_equal(
                                load_result.families, tmp_families, err_msg="Not good family")
                            np.testing.assert_equal(
                                load_result.bounds_tau, tmp_bounds_tau, err_msg="Not good Bounds")
                            np.testing.assert_equal(
                                load_result.vine_structure, quant_estimate.vine_structure, err_msg="Not good structure")
                            condition = False
                        except AssertionError:
                            filename = '%s_num_%d%s' % (name, k, extension)
                            condition = os.path.exists(filename)
                            k += 1
                    # Replace the actual results with the loaded results (this results + all the previous saved ones)
                    results = load_result

                tmp_quantity = results.min_quantity
                min_quantities_families[cop_id] = tmp_quantity
                n_evals += results.n_evals

                if verbose:
                    print("Quantity: {:3f} for copula {}".format(
                        tmp_quantity, cop_id))

            # Minimum among the copula families
            best_family = min(min_quantities_families,
                              key=min_quantities_families.get)
            min_quantity = min_quantities_families[best_family]

            best_families[i, j] = best_family

            # Save the minimum
            if not with_bootstrap:
                min_quantities_iter[i, j] = min_quantity
            else:
                assert isinstance(with_bootstrap, int), "Must be a number"
                n_bootstrap = with_bootstrap
                results.compute_bootstraps(n_bootstrap)
                print(results.bootstrap_samples.mean(axis=1))
                min_quantities_iter[i, j] = results[results.bootstrap_samples.mean(
                    axis=1).argmin()]

            if verbose:
                tmp = "n={}, K={}. Best family: {}. Worst quantile of {} at {:5f}"
                tmp = tmp.format(results.n_input_sample, n_dep_param,
                              best_family, selected_pairs + [(i, j)], min_quantity)
                print(tmp)
                if input_names:
                    pair_names = ["%s-%s" % (input_names[k1], input_names[k2])
                                  for k1, k2 in selected_pairs + [(i, j)]]
                    print("The variables are: " + " ".join(pair_names))

            # Store the result
            iterative_result[iteration, i, j] = results

        # Get the min from the iterations
        sorted_quantities = sorted(min_quantities_iter.items(),
                                   key=lambda x: x[1])
        # Delay of the first iteration
        if iteration == 0:
            delta_q_init = abs(
                sorted_quantities[0][1] - sorted_quantities[-1][1])

        min_quant_iter.append(sorted_quantities[0][1])

        if (n_remove_pairs > 0) and (n_remove_pairs < len(sorted_quantities)-1):
            # The pairs to remove
            for pair in sorted_quantities[-n_remove_pairs:]:
                candidate_pairs.remove(list(pair[0]))

        selected_pair = sorted_quantities[0][0]
        # Selected pairs to add
        for pair in sorted_quantities[:n_add_pairs]:
            i, j = pair[0][0], pair[0][1]
            families[i, j] = best_families[i, j]
            bounds_tau[i, j] = init_bounds_tau[i, j]
            bounds_tau[j, i] = init_bounds_tau[j, i]
            candidate_pairs.remove(list(pair[0]))
            selected_pairs.append(pair[0])

        iterative_result.selected_pairs.append(selected_pairs)
        if verbose:
            i, j = selected_pair
            tmp = '\nIteration {}: selected pair: {}, selected family: {}'.format(
                iteration+1, selected_pair, best_families[i, j])
            if input_names:
                tmp += " (" + "-".join(input_names[k]
                                       for k in selected_pair) + ")"
            print(tmp)
            print('Total number of evaluations = %d. Minimum quantity at %.2f.\n' % (
                n_evals, min_quantities_iter[selected_pair]))

        # Stop conditions
        if n_pairs >= max_n_pairs:
            stop_conditions = True
            print('Max number of pairs reached')

        if iteration > 0:
            delta_q = -(min_quant_iter[-1] - min_quant_iter[-2])
            if delta_q <= delta*delta_q_init:
                stop_conditions = True
                print('Minimum_variation not fulfiled: %.2f <= %0.2f' %
                      (delta_q, delta*delta_q_init))

        n_pairs += n_add_pairs
        if n_dep_param is not None:
            n_dep_param = n_param_iter(iteration+1)

        if not stop_conditions:
            iterative_result.new_iteration()

        iteration += 1

    iterative_result.n_evals = n_evals    
    return iterative_result


class IterativeDependenceResults(object):
    """

    """
    def __init__(self, dim):
        self.iteration = 0
        n_pairs = int(dim * (dim-1) / 2)
        self.results = [[]]
        tmp = np.zeros((dim, dim), dtype=object)
        tmp[:] == None
        self.results[self.iteration] = tmp
        self.selected_pairs = [[]]

        self.dim = dim
        self.n_pairs = n_pairs
        self.n_evals = 0

    def new_iteration(self):
        """
        """
        self.iteration += 1
        tmp = np.zeros((self.dim, self.dim), dtype=object)
        tmp[:] = None
        self.results.append(tmp)

    def __getitem__(self, item):
        """
        """
        iteration, i, j = item
        return self.results[iteration][i, j]

    def __setitem__(self, item, result):
        """
        """
        iteration, i, j = item
        self.results[iteration][i, j] = result

    def min_quantities(self, iteration):
        """
        """
        results = self.results[iteration]
        dim = self.dim
        min_quantities = np.zeros((dim, dim), dtype=np.float)
        for i in range(1, dim):
            for j in range(i):
                if results[i, j] is not None:
                    min_quantities[i, j] = results[i, j].min_quantity

        return min_quantities

    def min_results(self, iteration):
        """
        """
        results = self.results[iteration]
        dim = self.dim
        min_results = np.zeros((dim, dim), dtype=object)
        for i in range(1, dim):
            for j in range(i):
                if results[i, j] is not None:
                    min_results[i, j] = results[i, j].min_result

        return min_results

    def min_quantity(self, iteration):
        """
        """
        min_quantities = self.min_quantities(iteration)
        min_quantity = min_quantities.min()
        return min_quantity

    def min_result(self, iteration):
        """
        """
        min_quantities = self.min_quantities(iteration)
        id_min = min_quantities.argmin()
        min_result = self.min_results(iteration).item(id_min)
        return min_result
