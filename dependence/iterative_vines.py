import numpy as np
import os
import copy

from .dependence import ListDependenceResult
from .utils import get_pair_id, get_pairs_by_levels, get_possible_structures

GRIDS = ['lhs', 'rand', 'vertices']
LIB_PARAMS = ['iterative_save', 'iterative_load', 'input_names', 
              'output_names', 'keep_input_samples', 'load_input_samples',
              'use_grid', 'save_grid', 'grid_path', 'n_pairs_start']


def iterative_vine_minimize(estimate_object, n_input_sample=1000, n_dep_param_init=20, max_n_pairs=5, grid_type='lhs', 
                            q_func=np.var, n_add_pairs=1, n_remove_pairs=1, adapt_vine_structure=True, delta=0.1,
                            with_bootstrap=False, verbose=False, **kwargs):
    """Iteratively minimises the output quantity of interest.

    Parameters
    ----------


    Returns
    -------
    """
    quant_estimate = copy.copy(estimate_object)
    corr_dim = quant_estimate.corr_dim_
    dim = quant_estimate.input_dim
    
    max_n_pairs = min(max_n_pairs, corr_dim)
    
    assert grid_type in GRIDS, "Unknow Grid type {0}".format(grid_type)
    assert 0 < max_n_pairs < corr_dim, "Maximum number of pairs must be positive"
    assert 1 <= n_add_pairs < corr_dim, "Must add at least one pair at each iteration"
    assert 0 <= n_remove_pairs < corr_dim, "This cannot be negative"
    assert callable(q_func), "Quantity function must be callable"
    
    # Initial configurations
    init_family = quant_estimate.families
    init_bounds_tau = quant_estimate.bounds_tau
    fixed_params = quant_estimate.fixed_params.copy()
    init_indep_pairs = quant_estimate._indep_pairs[:]
    init_fixed_pairs = quant_estimate._fixed_pairs[:]
    
    # New empty configurations
    families = np.zeros((dim, dim))
    bounds_tau = np.zeros((dim, dim))
    bounds_tau[:] = np.nan

    # Selected pairs through iterations
    selected_pairs = []
    all_results = []

    n_dep_param = n_dep_param_init
    
    # The pairs to do at each iterations
    indices = np.asarray(np.tril_indices(dim, k=-1)).T.tolist()
    
    # Remove fixed pairs from the list and add in the family matrix
    for pair in init_fixed_pairs:
        indices.remove(pair)
        families[pair[0], pair[1]] = init_family[pair[0], pair[1]]
        
    # Remove independent pairs
    for pair in init_indep_pairs:
        indices.remove(pair)
    
    # Check if the given parameters are known
    for lib_param in kwargs:
        assert lib_param in LIB_PARAMS, "Unknow parameter %s" % (lib_param)
        
    iterative_save = False
    if 'iterative_save' in kwargs:
        iterative_save = kwargs['iterative_save']
        if iterative_save is True:
            save_dir = './iterative_result'
        elif isinstance(iterative_save, str):
            save_dir = os.path.abspath(iterative_save)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        elif iterative_save is False:
            pass
        else:
            raise TypeError("Wrong type for iterative_save: {0}".format(type(iterative_save)))
        
    iterative_load = False
    if 'iterative_load' in kwargs:
        iterative_load = kwargs['iterative_load']
        if iterative_load is True:
            load_dir = './iterative_result'
        elif isinstance(iterative_load, str):
            load_dir = os.path.abspath(iterative_load)
            if not os.path.exists(load_dir):
                print("Directory %s does not exists" % (load_dir))
        elif iterative_load is False:
            pass
        else:
            raise TypeError("Wrong type for iterative_load: {0}".format(type(iterative_load)))
            
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
        
    grid_path = '.'
    if 'grid_path' in kwargs:
        grid_path = kwargs['grid_path']
        
    n_pairs_start = 0
    if 'n_pairs_start' in kwargs:
        n_pairs_start = kwargs['n_pairs_start']

    if n_input_sample == 0:
        iterative_save = None
        
    ## Algorithm Loop
    cost = 0
    n_pairs = 1
    iteration = 0
    min_quant_iter = []
    stop_conditions = False
    while not stop_conditions:
        min_quantity = {}
        all_results.append({})
        for i, j in indices:
            # Family matrix for this iteration
            tmp_families = families.copy()
            tmp_families[i, j] = init_family[i, j]
            tmp_bounds_tau = bounds_tau.copy()
            tmp_bounds_tau[i, j] = init_bounds_tau[i, j]
            tmp_bounds_tau[j, i] = init_bounds_tau[j, i]
            
            # Adapt the vine structure matrix
            if adapt_vine_structure:
                pairs_iter = init_indep_pairs + init_fixed_pairs + selected_pairs + [(i, j)]
                pairs_iter_id = [get_pair_id(dim, pair, with_plus=False) for pair in pairs_iter]
                pairs_by_levels = get_pairs_by_levels(dim, pairs_iter_id)
                quant_estimate.vine_structure = get_possible_structures(dim, pairs_by_levels)[0]
            
            # Family matrix is changed
            quant_estimate.families = tmp_families
            quant_estimate.fixed_params = fixed_params
            quant_estimate.bounds_tau = tmp_bounds_tau
            

            # Lets get the results for this family structure
            if n_input_sample > 0 and n_pairs >= n_pairs_start:
                results = quant_estimate.gridsearch(n_dep_param=n_dep_param,
                                                             n_input_sample=n_input_sample,
                                                             grid_type=grid_type,
                                                             q_func=q_func,
                                                             keep_input_samples=keep_input_samples,
                                                             use_grid=use_grid,
                                                             save_grid=save_grid,
                                                             grid_path=grid_path)
            
            if iterative_save or iterative_load:
                cop_str = "_".join([str(l) for l in quant_estimate._family_list])
                vine_str = "_".join([str(l) for l in quant_estimate._vine_structure_list])
                filename = "%s/%s" % (load_dir, grid_type)
                if n_dep_param is None:
                    filename += "_K_None"
                else:
                    filename += "_K_%d" % (n_dep_param)
                filename += "_cop_%s_vine_%s.hdf5" % (cop_str, vine_str)

            if iterative_save and n_pairs >= n_pairs_start:
                results.to_hdf(filename, input_names, output_names, with_input_sample=keep_input_samples)

            if iterative_load :
                name, extension = os.path.splitext(filename)
                condition = os.path.exists(filename)
                k = 0
                while condition:
                    try:
                        load_result = ListDependenceResult.from_hdf(filename, with_input_sample=load_input_samples, q_func=q_func)
                        # TODO: create a function to check the configurations of two results
                        # TODO: is the testing necessary? If the saving worked, the loading should be ok.
                        np.testing.assert_equal(load_result.families, tmp_families, err_msg="Not good family")
                        np.testing.assert_equal(load_result.bounds_tau, tmp_bounds_tau, err_msg="Not good Bounds")
                        np.testing.assert_equal(load_result.vine_structure, quant_estimate.vine_structure, err_msg="Not good structure")
                        condition = False
                    except AssertionError:
                        filename = '%s_num_%d%s' % (name, k, extension)
                        condition = os.path.exists(filename)
                        k += 1
                # Replace the actual results with the loaded results (this results + all the previous saved ones)
                results = load_result
                
            # Name if the result dictionary
            result_name = str(selected_pairs + [(i, j)])[1:-1]
            all_results[iteration][result_name] = results
            
            # How much does it costs
            cost += results.n_evals

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
                print('n={0}. Worst quantile of {1} at {2}'.format(results.n_input_sample, selected_pairs + [(i, j)], min_quantity[i, j]))
                if input_names:
                    pair_names = [ "%s-%s" % (input_names[k1], input_names[k2]) for k1, k2 in selected_pairs + [(i, j)]]
                    print("The variables are: " + " ".join(pair_names))
        
        # Get the min from the iterations
        sorted_quantities = sorted(min_quantity.items(), key=lambda x: x[1])
        
        # Delay of the first iteration
        if iteration == 0:
            delta_q_init = abs(sorted_quantities[0][1] - sorted_quantities[-1][1])
            
        min_quant_iter.append(sorted_quantities[0][1])
        
        if (n_remove_pairs > 0) and (n_remove_pairs < len(sorted_quantities)-1):
            # The pairs to remove
            for pair in sorted_quantities[-n_remove_pairs:]:
                indices.remove(list(pair[0]))

        selected_pair = sorted_quantities[0][0]
        # Selected pairs to add 
        for pair in sorted_quantities[:n_add_pairs]:
            i, j = pair[0][0], pair[0][1]
            families[i, j] = init_family[i, j]
            bounds_tau[i, j] = init_bounds_tau[i, j]
            bounds_tau[j, i] = init_bounds_tau[j, i]
            indices.remove(list(pair[0]))
            selected_pairs.append(pair[0])
            
        if True:
            k1, k2 = selected_pair
            tmp = '\nIteration {0}: selected pair: {1}'.format(iteration+1, selected_pair)
            if input_names:
                tmp += " (" + "-".join(input_names[k] for k in selected_pair) + ")"
            print(tmp)
            print('Total number of evaluations = %d. Minimum quantity at %.2f.\n' % (cost, min_quantity[selected_pair]))
            

        # Stop conditions
        if n_pairs >= max_n_pairs:
            stop_conditions = True
            print('Max number of pairs reached')
            
        if iteration > 0:
            delta_q = abs(min_quant_iter[-1] - min_quant_iter[-2])
            if delta_q <= delta*delta_q_init:
                stop_conditions = True
                print('Minimum_variation not fulfiled: %.2f <= %0.2f' % (delta_q, delta*delta_q_init))
            
        n_pairs += n_add_pairs
        if n_dep_param is not None:
            n_dep_param = n_dep_param_init*int(np.sqrt(n_pairs))
            
        iteration += 1
    return all_results

