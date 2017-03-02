import numpy as np
import copy

from .dependence import ConservativeEstimate, ListDependenceResult

GRIDS = ['lhs', 'rand']

def iterative_vine_minimize(estimate_object, n_input_sample, n_dep_param_init, p_max, grid_type='lhs', 
                            q_func=np.var, n_add_pairs=1, n_remove_pairs=1, 
                            with_bootstrap=False, re_use_params=False, verbose=False):
    """
    """
    
    quant_estimate = copy.copy(estimate_object)
    max_n_pairs = quant_estimate.corr_dim
    
#    assert isinstance(quant_estimate, ConservativeEstimate), \
#        "Object must be from the ConservativeEstimate class"
        
    assert grid_type in GRIDS, "Unknow Grid type {0}".format(grid_type)
    
    assert 0 < p_max < max_n_pairs, "Maximum number of pairs must be positive"
    assert 1 < n_add_pairs < max_n_pairs, "Must add at least one pair at each iteration"
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
        for i in range(1, dim):
            for j in range(i):
                if ((i, j) not in selected_pairs) and ((i, j) not in removed_pairs):
                    # Family matrix for this iteration
                    tmp_families = np.copy(families)
                    tmp_families[i, j] = init_family[i, j]
                    
                    # Family matrix is changed
                    quant_estimate.families = tmp_families
                    
                    # Lets get the results for this family structure
                    results = quant_estimate.gridsearch_minimize(n_dep_param=n_dep_param,
                                                                 n_input_sample=n_input_sample, 
                                                                 grid_type=grid_type, 
                                                                 q_func=q_func,
                                                                 done_results=all_results)
                    
                    all_params = np.r_[all_params, results.full_dep_params]
                    # How much does it costs
                    cost += results.n_evals
                    
                    # Save the results                
                    all_results.extend(results)
                    
                    # Save the minimum
                    if not with_bootstrap:
                        min_quantity[i, j] = results.min_quantity
                    else:
                        assert isinstance(with_bootstrap, int), "Must be a number"
                        n_bootstrap = with_bootstrap                        
                        results.compute_bootstraps(n_bootstrap)
                        print results.bootstrap_samples.mean(axis=1)
                        min_quantity[i, j] = results[results.bootstrap_samples.mean(axis=1).argmin()]
                        
                    if verbose:
                        print('Worst quantile of {0} at {1}'.format(selected_pairs + [(i, j)], min_quantity[i, j]))

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