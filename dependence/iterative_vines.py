import numpy as np

def iterative_vine_minimize(quant_estimate, n, K_init, p_max, grid_type='lhs', 
                            q_func=np.var, n_add_pairs=1, n_remove_pairs=1, verbose=False):
    
    assert 1 < n_add_pairs, "Must add at least one pair at each iteration"
    assert 0 <= n_remove_pairs, "This cannot be negative"
    
    init_family = quant_estimate.families
    dim = quant_estimate.input_dim
    
    families = np.zeros((dim, dim))
    selected_pairs = []
    removed_pairs = []
    worst_quantities = []

    cost = 0
    p = 0
    K = K_init    
    while p < p_max:
        quantities = {}
        for i in range(1, dim):
            for j in range(i):
                if ((i, j) not in selected_pairs) and ((i, j) not in removed_pairs):
                    tmp_families = np.copy(families)
                    tmp_families[i, j] = init_family[i, j]
                    quant_estimate.families = tmp_families
                    results = quant_estimate.gridsearch_minimize(n_dep_param=K, 
                                                                 n_input_sample=n, 
                                                                 grid_type=grid_type, 
                                                                 q_func=q_func)
                    if K is None:
                        cost += ((3**(p+1))-1)*n
                    else:
                        cost += K*n

                    quantities[i, j] = results.min_quantity
                        
                    if verbose:
                        print('Worst quantile of {0} at {1}'.format(selected_pairs + [(i, j)], quantities[i, j]))

        sorted_quantities = sorted(quantities.items(), key=lambda x: x[1])
        if (n_remove_pairs > 0) and (n_remove_pairs < len(sorted_quantities)) :
            removed_pairs.extend([pair[0] for pair in sorted_quantities[-n_remove_pairs:]])
        
        selected_pair = sorted_quantities[0][0]
        i, j = selected_pair[0], selected_pair[1]
        families[i, j] = init_family[i, j]
        selected_pairs.extend([pair[0] for pair in sorted_quantities[:n_add_pairs]])
        worst_quantities.append(quantities[selected_pair])
        if verbose:
            print('p=%d, worst quantile at %.2f, cost = %d' % (p+1, quantities[selected_pair], cost))
            
        p += n_add_pairs
        if K is not None:
            K = K_init*int(np.sqrt(p+1))
        
    return worst_quantities, selected_pairs, removed_pairs