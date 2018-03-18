import numpy as np

from numpy.testing import assert_array_equal

from depimpact.utils import matrix_to_list, list_to_matrix, get_pairs_by_levels


def test_matrix_to_list():
    matrix = [[0, 0, 0, 0, 0],
              [2, 0, 0, 0, 0],
              [4, 0, 0, 0, 0],
              [7, 9, 8, 0, 0],
              [0, 6, 3, 1, 0]]
    
    matrix = np.asarray(matrix)
    true_list_sup = [2, 4, 7, 9, 8, 6, 3, 1]
    true_list_ids_sup = [0, 1, 3, 4, 5, 7, 8, 9]
    true_list_coor_sup = [[1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [4, 1], [4, 2], [4, 3]]

    true_list_sup_eq = [2, 4, 0, 7, 9, 8, 0, 6, 3, 1]
    true_list_ids_sup_eq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    true_list_coord_sup_eq = [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [4, 2], [4, 3]]
    
    values_sup, ids, coord = matrix_to_list(matrix, return_ids=True, return_coord=True, op_char='>')
    assert_array_equal(values_sup, true_list_sup)
    assert_array_equal(ids, true_list_ids_sup)
    assert_array_equal(coord, true_list_coor_sup)
    
    values_sup_eq, ids, coord = matrix_to_list(matrix, return_ids=True, return_coord=True, op_char='>=')
    assert_array_equal(values_sup_eq, true_list_sup_eq)
    assert_array_equal(ids, true_list_ids_sup_eq)
    assert_array_equal(coord, true_list_coord_sup_eq)
    
    assert_array_equal(list_to_matrix(values_sup_eq, 5), matrix)  
    
    
def test_group_pairs():
    forced_pairs_ids = [4, 5, 2, 0, 3]
    for dim in range(4, 5):
        pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
        assert pairs_by_levels == [[(4, 2), (4, 3), (2, 1)], [(3, 2), (4, 1)]], "Not the result we wanted"
    for dim in range(6, 10):
        pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
        assert pairs_by_levels == [[(4, 2), (4, 3), (2, 1)], [(4, 1), (3, 2)]], "Not the result we wanted"
    
    forced_pairs_ids = [11, 12,  8,  4,  2,  5,  7, 14, 13,  1,  3,  6,  0, 10]
    dim = 6
    pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
    assert pairs_by_levels == [[(6, 2), (6, 3), (5, 3), (4, 2)], [(4, 3), (5, 2), (6, 5), (6, 4)], [(3, 1), (4, 1), (5, 1)], [(2, 1), (6, 1)]], "Not the result we wanted"
    dim = 7
    pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
    assert pairs_by_levels == [[(6, 2), (6, 3), (5, 3), (4, 2)], [(5, 2), (6, 5), (6, 4), (3, 1), (4, 1)], [(5, 1), (2, 1), (6, 1)]], "Not the result we wanted"
    dim = 10
    pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
    assert pairs_by_levels == [[(6, 2), (6, 3), (5, 3), (4, 2)], [(3, 1), (4, 1), (5, 1), (2, 1), (6, 1)]], "Not the result we wanted"

    forced_pairs_ids = [14,  8, 27, 23, 20, 10,  1, 19,  7, 11, 12, 13, 25, 15, 18,  4,  2,
         6, 22, 16,  5, 21, 24, 26, 17,  9,  3]
    
    dim = 10
    pairs_by_levels = get_pairs_by_levels(dim, forced_pairs_ids, verbose=False)
    assert pairs_by_levels == [[(6, 5), (5, 3), (8, 7), (8, 3), (6, 1)], [(6, 2), (6, 3), (6, 4), (8, 5), (7, 1), (7, 4)], [(5, 1), (8, 2), (7, 2), (4, 3), (8, 1), (8, 4), (8, 6)], [(7, 3), (5, 4), (4, 1)]], "Not the result we wanted"
