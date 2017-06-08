import numpy as np

from numpy.testing import assert_array_equal

from dependence.utils import matrix_to_list, list_to_matrix


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
    
    
    
    