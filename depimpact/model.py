import numpy as np

class Tree(object):
    def __init__(self, dim, rank):
        self.dim = dim
        self.rank = rank
        self.nodes = []
        self.edges = []
        
def check_structure(matrix):
    """Check if a given matrix is a vine array and should respect the conditions a regular vine. 
    
    Parameters:
    ----------
    matrix : {array}
        A vine array.
    """
    # Convert the matrix in np.array if it is not
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)

    dim = matrix.shape[0]
    assert dim == matrix.shape[1], "The matrix should be symetric"
    return matrix

def init_trees(dim):
    """Initialize the tree objects of a vine array.
        
    Parameters:
    ----------
    dim : {int}
        The problem dimension.
    """
    trees = []
    for i in range(dim-1):
        tree = Tree(dim=dim, rank=i)
        trees.append(tree)

    return trees
class Vine(object):
    """Describe a vine structure
    
    Parameters:
    ----------
    structure : {array}
        The vine array describing the structure
    """
    def __init__(self, structure):
        self.structure = check_structure(structure)
        self.dim = structure.shape[0]
        self.trees = init_trees(self.dim)

    def build_new(self):
        dim = self.dim
        structure = self.structure
        conditionning = None
        for k_tree in range(dim-1):
            row = dim - k_tree - 1
            tree = Tree(dim, k_tree)
            for col in range(dim-k_tree-1):
                i = structure[col, col]
                j = structure[row, col]
                conditionned = [i, j]
                if k_tree > 0:
                    conditionning = structure[row+1:, col].tolist()
                print(conditionned, conditionning)
                
    def build(self):
        dim = self.dim
        structure = self.structure
        tmp = structure.diagonal().tolist()
        self.trees[0].nodes = [([k], []) for k in tmp]
        # Explore the structure matrix
        for col in range(dim-1):
            # The other pairs
            rows = range(1+col, dim)[::-1]
            for k_tree, row in enumerate(rows):
                tree = self.trees[k_tree]
                i = structure[col, col]
                j = structure[row, col]
                conditionned = [i, j]
                conditionning = structure[row+1:, col].tolist()
                edge = (conditionned, conditionning)
                tree.edges.append(edge)
                
        for k_tree in range(dim-2):
            self.trees[k_tree+1].nodes = self.trees[k_tree].edges