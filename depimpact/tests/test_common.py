import numpy as np
import openturns as ot

from depimpact import ConservativeEstimate


def test_len_params():
    dim = 2
    func = lambda x: None
    margins = [ot.Normal()]*dim
    families = np.tril(np.ones((dim, dim)), k=1)
    ConservativeEstimate(model_func=func,
                         margins=margins,
                         families=families)
    
