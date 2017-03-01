import numpy as np
import openturns as ot

def func_overflow(X, model=1):
    """
    X : input variables, shape : N x 8
    """
    X = np.asarray(X)
    if X.shape[0] == X.size: # It's a vector
        n = 1
        dim = X.size
        ids = None
    else:
        n, dim = X.shape
        ids = range(n)
    assert dim == 8, "Incorect dimension : dim = %d != 8" % dim
    
    Q = X[ids, 0]
    Ks = X[ids, 1]
    Zv = X[ids, 2]
    Zm = X[ids, 3]
    Hd = X[ids, 4]
    Cb = X[ids, 5]
    L = X[ids, 6]
    B = X[ids, 7]
    
    H = (Q / (B * Ks * np.sqrt((Zm - Zv) / L)))**1.
    S = Zv + H - Hd - Cb
    
    if model == 1:
        return S
    elif model == 2:
        Cp = (S > 0.) + (0.2 + 0.8 * (1. - np.exp(-1000. / (S**4)))) * (S <= 0.) + 1./20. * (Hd * (Hd > 8.) + 8*(Hd <= 8.))
        return Cp
    
    
tmp = ot.Gumbel(1013., 558., ot.Gumbel.MUSIGMA)
dist_Q = ot.TruncatedDistribution(tmp, 500., 3000.)
dist_Ks = ot.TruncatedNormal(30., 8., 15., np.inf)
dist_Zv = ot.Triangular(49., 50., 51.)
dist_Zm = ot.Triangular(54., 55., 56.)
dist_Hd = ot.Uniform(7., 9.)
dist_Cb = ot.Triangular(55., 55.5, 56.)
dist_L = ot.Triangular(4990., 5000., 5010.)
dist_B = ot.Triangular(295., 300., 305.)


margins_overflow = [dist_Q, dist_Ks, dist_Zv, dist_Zm, dist_Hd, dist_Cb, dist_L, dist_B]
var_names_overflow  = ["Q", "K_s", "Z_v", "Z_m", "H_d", "C_b", "L", "B"]