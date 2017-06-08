import numpy as np
import openturns as ot

def func_overflow(X, model=1):
    """Overflow model function.
    
    Parameters
    ----------
    X : np.ndarray, shape : N x 8
        Input variables
        - x1 : Flow,
        - x2 : Krisler Coefficient,
        - x3 : Zv, etc...
    model : bool, optional(default=1)
        If 1, the classical model. If 2, the economic model.
        
    Returns
    -------
        Overflow S (if model=1) or Cost Cp (if model=2).
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
    else:
        raise AttributeError('Unknow model.')
    
tmp = ot.Gumbel()
tmp.setParameter(ot.GumbelMuSigma()([1013., 558.]))

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

def func_sum(x, a=None):
    """Additive weighted model function.
    
    Parameters
    ----------
    x : np.ndarray
        The input values.
    a : np.ndarray
        The input coefficients.
        
    Returns
    -------
        y : a.x^t
    """
    if isinstance(x, list):
        x = np.asarray(x)
    n, dim = x.shape
    if a is None:
        a = np.ones((dim, 1))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        assert a.shape[0] == dim, "Shape not good"
    elif a.ndim > 2:
        raise AttributeError('Dimension problem for constant a')
        
    y = np.dot(x, a)
        
    if y.size == 1:
        return y.item()
    elif y.size == y.shape[0]:
        return y.ravel()
    else:
        return y
    
    
def func_prod(x, a=None):
    """Product weighted model function.
    
    Parameters
    ----------
    x : np.ndarray
        The input values.
    a : np.ndarray
        The input coefficients.
        
    Returns
    -------
        y : a.x^t
    """
    if isinstance(x, list):
        x = np.asarray(x)
    n, dim = x.shape
    if a is None:
        a = np.ones((dim, 1))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        assert a.shape[0] == dim, "Shape not good"
    elif a.ndim > 2:
        raise AttributeError('Dimension problem for constant a')
        
    y = np.sum(x, axis=1)
        
    if y.size == 1:
        return y.item()
    elif y.size == y.shape[0]:
        return y.ravel()
    else:
        return y
    
def func_cum_sum_weight(x, a=None):
    """Additive weighted model function.
    
    Parameters
    ----------
    x : np.ndarray
        The input values.
    a : np.ndarray
        The input coefficients.
        
    Returns
    -------
        y : a.x^t
    """
    if isinstance(x, list):
        x = np.asarray(x)
    n, dim = x.shape
    if a is None:
        a = np.zeros((dim, dim))
        corr_dim = dim * (dim-1)/2
        k = 1
        for i in range(1, dim):
            for j in range(i):
                a[i, j] = k
                k += 1
        a /= corr_dim
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        assert a.shape[0] == dim, "Shape not good"
    elif a.ndim > 2:
        raise AttributeError('Dimension problem for constant a')
        
    if False:
        y = 1
        for i in range(1, dim):
            for j in range(i):
                y *= (1. + a[i, j] * func_sum(np.c_[x[:, i], x[:, j]]))
    else:
        y = 0
        for i in range(1, dim):
            for j in range(i):
                y += a[i, j] * func_prod(np.c_[x[:, i], x[:, j]])
            
    return y


def multi_output_func_sum(x, output_dim=2):
    """Additive model function with multi output.

    Parameters
    ----------
    x : np.ndarray
        The input values.
    output_dim : int
        The number of output dimension.

    Returns
    -------
        y : [i * x]
    """
    return np.asarray([x.sum(axis=1)*a for a in range(output_dim)]).T