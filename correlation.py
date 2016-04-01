import numpy as np
import random
import openturns as ot

def get_random_rho(size, dim, rho_min=-1., rho_max=1.):
    if dim == 1:
        list_rho = np.asarray(ot.Uniform(rho_min, rho_max).getSample(size))
    else:  # TODO : make it available in d dim
        list_rho = np.zeros((size, dim))
        for i in range(size):
            rho1 = random.uniform(rho_min, rho_max)
            rho2 = random.uniform(rho_min, rho_max)
            l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
            u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
            rho3 = random.uniform(l_bound, u_bound)
            list_rho[i, :] = [rho1, rho2, rho3]

    return list_rho


def get_list_rho3(rho1, rho2, n):
    """
    """
    l_bound = rho1*rho2 - np.sqrt((1-rho1**2)*(1-rho2**2))
    u_bound = rho1*rho2 + np.sqrt((1-rho1**2)*(1-rho2**2))
    list_rho3 = np.linspace(l_bound, u_bound, n+1, endpoint=False)[1:]
    return list_rho3


def get_grid_rho(n_sample, dim=3, rho_min=-1., rho_max=1., all_sample=True):
    """
    """
    if dim == 1:
        return np.linspace(rho_min, rho_max, n_sample + 1,
                           endpoint=False)[1:]
    else:
        if all_sample:
            n = int(np.floor(n_sample**(1./dim)))
        else:
            n = n_sample
        v_rho_1 = [np.linspace(rho_min, rho_max, n + 1,
                               endpoint=False)[1:]]*(dim - 1)
        grid_rho_1 = np.meshgrid(*v_rho_1)
        list_rho_1 = np.vstack(grid_rho_1).reshape(dim - 1, -1).T

        list_rho = np.zeros((n**dim, dim))
        for i, rho_1 in enumerate(list_rho_1):
            a1 = rho_1[0]
            a2 = rho_1[1]
            list_rho_2 = get_list_rho3(a1, a2, n)
            for j, rho_2 in enumerate(list_rho_2):
                tmp = rho_1.tolist()
                tmp.append(rho_2)
                list_rho[n*i+j, :] = tmp

        return list_rho