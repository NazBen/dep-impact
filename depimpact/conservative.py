"""Impact of Dependencies.

The main class inspect the impact of correlation on a quantity
of interest of a model output.

TODO:
    - Make test functions
    - Clean the code
    - Add a load/save to DependenceResult class
    - User np.tril to take the values of the matrices
"""

import json
import operator
import os
import time
import warnings

import h5py
import numpy as np
import openturns as ot
import pandas as pd
from scipy.stats import gaussian_kde, norm
from sklearn.utils import check_random_state

from .utils import (asymptotic_error_quantile, bootstrap, dict_to_margins,
                    get_grid_sample, get_pair_id, get_pairs_by_levels,
                    get_possible_structures, list_to_matrix,
                    load_dependence_grid, margins_to_dict, matrix_to_list,
                    save_dependence_grid, to_copula_params, to_kendalls)

from .result import DependenceResult, ListDependenceResult
from .vinecopula import VineCopula, Conversion, check_matrix,\
    check_family, check_triangular, get_tau_interval

OPERATORS = {">": operator.gt, ">=": operator.ge,
             "<": operator.lt, "<=": operator.le}

GRID_TYPES = ['lhs', 'rand', "fixed", 'vertices']
DEP_MEASURES = ['kendall', 'parameter']


class ConservativeEstimate(object):
    """
    Conservative estimation, toward dependencies, of a quantity of interest at 
    the output of a computational model.

    In a problem with incomplete dependence information, one can try to 
    determine the worst case scenario of dependencies according to a certain 
    risk. The dependence structure is modeled using parametric copulas. For
    multidimensional problems, in addition of Elliptic copulas, one can use
    R-vines to construct multidimensional copulas.

    Parameters
    ----------
    model_func : callable
        The evaluation model :math:`g : \mathbb R^d \rightarrow \mathbb R`
        such as :math:`Y = g(\mathbf X) \in \mathbb R`.
    margins : list of :class:`~openturns.Distribution`
        The :math:`p` marginal distributions.
    families : :class:`~numpy.ndarray`
        The copula family matrix. It describes the family type of each pair
        of variables. See the Vine Copula package for a description of the
        available copulas and their respective indexes.
    vine_structure : :class:`~numpy.ndarray` or None, optional (default=None)
        The Vine copula structure matrix. It describes the construction of the
        vine tree.
        If None, a default matrix is created.
    fixed_params : :class:`~numpy.ndarray`, str or None, optional(default=None)
        The matrix of copula parameters for the fixed copula. Warning: the 
        matrix should contains NaN for all parameters which are not fixed.
        If str, it should be the path to a csv file describing the matrix.
        If None, no parameters are fixed and a default matrix is created.
    bounds_tau : :class:`~numpy.ndarray`, str or None, optional(default=None)
        The matrix of bounds for the exploration of dependencies. The bounds
        have to be on the Kendall's Tau.
        If str, it should be the path to a csv file describing the matrix.
        If None, no bounds are setted and a default matrix is created.
    copula_type : string, optional (default='vine')
        The type of copula. Available types:
        - 'vine': Vine Copula construction
        - 'normal': Multi dimensionnal Gaussian copula.
    params2 : array or None
        The second parameter of the pair-copulas with two parameters. This one
        is fixed for all the estimation.
    """
    def __init__(self,
                 model_func,
                 margins,
                 families,
                 fixed_params=None,
                 bounds_tau=None,
                 vine_structure=None,
                 copula_type='vine',
                 param2=None):
        self.model_func = model_func
        self.margins = margins
        self.families = families
        self.bounds_tau = bounds_tau
        self.fixed_params = fixed_params
        self.vine_structure = vine_structure
        self.copula_type = copula_type
        self.param2 = param2

    def gridsearch(self,
                   n_dep_param,
                   n_input_sample,
                   grid_type='lhs',
                   dep_measure='kendall',
                   lhs_grid_criterion='centermaximin',
                   keep_input_samples=True,
                   load_grid=None,
                   save_grid=None,
                   use_sto_func=False,
                   random_state=None,
                   verbose=False):
        """Grid search over the dependence parameter space.

        Parameters
        ----------
        n_dep_param : int
            The number of dependence parameters in the grid search.
        n_input_sample : int
            The sample size for each dependence parameter.
        grid_type : 'lhs', 'rand' or 'fixed, optional (default='lhs')
            The type of grid :

            - 'lhs' : a Latin Hypercube Sampling (from pyDOE package) with criterion defined by the parameter lhs_grid_criterion,
            - 'rand' : a random sampling,
            - 'fixed' : an uniform grid,
            - 'vertices' : sampling over the vertices of the space.
        dep_measure : 'kendall' or 'parameter', optional (default='kendall')
            The space in which the dependence parameters are created.
        lhs_grid_criterion : string, optional (default = 'centermaximin')
            Configuration of the LHS grid sampling:

            - 'centermaximin' (default),
            - 'center',
            - 'maximin',
            - 'correlation'.

        Returns
        -------
        A list of DependenceResult instances.

        """
        # TODO: gather in functions
        assert isinstance(n_input_sample, int), \
            "The sample size should be an integer."
        assert isinstance(grid_type, str), "Grid type should be a string."
        assert n_input_sample > 0, "The sample size should be positive."
        if isinstance(n_dep_param, int):
            assert n_dep_param > 0, "The grid-size should be positive."
        elif n_dep_param is None:
            assert grid_type == 'vertices', \
                "NoneType for n_dep_params only works for vertices type of grids."
        else:
            "The grid-size should be an integer or a NoneType."
        assert grid_type in GRID_TYPES,\
            "Unknow grid type: {}".format(grid_type)
        assert isinstance(
            dep_measure, str), "Dependence measure should be a string."
        assert dep_measure in DEP_MEASURES,\
            "Unknow dependence measure: {}".format(dep_measure)
        assert isinstance(keep_input_samples,
                          bool), "keep_input_samples should be a bool."

        rng = check_random_state(random_state)

        grid_filename = None

        if n_dep_param == None and grid_type == 'vertices':
            load_grid = None
            save_grid = None

        if load_grid in [None, False]:
            bounds = self._get_bounds(dep_measure)
            values = get_grid_sample(bounds, n_dep_param, grid_type)
            n_dep_param = len(values)
            if dep_measure == "parameter":
                params = values
            else:
                params = self.to_copula_parameters(values, dep_measure)
        # else:
        #     # TODO: correct the loading
        #     # Load the sample from file and get the filename
        #     kendalls, grid_filename = load_dependence_grid(
        #         dirname=grid_path,
        #         n_pairs=self._n_pairs,
        #         n_params=n_dep_param,
        #         bounds_tau=self._bounds_tau_list,
        #         grid_type=grid_type,
        #         use_grid=use_grid)
        #     converter = [self._copula_converters[k] for k in self._pair_ids]
        #     params = to_copula_params(converter, kendalls)

        # TODO: correct the saving
        # The grid is save if it was asked and if it does not already exists
        # if save_grid not in [None, False] and load_grid in [None, False]:
        #     if kendalls is None:
        #         kendalls = to_kendalls(self._copula_converters, params)
        #     grid_filename = save_dependence_grid(grid_path, kendalls, self._bounds_tau_list,
        #                                          grid_type)

        # TODO: clean this...
        # Use less memory
        if use_sto_func:
            output_samples = []
            input_samples = None if not keep_input_samples else []
            for i, param in enumerate(params):
                result = self.stochastic_function(param, n_input_sample,
                                                  return_input_sample=keep_input_samples)
                if keep_input_samples:
                    output_sample, input_sample = result
                    input_samples.append(input_sample)
                else:
                    output_sample = result

                output_samples.append(output_sample)

                if verbose:
                    if n_dep_param > 10:
                        if i % int(n_dep_param/10) == 0:
                            print('Time taken:', time.clock())
                            print('Iteration %d' % (i))
                    else:
                        print('Time taken:', time.clock())
                        print('Iteration %d' % (i))
        else:
            if keep_input_samples:
                output_samples, input_samples = self.run_stochastic_models(
                    params, n_input_sample, return_input_samples=keep_input_samples)
            else:
                output_samples = self.run_stochastic_models(
                    params, n_input_sample, return_input_samples=keep_input_samples)
                input_samples = None

        return ListDependenceResult(margins=self.margins,
                                    families=self.families,
                                    vine_structure=self.vine_structure,
                                    bounds_tau=self.bounds_tau,
                                    fixed_params=self.fixed_params,
                                    dep_params=params,
                                    input_samples=input_samples,
                                    output_samples=output_samples,
                                    run_type='grid-search',
                                    grid_type=grid_type,
                                    random_state=rng,
                                    lhs_grid_criterion=lhs_grid_criterion,
                                    grid_filename=grid_filename)

    def run_stochastic_models(self,
                              params,
                              n_input_sample,
                              return_input_samples=True,
                              random_state=None,
                              verbose=False):
        """This function considers the model output as a stochastic function by 
        taking the dependence parameters as inputs.

        Parameters
        ----------
        params : list, or `np.ndarray`
            The list of parameters associated to the predefined copula.
        n_input_sample : int, optional (default=1)
            The number of evaluations for each parameter
        random_state : 
        """
        check_random_state(random_state)
        func = self.model_func

        # Get all the input_sample
        if verbose:
            print('Time taken:', time.clock())
            print('Creating the input samples')

        input_samples = []
        for param in params:
            full_param = np.zeros((self._corr_dim, ))
            full_param[self._pair_ids] = param
            full_param[self._fixed_pairs_ids] = self._fixed_params_list
            intput_sample = self._get_sample(full_param, n_input_sample)
            input_samples.append(intput_sample)

        if verbose:
            print('Time taken:', time.clock())
            print('Evaluate the input samples')

        # Evaluate the through the model
        outputs = func(np.concatenate(input_samples))
        # List of output sample for each param
        output_samples = np.split(outputs, len(params))

        if verbose:
            print('Time taken:', time.clock())
        if return_input_samples:
            return output_samples, input_samples
        else:
            return output_samples

    def stochastic_function(self,
                            param,
                            n_input_sample=1,
                            return_input_sample=True,
                            random_state=None):
        """This function considers the model output as a stochastic function by 
        taking the dependence parameters as inputs.

        Parameters
        ----------
        param : float, list, or `np.ndarray`
            The parameters associated to the predefined copula.
        n_input_sample : int, optional (default=1)
            The number of evaluations.
        random_state : 
        """
        check_random_state(random_state)
        func = self.model_func

        if isinstance(param, list):
            param = np.asarray(param)
        elif isinstance(param, float):
            param = np.asarray([param])

        assert param.ndim == 1, 'Only one parameter at a time for the moment'

        full_param = np.zeros((self._corr_dim, ))
        full_param[self._pair_ids] = param
        full_param[self._fixed_pairs_ids] = self._fixed_params_list
        input_sample = self._get_sample(full_param, n_input_sample)
        output_sample = func(input_sample)

        if return_input_sample:
            return output_sample, input_sample
        else:
            return output_sample

    def incomplete(self, n_input_sample, q_func=np.var,
                   keep_input_sample=True, random_state=None):
        """Simulates from the incomplete probability distribution.

        Parameters
        ----------
        n_input_sample : int
            The number of observations in the sampling of X.

        Returns
        -------
        """
        rng = check_random_state(random_state)
        assert callable(q_func), "Quantity function is not callable"

        param = [0.]*self._n_pairs
        out = self.stochastic_function(param=param,
                                       n_input_sample=n_input_sample,
                                       return_input_sample=keep_input_sample,
                                       random_state=rng)

        if keep_input_sample:
            output_sample, input_sample = out
        else:
            output_sample = out

        return DependenceResult(margins=self._margins,
                                families=self.families,
                                vine_structure=self.vine_structure,
                                fixed_params=self.fixed_params,
                                input_sample=input_sample,
                                output_sample=output_sample,
                                q_func=q_func,
                                random_state=rng)

    def independence(self, n_input_sample, keep_input_sample=True,
                     random_state=None):
        """Generates and evaluates observations at the independence
        configuration.

        Parameters
        ----------
        n_input_sample : int
            The number of observations in the sampling of X`.

        Returns
        -------
        """
        rng = check_random_state(random_state)

        assert isinstance(
            n_input_sample, int), "The sample size should be an integer."
        assert n_input_sample > 0, "The sample size should be positive."

        # Creates the sample of input parameters
        tmp = ot.ComposedDistribution(self._margins).getSample(n_input_sample)
        input_sample = np.asarray(tmp)
        func = self.model_func
        output_sample = func(input_sample)

        if not keep_input_sample:
            input_sample = None

        return DependenceResult(margins=self._margins,
                                input_sample=input_sample,
                                output_sample=output_sample,
                                random_state=rng)

    def _get_sample(self, param, n_sample):
        """Creates the observations of the joint input distribution.

        Parameters
        ----------
        param : :class:`~numpy.ndarray`
            A list of :math:`p` copula dependence parameters.
        n_sample : int
            The number of observations.
        """
        dim = self._input_dim
        matrix_param = list_to_matrix(param, dim)

        if self._copula_type == 'vine':
            # TODO: One param is used. Do it for two parameters copulas.
            vine_copula = VineCopula(self._vine_structure, self._families,
                                     matrix_param, param2=self._param2)
            # Sample from the copula
            # The reshape is in case there is only one sample (for RF tests)
            cop_sample = vine_copula.get_sample(
                n_sample).reshape(n_sample, dim)
        elif self._copula_type == 'normal':
            # Create the correlation matrix
            cor_matrix = matrix_param + matrix_param.T + np.identity(dim)
            cop = ot.NormalCopula(ot.CorrelationMatrix(cor_matrix))
            cop_sample = np.asarray(cop.getSample(n_sample), dtype=float)
        else:
            raise AttributeError('Unknown type of copula.')

        # Applied the inverse transformation to get the sample of the joint distribution
        # TODO: this part is pretty much time consuming...
        input_sample = np.zeros((n_sample, dim))
        for i, marginal in enumerate(self.margins):
            quantile_func = marginal.computeQuantile
            input_sample[:, i] = np.asarray(
                quantile_func(cop_sample[:, i])).ravel()
        return input_sample

    @property
    def model_func(self):
        """The callable model function.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        assert callable(func), "The model function must be callable."
        self._model_func = func

    @property
    def margins(self):
        """The marginal distributions. 

        List of :class:`~openturns.Distribution` objects.
        """
        return self._margins

    @margins.setter
    def margins(self, margins):
        margins = check_margins(margins)
        self._margins = margins
        self._input_dim = len(margins)
        self._corr_dim = int(self._input_dim * (self._input_dim - 1) / 2)

        # TODO: clean these checking steps in functions
        if hasattr(self, '_families'):
            if self._families.shape[0] != self._input_dim:
                print("Don't forget to change the family matrix.")
        if hasattr(self, '_vine_structure'):
            if self._vine_structure.shape[0] != self._input_dim:
                # If it was a custom vine
                if self._custom_vine_structure:
                    print("Don't forget to change the R-vine array")
                else:
                    self.vine_structure = None
        if hasattr(self, '_bounds_tau'):
            if self._bounds_tau.shape[0] != self._input_dim:
                # If the user cares about the bounds
                if self._custom_bounds_tau:
                    print("Don't forget to change the bounds matrix")
                else:
                    self.bounds_tau = None
        if hasattr(self, '_fixed_params'):
            if self._fixed_params.shape[0] != self._input_dim:
                if self._custom_fixed_params:
                    print("Don't forget to change the fixed params matrix")

    def _get_bounds(self, dep_measure):
        if dep_measure == "parameter":
            bounds = self._bounds_par_list
        elif dep_measure == "kendall":
            bounds = self._bounds_tau_list
        else:
            raise TypeError('Unknow dep measure {}'.format(bounds))
        return bounds

    def to_copula_parameters(self, values, dep_measure):
        """
        """
        # TODO: do it better
        if isinstance(values, list):
            values = np.asarray(values)
        elif isinstance(values, float):
            values = np.asarray([values])

        n_params, n_pairs = values.shape
        params = np.zeros((n_params, n_pairs))
        converters = [self._copula_converters[k] for k in self._pair_ids]
        # For each pair-copula
        for k in range(n_pairs):
            params[:, k] = converters[k].to_parameter(
                values[:, k], dep_measure=dep_measure)
        return params

    @property
    def families(self):
        """The copula families.

        Matrix array of shape (dim, dim).
        """
        return self._families

    @families.setter
    def families(self, families):
        # If load from a file
        if isinstance(families, str):
            # It should be a path to a csv file
            # TODO: replace pandas with numpy
            families = pd.read_csv(families, index_col=0).values
        elif isinstance(families, np.ndarray):
            pass
        else:
            raise TypeError("Not a good type for the family matrix.")

        families = check_family(families)

        # The family list values. Event the independent ones
        self._family_list = matrix_to_list(families, op_char='>=')

        # Dpendent pairs
        _, self._pair_ids, self._pairs = matrix_to_list(families, return_ids=True,
                                                        return_coord=True, op_char='>')

        self._families = families
        self._n_pairs = len(self._pair_ids)

        # Independent pairs
        _, self._indep_pairs_ids, self._indep_pairs = matrix_to_list(
            families, return_ids=True, return_coord=True, op_char='==')

        self._copula_converters = [Conversion(
            family) for family in self._family_list]

        # TODO: add check part in functions
        if hasattr(self, '_input_dim'):
            if self._families.shape[0] != self._input_dim:
                print("Don't forget to change the margins.")

        if hasattr(self, '_vine_structure'):
            if self._families.shape[0] != self._vine_structure.shape[0]:
                if self._custom_vine_structure:
                    print("Don't forget to change the R-vine array.")

            # It should always be done in case if a pair has been set independent
            if not self._custom_vine_structure:
                self.vine_structure = None

        if hasattr(self, '_fixed_params'):
            if self._families.shape[0] != self._fixed_params.shape[0]:
                if self._custom_fixed_params:
                    print("Don't forget to change the fixed parameters.")
                else:
                    self.fixed_params = None

        if hasattr(self, '_bounds_tau'):
            if self._families.shape[0] != self._fixed_params.shape[0]:
                if self._custom_bounds_tau:
                    print("Don't forget to change the fixed parameters.")
                else:
                    self.bounds_tau = None

            # It should always be done in case if a pair has been set independent
            if not self._custom_bounds_tau:
                self.bounds_tau = None

    @property
    def corr_dim(self):
        """The number of pairs.
        """
        return self._corr_dim

    @property
    def pairs(self):
        """The possibly dependent pairs.
        """
        return self._pairs

    @property
    def pair_ids(self):
        """The possibly dependent pairs.
        """
        return self._pair_ids

    @property
    def n_pairs(self):
        """The number of possibly dependent pairs.
        """
        return self._n_pairs

    @property
    def copula_type(self):
        """The type of copula.

        Can be "vine", or gaussian type.
        """
        return self._copula_type

    @copula_type.setter
    def copula_type(self, value):
        # TODO: check this
        assert isinstance(value, str), \
            TypeError('Type must be a string. Type given:', type(value))

        if value == "normal":
            families = self._families
            # Warn if the user added a wrong type of family
            if (families[families > 0] != 1).any():
                warnings.warn(
                    'Some families were not normal and you want an elliptic copula.')

            # Set all families to gaussian
            families[families > 0] = 1
            self.families = families
        self._copula_type = value

    @property
    def vine_structure(self):
        """The R-vine array.
        """
        return self._vine_structure

    @vine_structure.setter
    def vine_structure(self, structure):
        # TODO: check this
        if structure is None:
            listed_pairs = self._indep_pairs + self._fixed_pairs
            dim = self.input_dim
            # TODO : this should be corrected...
            if len(listed_pairs) > 0:
                # if False:
                pairs_iter_id = [get_pair_id(
                    dim, pair, with_plus=False) for pair in listed_pairs]
                pairs_by_levels = get_pairs_by_levels(dim, pairs_iter_id)
                structure = get_possible_structures(dim, pairs_by_levels)[1]
            else:
                structure = np.zeros((dim, dim), dtype=int)
                for i in range(dim):
                    structure[i, 0:i+1] = i + 1
            self._custom_vine_structure = False
        else:
            check_matrix(structure)
            self._custom_vine_structure = True
        self._vine_structure = structure
        self._vine_structure_list = structure[np.tril_indices_from(structure)]

    @property
    def input_dim(self):
        """The input dimension.
        """
        return self._input_dim

    @property
    def bounds_tau(self):
        """The matrix bound for the kendall's tau.
        """
        return self._bounds_tau

    @property
    def bounds_par(self):
        """The matrix bound for the dependence parameters.
        """
        return self._bounds_par

    @bounds_tau.setter
    def bounds_tau(self, value):
        """Set the upper bound of the Kendall Tau parameter space.

        Parameters
        ----------
        value : :class:`~numpy.ndarray`, str or None
            Matrix of bounds.
        """
        # TODO: check this
        dim = self._input_dim
        self._custom_bounds_tau = True
        # If no bounds given, we take the min and max, depending on the copula family
        if value is None:
            bounds_tau = np.zeros((dim, dim))
            for i, j in self._pairs:
                bounds_tau[i, j], bounds_tau[j, i] = get_tau_interval(
                    self._families[i, j])
            self._custom_bounds_tau = False
        elif isinstance(value, str):
            # It should be a path to a csv file
            bounds_tau = pd.read_csv(value, index_col=0).values
        else:
            bounds_tau = value

        bounds_par = np.zeros(bounds_tau.shape)
        bounds_tau_list = []
        bounds_par_list = []
        for k, p in enumerate(self._pair_ids):
            i, j = self._pairs[k]
            tau_min, tau_max = get_tau_interval(self._family_list[p])

            if not np.isnan(bounds_tau[i, j]):
                tau_min = max(bounds_tau[i, j], tau_min)

            if not np.isnan(bounds_tau[j, i]):
                tau_max = min(bounds_tau[j, i], tau_max)
            bounds_tau_list.append([tau_min, tau_max])

            # Conversion to copula parameters
            param_min = self._copula_converters[p].to_parameter(
                tau_min, 'kendall')
            param_max = self._copula_converters[p].to_parameter(
                tau_max, 'kendall')

            bounds_par[i, j] = tau_min
            bounds_par[j, i] = tau_max
            bounds_par_list.append([param_min, param_max])

        check_matrix(bounds_tau)
        check_matrix(bounds_par)
        self._bounds_tau = bounds_tau
        self._bounds_par = bounds_par
        self._bounds_tau_list = bounds_tau_list
        self._bounds_par_list = bounds_par_list

    @property
    def param2(self):
        """The second parameter for bi-variate copulas.
        """
        return self._param2


    @param2.setter
    def param2(self, param):
        if param is not None:
            param = check_matrix(param)
            param = check_triangular(param)
        self._param2 = param

    @property
    def fixed_params(self):
        """The pairs that are fixed to a given dependence parameter value.
        """
        return self._fixed_params

    @fixed_params.setter
    def fixed_params(self, value):
        # TODO: check this
        # TODO: if it is changed multiple times, it keeps deleting pairs...
        self._custom_fixed_params = True
        if value is None:
            # There is no fixed pairs
            matrix = np.zeros((self._input_dim, self._input_dim), dtype=float)
            matrix[:] = np.nan
            self._custom_fixed_params = False
        elif isinstance(value, str):
            # It should be a path to a csv file
            matrix = pd.read_csv(value, index_col=0).values
        else:
            matrix = value

        # The matrix should be checked
        check_matrix(matrix)

        # The lists only contains the fixed pairs informations
        self._fixed_pairs = []
        self._fixed_pairs_ids = []
        self._fixed_params = matrix
        self._fixed_params_list = []
        k = 0
        # TODO: do it like for the families property
        for i in range(1, self._input_dim):
            for j in range(i):
                if self._families[i, j] > 0:
                    if matrix[i, j] == 0.:
                        warnings.warn(
                            'The parameter of the pair %d-%d is set to 0. Check if this is correct.' % (i, j))
                    if not np.isnan(matrix[i, j]):
                        # The pair is fixed we add it in the list
                        self._fixed_pairs.append([i, j])
                        self._fixed_pairs_ids.append(k)
                        self._fixed_params_list.append(matrix[i, j])
                        # And we remove it from the list of dependent pairs
                        if k in self._pair_ids:
                            self._pair_ids.remove(k)
                            self._pairs.remove([i, j])
                            self._n_pairs -= 1
                        self._bounds_tau[i, j] = np.nan
                        self._bounds_tau[j, i] = np.nan
                k += 1

        if hasattr(self, '_vine_structure'):
            # It should always be done in case if a pair has been set independent
            if not self._custom_vine_structure:
                self.vine_structure = None

        if hasattr(self, '_bounds_tau'):
            # It should always be done in case if a pair has been set independent
            if not self._custom_bounds_tau:
                self.bounds_tau = None
            else:
                if self.bounds_tau.shape[0] != self.input_dim:
                    print("Dont't foget to update the bounds matrix")
                else:
                    bounds_tau = self.bounds_tau
                    # We delete the bounds for the fixed pairs
                    for fixed_pair in self._fixed_pairs:
                        bounds_tau[fixed_pair[0], fixed_pair[1]] = np.nan
                        bounds_tau[fixed_pair[1], fixed_pair[0]] = np.nan
                    self.bounds_tau = bounds_tau


def check_margins(margins):
    assert isinstance(margins, (list, tuple)), \
        TypeError("It should be a sequence of OT distribution objects.")

    for marginal in margins:
        assert isinstance(marginal, ot.DistributionImplementation), \
            TypeError("Must be an OpenTURNS distribution objects.")

    return margins
