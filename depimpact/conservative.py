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

from .conversion import Conversion, get_tau_interval
from .utils import (asymptotic_error_quantile, bootstrap, dict_to_margins,
                    get_grid_sample, get_pair_id, get_pairs_by_levels,
                    get_possible_structures, list_to_matrix,
                    load_dependence_grid, margins_to_dict, matrix_to_list,
                    save_dependence_grid, to_copula_params, to_kendalls)
from .vinecopula import VineCopula, check_matrix, check_family

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
    """

    def __init__(self,
                 model_func,
                 margins,
                 families,
                 fixed_params=None,
                 bounds_tau=None,
                 vine_structure=None,
                 copula_type='vine'):
        self.model_func = model_func
        self.margins = margins
        self.families = families
        self.bounds_tau = bounds_tau
        self.fixed_params = fixed_params
        self.vine_structure = vine_structure
        self.copula_type = copula_type

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

        kendalls = None
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
        else:
            # TODO: correct the loading
            # Load the sample from file and get the filename
            kendalls, grid_filename = load_dependence_grid(
                dirname=grid_path,
                n_pairs=self._n_pairs,
                n_params=n_dep_param,
                bounds_tau=self._bounds_tau_list,
                grid_type=grid_type,
                use_grid=use_grid)
            converter = [self._copula_converters[k] for k in self._pair_ids]
            params = to_copula_params(converter, kendalls)

        # TODO: correct the saving
        # The grid is save if it was asked and if it does not already exists
        if save_grid not in [None, False] and load_grid in [None, False]:
            if kendalls is None:
                kendalls = to_kendalls(self._copula_converters, params)
            grid_filename = save_dependence_grid(grid_path, kendalls, self._bounds_tau_list,
                                                 grid_type)

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
        outputs = self.model_func(np.concatenate(input_samples))
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

        if isinstance(param, list):
            param = np.asarray(param)
        elif isinstance(param, float):
            param = np.asarray([param])

        assert param.ndim == 1, 'Only one parameter at a time for the moment'

        full_param = np.zeros((self._corr_dim, ))
        full_param[self._pair_ids] = param
        full_param[self._fixed_pairs_ids] = self._fixed_params_list
        input_sample = self._get_sample(full_param, n_input_sample)

        output_sample = self.model_func(input_sample)

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
            The number of observations in the sampling of :math:`\mathbf X`.

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
            The number of observations in the sampling of :math:`\mathbf{X}`.

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
        output_sample = self.model_func(input_sample)

        if not keep_input_sample:
            input_sample = None

        return DependenceResult(margins=self._margins,
                                input_sample=input_sample,
                                output_sample=output_sample,
                                random_state=rng)

    def _get_sample(self, param, n_sample, param2=None):
        """Creates the observations of the joint input distribution.

        Parameters
        ----------
        param : :class:`~numpy.ndarray`
            A list of :math:`p` copula dependence parameters.
        n_sample : int
            The number of observations.
        param2 : :class:`~numpy.ndarray`, optional (default=None)
            The 2nd copula parameters. For some copula families
            (e.g. Student)
        """
        dim = self._input_dim
        matrix_param = list_to_matrix(param, dim)

        if self._copula_type == 'vine':
            # TODO: One param is used. Do it for two parameters copulas.
            vine_copula = VineCopula(self._vine_structure, self._families,
                                     matrix_param)
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
        assert callable(func), \
            TypeError("The model function must be callable.")
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

        # If there is only one parameter, no need to return the list
        if params.size == 1:
            params = params.item()
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


class ListDependenceResult(list):
    """The result from the Conservative Estimation.

    The results gather in the list must have the same configurations: the same
    copula families, vine structure, grid.

    Parameters
    ----------
    margins : list of OpenTURNS distributions
        The OT distributions.
    families : array
        The matrix array of the families.
    vine_structure : array
        The matrix array of the R-vine. If None, it is considered as Gaussian.
    bounds_tau : array,
        The matrix array of the bounds for the dependence parameters.
    dep_param : array
        The dependence parameters.
    input_sample : array
        The input sample.
    output_sample : array
        The output sample.
    q_func : callable or None
        The output quantity of intereset function.
    run_type : str
        The type of estimation: independence, grid-search, iterative, ...
    grid_type : str
        The type of grid use if it was a grid search.
    random_state : int, RandomState or None,
        The random state of the computation.

    """

    def __init__(self,
                 margins=None,
                 families=None,
                 vine_structure=None,
                 bounds_tau=None,
                 fixed_params=None,
                 dep_params=None,
                 input_samples=None,
                 output_samples=None,
                 q_func=None,
                 run_type=None,
                 n_evals=None,
                 grid_type=None,
                 random_state=None,
                 **kwargs):

        self.margins = margins
        self.families = families
        self.vine_structure = vine_structure
        self.bounds_tau = bounds_tau
        self.fixed_params = fixed_params
        self._q_func = q_func
        self.run_type = run_type
        self.grid_type = grid_type
        self.input_dim = len(margins)
        self.corr_dim = int(self.input_dim * (self.input_dim - 1) / 2)

        self.grid_filename = None
        if "grid_filename" in kwargs:
            self.grid_filename = kwargs["grid_filename"]

        self.lhs_grid_criterion = None
        if "lhs_grid_criterion" in kwargs:
            self.lhs_grid_criterion = kwargs["lhs_grid_criterion"]

        self.output_id = 0
        if "output_id" in kwargs:
            self.output_id = kwargs["output_id"]

        if run_type in ['grid-search', 'iterative']:
            assert output_samples is not None, \
                "Add some output sample if you're adding dependence parameters"

            for k, dep_param in enumerate(dep_params):
                input_sample = None if input_samples is None else input_samples[k]
                output_sample = output_samples[k]

                result = DependenceResult(margins=margins,
                                          families=families,
                                          vine_structure=vine_structure,
                                          fixed_params=fixed_params,
                                          dep_param=dep_param,
                                          input_sample=input_sample,
                                          output_sample=output_sample,
                                          q_func=q_func,
                                          random_state=random_state,
                                          output_id=self.output_id)
                self.append(result)

            if output_sample.shape[0] == output_sample.size:
                self.output_dim = 1
            else:
                self.output_dim = output_sample.shape[1]

        elif run_type == 'independence':
            # There is data and we suppose it's at independence or a fixed params
            result = DependenceResult(margins=margins,
                                      families=families,
                                      vine_structure=vine_structure,
                                      fixed_params=fixed_params,
                                      dep_param=0,
                                      input_sample=input_samples,
                                      output_sample=output_samples[0],
                                      q_func=q_func,
                                      random_state=random_state,
                                      output_id=self.output_id)
            self.families = 0
            self.vine_structure = 0
            self.bounds_tau = 0
            self.fixed_params = 0
            self.grid_type = 0
            self.append(result)
            self.output_dim = result.output_dim

        elif run_type == 'incomplete':
            # There is data and we suppose it's at independence or a fixed params
            result = DependenceResult(margins=margins,
                                      families=families,
                                      vine_structure=vine_structure,
                                      fixed_params=fixed_params,
                                      dep_param=0,
                                      input_sample=input_samples,
                                      output_sample=output_samples[0],
                                      q_func=q_func,
                                      random_state=random_state,
                                      output_id=self.output_id)

            self.grid_type = 0
            self.append(result)
            self.output_dim = result.output_dim

        self.rng = check_random_state(random_state)
        self._bootstrap_samples = None

    def __add__(self, results):
        """
        """
        if self.n_params > 0:
            # Assert the results are the same categories
            np.testing.assert_equal(
                self.margins, results.margins, err_msg="Same margins")
            np.testing.assert_array_equal(
                self.families, results.families, err_msg="Different copula families")
            np.testing.assert_array_equal(
                self.vine_structure, results.vine_structure, err_msg="Different copula structures")
            np.testing.assert_array_equal(
                self.bounds_tau, results.bounds_tau, err_msg="Different bounds on Tau")
            np.testing.assert_array_equal(
                self.fixed_params, results.fixed_params, err_msg="Different fixed params")
            np.testing.assert_allclose(
                self.dep_params, results.dep_params, err_msg="Different dependence parameters")
            assert self.run_type == results.run_type, "Different run type"
            assert self.grid_type == results.grid_type, "Different grid type"
            assert self.grid_filename == results.grid_filename, "Different grid type"
            assert self.lhs_grid_criterion == results.lhs_grid_criterion, "Different grid type"

            input_samples = []
            output_samples = []
            for res1, res2 in zip(self, results):
                if res1.input_sample is not None:
                    input_samples.append(
                        np.r_[res1.input_sample, res2.input_sample])
                output_samples.append(
                    np.r_[res1.output_sample, res2.output_sample])

            if len(input_samples) == 0:
                input_samples = None
            new_results = ListDependenceResult(
                margins=self.margins,
                families=self.families,
                vine_structure=self.vine_structure,
                bounds_tau=self.bounds_tau,
                fixed_params=self.fixed_params,
                dep_params=self.dep_params,
                input_samples=input_samples,
                output_samples=output_samples,
                grid_type=self.grid_type,
                q_func=self.q_func,
                run_type=self.run_type,
                grid_filename=self.grid_filename,
                lhs_grid_criterion=self.lhs_grid_criterion,
                output_id=self.output_id)
        return new_results

    def extend(self, value):
        super(ListDependenceResult, self).extend(value)
        self.families = value.families

    @property
    def output_id(self):
        """Id of the output.
        """
        return self._output_id

    @output_id.setter
    def output_id(self, output_id):
        for result in self:
            result.output_id = output_id
        self._output_id = output_id

    @property
    def q_func(self):
        """The quantity function
        """
        return self._q_func

    @q_func.setter
    def q_func(self, q_func):
        assert callable(q_func), "Function must be callable"
        if self.n_params == 0:
            print("There is no data...")
        else:
            for result in self:
                result.q_func = q_func
        self._q_func = q_func

    @property
    def pairs(self):
        """The dependent pairs of the problem.
        """
        if self.families is None:
            print('Family matrix was not defined')
        else:
            return matrix_to_list(self.families)[1]

    @property
    def dep_params(self):
        """The dependence parameters.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.dep_param for result in self])

    @property
    def kendalls(self):
        """The Kendall's tau dependence measure.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.kendall_tau for result in self])

    @property
    def n_pairs(self):
        """The number of dependente pairs.
        """
        if self.n_params == 0:
            return 0
        else:
            return (self.families > 0).sum()

    @property
    def output_samples(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return [result.output_sample for result in self]

    @property
    def input_samples(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return [result.input_sample for result in self]

    @property
    def n_input_sample(self):
        """The sample size for each dependence parameter.
        """
        # TODO: maybe not all the samples have the same number of observations...
        if self.n_params == 0:
            return 0
        else:
            return self[0].n_sample

    @property
    def n_evals(self):
        """The total number of observations.
        """
        return self.n_params*self.n_input_sample

    @property
    def n_params(self):
        """The number of dependence parameters.
        """
        return len(self)

    @property
    def quantities(self):
        """The quantity values of each parameters.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.quantity for result in self])

    @property
    def min_result(self):
        """The dependence parameter that minimizes the output quantity.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return self[self.quantities.argmin()]

    @property
    def min_quantity(self):
        """The minimum quantity from all the dependence parameters.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return self.quantities.min()

    @property
    def full_dep_params(self):
        """The dependence parameters with the columns from the fixed parameters.
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.full_dep_params for result in self])

    @property
    def bootstrap_samples(self):
        """The computed bootstrap sample of all the dependence parameters.
        """
        sample = [result._bootstrap_sample for result in self]
        if not any((boot is None for boot in sample)):
            return np.asarray(sample)
        else:
            raise AttributeError('The boostrap must be computed first')

    def compute_bootstraps(self, n_bootstrap=1000, inplace=True):
        """Compute bootstrap of the quantity for each element of the list
        """
        if self.n_params == 0:
            print("There is no data...")
        else:
            for result in self:
                result.compute_bootstrap(n_bootstrap)

            if not inplace:
                return self.bootstrap_samples

    def to_hdf(self, path_or_buf, input_names=[], output_names=[], verbose=False, with_input_sample=True):
        """Write the contained data to an HDF5 file using HDFStore.

        Parameters
        ----------
        path_or_buf : the path (string) or HDFStore object
            The path of the file or an hdf instance.
        input_names : list of strings, optional
            The name of the inputs variables.
        output_names : list of strings, optional
            The name of the outputs.
        """
        filename, extension = os.path.splitext(path_or_buf)
        dirname = os.path.dirname(path_or_buf)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        assert extension in ['.hdf', '.hdf5'], "File extension should be hdf"

        # List of input variable names
        if input_names:
            assert len(input_names) == self.input_dim, \
                AttributeError("Dimension problem for input_names")
        else:
            for i in range(self.input_dim):
                input_names.append("x_%d" % (i + 1))

        # List of output variable names
        if output_names:
            assert len(output_names) == self.output_dim, \
                AttributeError("Dimension problem for output_names")
        else:
            for i in range(self.output_dim):
                output_names.append("y_%d" % (i + 1))

        margin_dict = margins_to_dict(self.margins)

        filename_exists = True
        k = 0
        while filename_exists:
            # If the file has the same run configuration
            try:
                with h5py.File(path_or_buf, 'a') as hdf_store:
                    # If the file already exists and already has data
                    if hdf_store.attrs.keys():
                        # Check the attributes of the file, if it already exists
                        np.testing.assert_allclose(
                            hdf_store['dependence_params'].value, self.dep_params, err_msg="Different dependence parameters")
                        assert hdf_store.attrs['Input Dimension'] == self.input_dim, "Different input dimension"
                        assert hdf_store.attrs['Output Dimension'] == self.output_dim, "Different output dimension"
                        assert hdf_store.attrs['Run Type'] == self.run_type, "Different run type"
                        np.testing.assert_array_equal(
                            hdf_store.attrs['Copula Families'], self.families, err_msg="Different copula families")
                        if 'Fixed Parameters' in hdf_store.attrs.keys():
                            np.testing.assert_array_equal(
                                hdf_store.attrs['Fixed Parameters'], self.fixed_params, err_msg="Different fixed copulas")
                        elif self._fixed_pairs:
                            # Save only if there is no fixed params
                            raise ValueError(
                                'It should not have constraints to be in the same output file.')
                        if 'Bounds Tau' in hdf_store.attrs.keys():
                            np.testing.assert_array_equal(
                                hdf_store.attrs['Bounds Tau'], self.bounds_tau, err_msg="Different bounds on Tau")
                        elif self._fixed_pairs:
                            raise ValueError(
                                'It should not have constraints to be in the same output file.')
                        np.testing.assert_array_equal(
                            hdf_store.attrs['Copula Structure'], self.vine_structure, err_msg="Different vine structures")
                        np.testing.assert_array_equal(
                            hdf_store.attrs['Input Names'], input_names, err_msg="Different Input Names")
                        np.testing.assert_array_equal(
                            hdf_store.attrs['Output Names'], output_names, err_msg="Different output Names")

                        loaded_margin_dict = json.loads(
                            hdf_store.attrs['Margins'])
                        assert all(loaded_margin_dict[str(
                            k)] == margin_dict[k] for k in margin_dict), "Not the same dictionary"

                        if self.run_type == 'grid-search':
                            assert hdf_store.attrs['Grid Type'] == self.grid_type, "Different grid type"
                    else:
                        # We save the attributes in the empty new file
                        hdf_store.create_dataset(
                            'dependence_params', data=self.dep_params)
                        # Margins
                        hdf_store.attrs['Margins'] = json.dumps(
                            margin_dict, ensure_ascii=False)
                        hdf_store.attrs['Copula Families'] = self.families
                        hdf_store.attrs['Copula Structure'] = self.vine_structure
                        hdf_store.attrs['Bounds Tau'] = self.bounds_tau
                        hdf_store.attrs['Grid Size'] = self.n_params
                        hdf_store.attrs['Input Dimension'] = self.input_dim
                        hdf_store.attrs['Output Dimension'] = self.output_dim
                        hdf_store.attrs['Fixed Parameters'] = self.fixed_params
                        hdf_store.attrs['Run Type'] = self.run_type
                        hdf_store.attrs['Input Names'] = input_names
                        hdf_store.attrs['Output Names'] = output_names

                        if self.run_type == 'grid-search':
                            hdf_store.attrs['Grid Type'] = self.grid_type
                            if self.grid_filename is not None:
                                hdf_store.attrs['Grid Filename'] = os.path.basename(
                                    self.grid_filename)
                            if self.grid_type == 'lhs':
                                hdf_store.attrs['LHS Criterion'] = self.lhs_grid_criterion

                    # Check the number of experiments
                    grp_number = 0
                    list_groups = hdf_store.keys()
                    list_groups.remove('dependence_params')
                    list_groups = [int(g) for g in list_groups]
                    list_groups.sort()

                    # If there is already groups in the file
                    if list_groups:
                        grp_number = list_groups[-1] + 1

                    grp = hdf_store.create_group(str(grp_number))
                    for i in range(self.n_params):
                        grp_i = grp.create_group(str(i))
                        grp_i.attrs['n'] = self[i].n_sample
                        grp_i.create_dataset(
                            'output_sample', data=self[i].output_sample)
                        if with_input_sample:
                            grp_i.create_dataset(
                                'input_sample', data=self[i].input_sample)
                    filename_exists = False
            except AssertionError as msg:
                print('File %s has different configurations' % (path_or_buf))
                if verbose:
                    print(str(msg))
                path_or_buf = '%s_num_%d%s' % (filename, k, extension)
                k += 1

        if verbose:
            print('Data saved in %s' % (path_or_buf))

    @classmethod
    def from_hdf(cls, filepath_or_buffer, id_of_experiment='all', output_id=0,
                 with_input_sample=True, q_func=np.var):
        """Loads result from HDF5 file.

        This class method creates an instance of :class:`~ConservativeEstimate` 
        by loading a HDF5 with the saved result of a previous run.

        Parameters
        ----------
        filepath_or_buffer : str
            The path of the file to hdf5 file read.
        id_of_experiment : str or int, optional (default='all')
            The experiments to load. The hdf5 file can gather multiple 
            experiments with the same metadatas. The user can chose to load all
            or one experiments.
        output_id : int, optional (default=0)
            The index of the output if the function output is multidimensional.
        with_input_sample : bool, optional (default=True)
            If False the samples of input observations are not loaded. Input 
            observations are not necessary to compute output quantity of 
            interests.

        Returns
        -------
        obj : :class:`~ImpactOfDependence`
            The Impact Of Dependence instance with the loaded informations.
        """
        # Load of the hdf5 file
        with h5py.File(filepath_or_buffer, 'r') as hdf_store:
            # The file may contain multiple experiments. The user can load one
            # or multiple experiments if they have similiar configurations.
            if id_of_experiment == 'all':
                # All groups of experiments are loaded and concatenated
                list_index = hdf_store.keys()
                list_index.remove('dependence_params')
            else:
                # Only the specified experiment is loaded
                assert isinstance(id_of_experiment, int), 'It should be an int'
                list_index = [str(id_of_experiment)]

            params = hdf_store['dependence_params'].value
            run_type = hdf_store.attrs['Run Type']
            families = hdf_store.attrs['Copula Families']
            vine_structure = hdf_store.attrs['Copula Structure']
            #copula_type = hdf_store.attrs['Copula Type']
            input_dim = hdf_store.attrs['Input Dimension']
            input_names = hdf_store.attrs['Input Names']

            # Many previous experiments did not have this attribute.
            # The checking is temporary and should be deleted in future
            # versions.
            fixed_params = None
            if 'Fixed Parameters' in hdf_store.attrs.keys():
                fixed_params = hdf_store.attrs['Fixed Parameters']
            bounds_tau = None
            if 'Bounds Tau' in hdf_store.attrs.keys():
                bounds_tau = hdf_store.attrs['Bounds Tau']

            margins = dict_to_margins(json.loads(hdf_store.attrs['Margins']))

            grid_type = None
            grid_filename = None
            lhs_grid_criterion = None
            if run_type == 'grid-search':
                grid_type = hdf_store.attrs['Grid Type']
                if 'Grid Filename' in hdf_store.attrs.keys():
                    grid_filename = hdf_store.attrs['Grid Filename']
                if grid_type == 'lhs':
                    lhs_grid_criterion = hdf_store.attrs['LHS Criterion']

            output_names = hdf_store.attrs['Output Names']

            # For each experiment
            for j_exp, index in enumerate(list_index):
                grp = hdf_store[index]  # Group of the experiment

                input_samples = None
                if with_input_sample:
                    input_samples = []

                output_samples = []
                n_samples = []
                elements = [int(i) for i in grp.keys()]
                for k in sorted(elements):
                    res = grp[str(k)]
                    if with_input_sample:
                        data_in = res['input_sample'].value
                    data_out = res['output_sample'].value

                    if with_input_sample:
                        input_samples.append(data_in)
                    output_samples.append(data_out)
                    n_samples.append(res.attrs['n'])

                result = cls(margins=margins,
                             families=families,
                             vine_structure=vine_structure,
                             bounds_tau=bounds_tau,
                             fixed_params=fixed_params,
                             dep_params=params,
                             input_samples=input_samples,
                             output_samples=output_samples,
                             grid_type=grid_type,
                             q_func=q_func,
                             run_type=run_type,
                             grid_filename=grid_filename,
                             lhs_grid_criterion=lhs_grid_criterion,
                             output_id=output_id)

                if j_exp == 0:
                    results = result
                else:
                    results = results + result

        return results


class DependenceResult(object):
    """Result from conservative estimate.

    Parameters
    ----------
    margins : list
        The OT distributions.
    families : array
        The matrix array of the families.
    vine_structure : array
        The matrix array of the R-vine. If None, it is considered as Gaussian.
    dep_param : array
        The dependence parameters.
    input_sample : array
        The input sample.
    output_sample : array
        The output sample.
    q_func : callable or None
        The output quantity of intereset function.
    random_state : int, RandomState or None,
        The random state of the computation.
    """

    def __init__(self,
                 margins=None,
                 families=None,
                 vine_structure=None,
                 fixed_params=None,
                 dep_param=None,
                 input_sample=None,
                 output_sample=None,
                 q_func=None,
                 output_id=0,
                 random_state=None):

        self.margins = margins
        self.families = families
        self.vine_structure = vine_structure
        self.fixed_params = fixed_params
        self.dep_param = dep_param
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.q_func = q_func
        self.rng = check_random_state(random_state)
        self.output_id = output_id

        self.n_sample = output_sample.shape[0]
        self.input_dim = len(margins)
        if output_sample.shape[0] == output_sample.size:
            self.output_dim = 1
        else:
            self.output_dim = output_sample.shape[1]
        self.corr_dim = int(self.input_dim * (self.input_dim - 1) / 2)
        self._bootstrap_sample = None
        self._n_bootstrap_sample = None
        self._gaussian_kde = None

    def compute_bootstrap(self, n_bootstrap=1000, inplace=True):
        """Bootstrap of the output quantity of interest.

        Parameters
        ----------
        n_bootstrap : int, optional
            The number of bootstrap samples.
        inplace : bool, optional
           If true, the bootstrap sample is returned

        Returns
        -------
            The bootstrap sample if inplace is true.
        """
        self._bootstrap_sample = bootstrap(
            self.output_sample_id, n_bootstrap, self.q_func)
        self._n_bootstrap_sample = self._bootstrap_sample.shape[0]
        if (self._bootstrap_sample is None) or (self._n_bootstrap_sample != n_bootstrap):
            self.compute_bootstrap(n_bootstrap)

        self._std = self._bootstrap_sample.std()
        self._mean = self._bootstrap_sample.mean()
        self._cov = abs(self._std/self._mean)

        if not inplace:
            return self._bootstrap_sample

    def compute_quantity_bootstrap_ci(self, alphas=[0.025, 0.975], n_bootstrap=1000):
        """Boostrap confidence interval.
        """
        if (self._bootstrap_sample is None) or (self._n_bootstrap_sample != n_bootstrap):
            self.compute_bootstrap(n_bootstrap)

        return np.percentile(self._bootstrap_sample, [a*100. for a in alphas]).tolist()

    def compute_quantity_asymptotic_ci(self, quantity_name, quantity_param, ci=0.95):
        """Asymptotic confidence interval.
        """
        quantity = self.quantity

        if quantity_name == 'quantile':
            density = self.kde_estimate(self.quantity)[0]
            error = asymptotic_error_quantile(
                self.n_sample, density, quantity_param)
        elif quantity_name == 'probability':
            error = asymptotic_error_quantile(self.n_sample, quantity)
        else:
            raise 'Unknow quantity_name: {0}'.format(quantity_name)
        gaussian_quantile = norm.ppf(1. - (1. - ci)/2.)
        deviation = gaussian_quantile*error
        return [quantity - deviation, quantity + deviation]

    @property
    def boot_cov(self):
        """Coefficient of variation.
        """
        if self._cov is None:
            print('Create a bootstrap sample first.')
        return self._cov

    @property
    def boot_mean(self):
        """Mean of the quantity.
        """
        if self._mean is None:
            print('Create a bootstrap sample first.')
        return self._mean

    @property
    def boot_var(self):
        """Standard deviation of the quantity
        """
        if self._std is None:
            print('Create a bootstrap sample first.')
        return self._std

    @property
    def kde_estimate(self):
        """
        """
        if self._gaussian_kde is not None:
            return self._gaussian_kde
        else:
            self._gaussian_kde = gaussian_kde(self.output_sample_id)
            return self._gaussian_kde

    @property
    def bootstrap_sample(self):
        """The computed bootstrap sample.
        """
        if self._bootstrap_sample is not None:
            return self._bootstrap_sample
        else:
            raise AttributeError('The boostrap must be computed first')

    @property
    def quantity(self):
        """The computed output quantity.
        """
        # TODO: don't compute it everytime...
        quantity = self.q_func(self.output_sample_id, axis=0)
        return quantity.item() if quantity.size == 1 else quantity

    @property
    def output_sample_id(self):
        """
        """
        if self.output_dim == 1:
            return self.output_sample
        else:
            return self.output_sample[:, self.output_id]

    @property
    def full_dep_params(self):
        """The matrix of parameters for all the pairs.
        """
        full_params = np.zeros((self.corr_dim, ))
        pair_ids = matrix_to_list(self.families, return_ids=True)[1]
        full_params[pair_ids] = self.dep_param
        if self.fixed_params is not None:
            fixed_params, fixed_pairs = matrix_to_list(
                self.fixed_params, return_ids=True)
            full_params[fixed_pairs] = fixed_params
        return full_params

    @property
    def kendall_tau(self):
        """The Kendall's tau of the dependence parameters.
        """
        kendalls = []
        for family, id_param in zip(*matrix_to_list(self.families, return_ids=True)):
            kendall = Conversion(family).to_kendall(
                self.full_dep_params[id_param])
            # if kendall.size == 1:
            #     kendall = kendall.item()
            kendalls.append(kendall)
        return kendalls

    @property
    def full_kendall_tau(self):
        """The Kendall's tau of the dependence parameters.
        """
        kendalls = []
        for family, id_param in zip(*matrix_to_list(self.families, return_ids=True, op_char='>=')):
            kendall = Conversion(family).to_kendall(
                self.full_dep_params[id_param])
            # if kendall.size == 1:
            #     kendall = kendall.item()
            kendalls.append(kendall)
        return kendalls


def check_margins(margins):
    assert isinstance(margins, (list, tuple)), \
        TypeError("It should be a sequence of OT distribution objects.")

    for marginal in margins:
        assert isinstance(marginal, ot.DistributionImplementation), \
            TypeError("Must be an OpenTURNS distribution objects.")

    return margins
