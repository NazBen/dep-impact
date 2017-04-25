"""Impact of Dependencies.

The main class inspect the impact of correlation on a quantity
of interest of a model output.

TODO:
    - Make test functions
    - Clean the code
    - Add the algorithm in the class or a seperated one
"""

import operator
import warnings
import itertools
import os

import numpy as np
import pandas as pd
import openturns as ot
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state

from .gridsearch import Space
from .vinecopula import VineCopula, check_matrix
from .conversion import Conversion, get_tau_interval

OPERATORS = {">": operator.gt, ">=": operator.ge,
             "<": operator.lt, "<=": operator.le}


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

    copula_type : string, optionnal (default='vine')
        The type of copula. Available types:
        - 'vine': Vine Copula construction
        - 'normal': Multi dimensionnal Gaussian copula.

    Attributes
    ----------

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
        self.vine_structure = vine_structure
        self.fixed_params = fixed_params
        self.bounds_tau = bounds_tau
        self.copula_type = copula_type

        self._forest_built = False
        self._lhs_grid_criterion = 'centermaximin'
        self._grid_folder = './experiment_designs'
        self._dep_measure = None
        self._load_data = False
        self.eps = 1.E-4

    @classmethod
    def from_hdf(cls, filepath_or_buffer, id_of_experiment='all', out_ID=0, 
        with_input_sample=True):
        """Loads result from HDF5 file.

        This class method creates an instance of :class:`~ImpactOfDependence` 
        by loading a HDF5 with the saved result of a previous run.

        Parameters
        ----------
        filepath_or_buffer : str
            The path of the file to hdf5 file read.
        id_of_experiment : str or int, optional (default='all')
            The experiments to load. The hdf5 file can gather multiple 
            experiments with the same metadatas. The user can chose to load all
            or one experiments.
        out_ID : int, optional (default=0)
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
        # Ghost function
        def tmp(): return None
        
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
            n_params = hdf_store.attrs['K']
            families = hdf_store.attrs['Copula Families']
            structure = hdf_store.attrs['Copula Structure']
            copula_type = hdf_store.attrs['Copula Type']
            input_dim = hdf_store.attrs['Input Dimension']
            input_names = hdf_store.attrs['Input Names']
            
            # Many previous experiments did not have this attribute. 
            # The checking is temporary and should be deleted in future
            # versions.
            if 'Fixed Parameters' in hdf_store.attrs.keys():
                fixed_params = hdf_store.attrs['Fixed Parameters']
            else:
                fixed_params = None
            if 'Bounds Tau' in hdf_store.attrs.keys():
                bounds_tau = hdf_store.attrs['Bounds Tau']
            else:
                bounds_tau = None

            margins = []
            for i in range(input_dim):
                marg_f = 'Marginal_%d Family' % (i)
                marg_p = 'Marginal_%d Parameters' % (i)
                marginal = getattr(ot, hdf_store.attrs[marg_f])(*hdf_store.attrs[marg_p])
                margins.append(marginal)
                
            if run_type == 'Classic':
                dep_measure = hdf_store.attrs['Dependence Measure']
                grid_type = hdf_store.attrs['Grid Type']
                if 'Grid Filename' in hdf_store.attrs.keys():
                    grid_filename = hdf_store.attrs['Grid Filename']
                else:
                    grid_filename = None
                if grid_type == 'lhs':
                    lhs_grid_criterion = hdf_store.attrs['LHS Criterion']
                else:
                    lhs_grid_criterion = None
                
            output_dim = hdf_store.attrs['Output Dimension']
            output_names = hdf_store.attrs['Output Names']
            
            list_input_sample = []
            list_output_sample = []
            list_n = []
            n = 0
            for k, index in enumerate(list_index):
                grp = hdf_store[index] # Group of the experiment
                
                data_in = grp['input_sample']
                data_out = grp['output_sample']

                list_input_sample.append(data_in.value)
                list_output_sample.append(data_out.value)
                list_n.append(grp.attrs['n'])

        # Each sample is made from the same dependence parameters
        # They need to be reordered
        n = sum(list_n)
        n_sample = n*n_params
        input_sample = np.zeros((n_sample, input_dim))
        output_sample = np.zeros((n_sample, output_dim))
        a = 0
        for i, ni in enumerate(list_n):
            for k in range(n_params):
                start = a + k*n
                end = start + ni
                input_sample[start:end, :] = list_input_sample[i][k*ni:(k+1)*ni, :]
                output_sample[start:end, :] = list_output_sample[i][k*ni:(k+1)*ni, :]
            a += ni
                    
        obj = cls(tmp, margins, families, fixed_params=fixed_params,
                  bounds_tau=bounds_tau, vine_structure=structure)
                
        obj._params = params
        obj._n_param = n_params
        obj._n_sample = n_sample
        obj._n_input_sample = n
        if with_input_sample:
            obj._input_sample = input_sample
        obj._all_output_sample = output_sample
        obj._output_dim = output_dim
        obj._copula_type = copula_type
        obj._input_names = input_names
        obj._output_names = output_names
        obj._run_type = run_type
        
        obj._output_ID = out_ID
        obj._load_data = True
        
        if run_type == 'Classic':
            obj._dep_measure = dep_measure
            obj._grid = grid_type
            obj._grid_filename = grid_filename
            obj._lhs_grid_criterion = lhs_grid_criterion

        return obj

    def gridsearch_minimize(self, n_dep_param, n_input_sample, grid_type='lhs',
                            dep_measure='KendallTau', q_func=np.var,
                            use_grid=None, save_grid=None,
                            done_results=None, random_state=None,):
        """Quantile minimization through a grid in the dependence parameter
        space.
        
        Parameters
        ----------
        n_dep_param : int
            The number of dependence parameters in the grid search.
        n_input_sample : int
            The sample size for each dependence parameter.
        grid_type : 'lhs', 'rand' or 'fixed, optional (default='lhs')
            The type of grid :
                
            - 'lhs' : a Latin Hypercube Sampling (from pyDOE package) with
            criterion defined by the attribute lhs_grid_criterion :
                    - 'centermaximin' (default),
                    - 'center',
                    - 'maximin',
                    - 'correlation'.
            - 'rand' : a random sampling,
            - 'fixed' : an uniform grid.
        dep_measure : 'KendallTau' or 'copula-parameter', 
        optional (default='KendallTau')
            The measure of dependence in which the dependence parameters are
            created.
        q_func : callable, optional (default=np.var)
            The function output quantity of interest.
        
        Returns
        -------
        A list of DependenceResult instances.                
        
        """
        rng = check_random_state(random_state)
        run_type = 'Classic'
        assert callable(q_func), "Quantity function is not callable"
         
        dimensions = [self._bounds_tau_list[pair] for pair in self.pairs_]
        params = get_dependence_parameters(dimensions, n_dep_param, grid_type)
        n_dep_param = len(params)

        params_not_to_compute = []
        # Params not to compute
        if (done_results is not None) and (done_results.n_params > 0):
            full_params = np.zeros((n_dep_param, self.corr_dim))
            full_params[:, self.pairs] = params
            done_dep_params = done_results.full_dep_params

            params_not_to_compute = []
            params_id_not_to_compute = []
            for param in full_params:
                k = np.where((param == done_dep_params).all(axis=1))[0].tolist()
                if k:
                    params_not_to_compute.append(param)
                    params_id_not_to_compute.append(k[0])

            params_not_to_compute = np.asarray(params_not_to_compute)
            n_dep_param -= len(params_id_not_to_compute)

            #print("You saved %d params to compute" % len(params_id_not_to_compute))
            params = np.delete(params, params_id_not_to_compute, axis=0)

        # Evaluate the sample
        param_func = lambda param: self.stochastic_function(param, n_input_sample)
        tmp = map(param_func, params)
        output_samples = np.asarray(tmp).reshape(n_dep_param, n_input_sample)

        # Add back the results
        if len(params_not_to_compute) > 0:
            params = np.r_[params, params_not_to_compute[:, self.pairs]]
            output_samples = np.r_[output_samples, done_results.output_samples]
            saved_nevals = len(params_not_to_compute)*n_input_sample
        else:
            saved_nevals = 0

        return ListDependenceResult(dep_params=params,
                                    output_samples=output_samples,
                                    q_func=q_func, run_type=run_type,
                                    families=self.families,
                                    random_state=rng,
                                    saved_nevals=saved_nevals)

    def stochastic_function(self, param, n_input_sample=1, random_state=None):
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
        rng = check_random_state(random_state)

        if isinstance(param, list):
            param = np.asarray(param)
        elif isinstance(param, float):
            param = np.asarray([param])

        assert param.ndim == 1, 'Only one parameter at a time for the moment'

        full_param = np.zeros((self._corr_dim, ))
        full_param[self.pairs_] = param
        
        input_sample = self._get_sample(full_param, n_input_sample)
        output_sample = self.model_func(input_sample)

        return output_sample

    def independence(self, n_input_sample, q_func=np.var, random_state=None):
        """Generates and evaluates observations at the independence
        configuration.

        Parameters
        ----------
        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf{X}`.

        seed : int or None, optional (default=None)
            If int, ``seed`` is the seed used by the random number generator;
            If None, ``seed`` is the seed is random.
        """
        rng = check_random_state(random_state)
        run_type = 'Independence'
        assert callable(q_func), "Quantity function is not callable"

        # Creates the sample of input parameters
        input_sample = np.asarray(ot.ComposedDistribution(self._margins).getSample(n_input_sample))
        output_sample = self.model_func(input_sample)

        return DependenceResult(input_sample=input_sample, output_sample=output_sample, q_func=q_func, random_state=rng)

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
        matrix_param = to_matrix(param, dim)

        if self._copula_type == 'vine':
            # TODO: One param is used. Do it for two parameters copulas.
            vine_copula = VineCopula(self._vine_structure, self._families, 
                                     matrix_param)
            # Sample from the copula
            # The reshape is in case there is only one sample (for RF tests)
            cop_sample = vine_copula.get_sample(n_sample).reshape(n_sample, dim)
        elif self._copula_type == 'normal':
            # Create the correlation matrix
            cor_matrix = matrix_param + matrix_param.T + np.identity(dim)
            cop = ot.NormalCopula(ot.CorrelationMatrix(cor_matrix))
            cop_sample = np.asarray(cop.getSample(n_sample), dtype=float)
        else:
            raise AttributeError('Unknown type of copula.')

        # Applied the inverse transformation to get the sample of the joint distribution
        input_sample = np.zeros((n_sample, dim))
        for i, inv_CDF in enumerate(self._margins_inv_CDF):
            input_sample[:, i] = np.asarray(inv_CDF(cop_sample[:, i])).ravel()

        return input_sample

    def _load_grid(self, n_param, dep_measure, use_grid, gridname):
        """
        """
        p = self._n_pairs    
        if isinstance(use_grid, str):
            assert gridname in use_grid, "Not the same configurations"
            filename = use_grid
            name = os.path.basename(filename)
        elif isinstance(use_grid, (int, bool)):
            k = int(use_grid)
            name = '%s_p_%d_n_%d_%s_%d.csv' % (gridname, p, n_param, dep_measure, k)
            filename = os.path.join(self._grid_folder, name)
        else:
            raise AttributeError('Unknow use_grid')
        assert os.path.exists(filename), 'Grid file %s does not exists' % name
        print('loading file %s' % name)
        sample = np.loadtxt(filename).reshape(n_param, -1)
        assert n_param == sample.shape[0], 'Wrong grid size'
        assert p == sample.shape[1], 'Wrong dimension'
        return sample, filename

    def save_data_hdf(self, input_names=[], output_names=[],
                  path=".", file_name="output_result.hdf5"):
        """
        """
        # List of input variable names
        if input_names:
            assert len(input_names) == self._input_dim, \
                AttributeError("Dimension problem for input_names")
        else:
            for i in range(self._input_dim):
                input_names.append("x_%d" % (i + 1))

        # List of output variable names
        if output_names:
            assert len(output_names) == self._output_dim, \
                AttributeError("Dimension problem for output_names")
        else:
            for i in range(self._output_dim):
                output_names.append("y_%d" % (i + 1))

        margin_dict = {}
        # List of marginal names
        for i, marginal in enumerate(self._margins):
                name = marginal.getName()
                params = list(marginal.getParameter())
                margin_dict['Marginal_%d Family' % (i)] = name
                margin_dict['Marginal_%d Parameters' % (i)] = params

        filename_exists = True
        init_file_name = file_name
        k = 0
        while filename_exists:
            try:
                with h5py.File(os.path.join(path, file_name), 'a') as hdf_store:
                    # General attributes
                    # Check the attributes of the file, if it already exists
                    if hdf_store.attrs.keys():                        
                        np.testing.assert_allclose(hdf_store['dependence_params'].value, self._params)
                        assert hdf_store.attrs['Input Dimension'] == self._input_dim
                        assert hdf_store.attrs['Output Dimension'] == self._output_dim
                        assert hdf_store.attrs['Run Type'] == self._run_type
                        np.testing.assert_array_equal(hdf_store.attrs['Copula Families'], self._families)                        
                        if 'Fixed Parameters' in hdf_store.attrs.keys():
                            np.testing.assert_array_equal(hdf_store.attrs['Fixed Parameters'], self._fixed_params)  
                        elif self._fixed_pairs:
                            # Save only if there is no fixed params
                            raise ValueError('It should not have constraints to be in the same output file.')
                        if 'Bounds Tau' in hdf_store.attrs.keys():
                            np.testing.assert_array_equal(hdf_store.attrs['Bounds Tau'], self._bounds_tau)           
                        elif self._fixed_pairs:
                            raise ValueError('It should not have constraints to be in the same output file.')
                        np.testing.assert_array_equal(hdf_store.attrs['Copula Structure'], self._vine_structure)
                        assert hdf_store.attrs['Copula Type'] == self._copula_type
                        np.testing.assert_array_equal(hdf_store.attrs['Input Names'], input_names)
                        np.testing.assert_array_equal(hdf_store.attrs['Output Names'], output_names)
                        for i in range(self._input_dim):
                            assert hdf_store.attrs['Marginal_%d Family' % (i)] == margin_dict['Marginal_%d Family' % (i)]
                            np.testing.assert_array_equal(hdf_store.attrs['Marginal_%d Parameters' % (i)], margin_dict['Marginal_%d Parameters' % (i)])
                            
                        if self._run_type == 'Classic':
                            assert hdf_store.attrs['Dependence Measure'] == self._dep_measure
                            assert hdf_store.attrs['Grid Type'] == self._grid
                    else:
                        # We save the attributes in the empty new file
                        hdf_store.create_dataset('dependence_params', data=self._params)
                        hdf_store.attrs['K'] = self._n_param
                        hdf_store.attrs['Input Dimension'] = self._input_dim
                        hdf_store.attrs['Output Dimension'] = self._output_dim
                        hdf_store.attrs['Run Type'] = self._run_type
                        hdf_store.attrs['Copula Families'] = self._families
                        hdf_store.attrs['Fixed Parameters'] = self._fixed_params
                        hdf_store.attrs['Bounds Tau'] = self._bounds_tau
                        hdf_store.attrs['Copula Structure'] = self._vine_structure
                        hdf_store.attrs['Copula Type'] = self._copula_type
                        hdf_store.attrs['Input Names'] = input_names
                        hdf_store.attrs['Output Names'] = output_names
                        for i in range(self._input_dim):
                            hdf_store.attrs['Marginal_%d Family' % (i)] = margin_dict['Marginal_%d Family' % (i)]
                            hdf_store.attrs['Marginal_%d Parameters' % (i)] = margin_dict['Marginal_%d Parameters' % (i)]
                        
                        if self._run_type == 'Classic':
                            hdf_store.attrs['Dependence Measure'] = self._dep_measure
                            hdf_store.attrs['Grid Type'] = self._grid
                            if self._grid_filename:
                                hdf_store.attrs['Grid Filename'] = os.path.basename(self._grid_filename)
                            if self._grid == 'lhs':
                                hdf_store.attrs['LHS Criterion'] = self._lhs_grid_criterion      
                    
                    # Check the number of experiments
                    grp_number = 0
                    list_groups = hdf_store.keys()
                    list_groups.remove('dependence_params')
                    list_groups = [int(g) for g in list_groups]
                    list_groups.sort()
                    if list_groups:
                        grp_number = list_groups[-1] + 1

                    grp = hdf_store.create_group(str(grp_number))
                    grp.attrs['n'] = self._n_input_sample
                    grp.create_dataset('input_sample', data=self._input_sample)
                    grp.create_dataset('output_sample', data=self._all_output_sample.reshape((self._n_sample, self._output_dim)))
                    filename_exists = False
            except AssertionError:
                print('File %s already has different configurations' % (file_name))
                file_name = '%s_%d.hdf5' % (init_file_name[:-5], k)
                k += 1

        print('Data saved in %s' % (file_name))

        return file_name

# =============================================================================
# Properties
# =============================================================================
    @property
    def model_func(self):
        """The callable model function.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        """
        """
        assert callable(func), TypeError("The model function must be callable.")
        self._model_func = func

    @property
    def margins(self):
        """The marginal distributions. 

        List of :class:`~openturns.Distribution` objects.
        """
        return self._margins

    @margins.setter
    def margins(self, margins):
        assert isinstance(margins, (list, tuple)), \
            TypeError("It should be a sequence of OT distribution objects.")

        self._margins_inv_CDF = []
        for marginal in margins:
            assert isinstance(marginal, ot.DistributionImplementation), \
                TypeError("Must be an OpenTURNS distribution objects.")
            self._margins_inv_CDF.append(marginal.computeQuantile)
                
        self._margins = margins
        self._input_dim = len(margins)
        self._corr_dim = self._input_dim * (self._input_dim - 1) / 2

    @property
    def families(self):
        """The copula families.
        """
        return self._families

    @families.setter
    def families(self, families):
        """
        """
        if isinstance(families, str):
            # It should be a path to a csv file
            # TODO: replace pandas with numpy
            matrix = pd.read_csv(families, index_col=0).values
        else:
            matrix = families

        matrix = matrix.astype(int) # Convert elements to integers
        check_matrix(matrix) # Check if the matrix is ok

        self._families = matrix
        self._family_list = []
        self._n_pairs = 0
        self._pairs = []
        self._pairs_ij = []
        k = 0
        for i in range(1, self._input_dim):
            for j in range(i):
                self._family_list.append(matrix[i, j])
                if matrix[i, j] > 0:
                    self._pairs.append(k)
                    self._n_pairs += 1
                    self._pairs_ij.append([i, j])
                k += 1

        self._copula_converters = [Conversion(family) for family in self._family_list]

    @property
    def corr_dim_(self):
        """The number of pairs.
        """
        return self._corr_dim

    @property
    def pairs_(self):
        """The possibly dependent pairs.
        """
        return self._pairs

    @property
    def n_pairs_(self):
        """The number of possibly dependent pairs.
        """
        return self._n_pairs

    @property
    def copula_type(self):
        """The type of copula.
        """
        return self._copula_type

    @copula_type.setter
    def copula_type(self, value):
        assert isinstance(value, str), \
            TypeError('Type must be a string. Type given:', type(value))
            
        if value == "normal":
            families = self._families
            # Warn if the user added a wrong type of family
            if (families[self._families != 0] != 1).any():
                warnings.warn('Some families were not normal and you want an elliptic copula.')
            
            # Set all to families to normal
            families[self._families != 0] = 1
            self.families = families
        self._copula_type = value

    @property
    def vine_structure(self):
        return self._vine_structure

    @vine_structure.setter
    def vine_structure(self, structure):
        if structure is None:
            # TODO: The structure is standard, think about changing it.
            dim = self._input_dim
            structure = np.zeros((dim, dim), dtype=int)
            for i in range(dim):
                structure[i, 0:i+1, ] = i + 1
        else:
            check_matrix(structure)
        self._vine_structure = structure

    @property
    def input_dim(self):
        return self._input_dim
    
    @property
    def bounds_tau(self):
        return self._bounds_tau

    @bounds_tau.setter
    def bounds_tau(self, value):
        """Set the upper bound of the Kendall Tau parameter space.

        Parameters
        ----------
        bounds : :class:`~numpy.ndarray`, str or None
            Matrix of bounds.
        """
        
        dim = self._input_dim
        # If no bounds given, we take the min and max, depending on the copula family
        if value is None:
            bounds = np.zeros((dim, dim))
            for i, j in self._pairs_ij:
                bounds[i, j], bounds[j, i] = get_tau_interval(self._families[i, j])
        elif isinstance(value, str):
            # It should be a path to a csv file
            bounds = pd.read_csv(value, index_col=0).values
        else:
            bounds = value

        bounds_list = []
        for i, j in self._pairs_ij:
            tau_min, tau_max = get_tau_interval(self._families[i, j])
            if np.isnan(bounds[i, j]):
                tau_min = tau_min
            else:
                tau_min = max(bounds[i, j], tau_min)
            if np.isnan(bounds[j, i]):
                tau_max = tau_max
            else:
                tau_max = min(bounds[j, i], tau_max)
            bounds_list.append([tau_min, tau_max])

        check_matrix(bounds)
        self._bounds_tau = bounds
        self._bounds_tau_list = bounds_list

    @property
    def output_sample_(self):
        if self._output_dim == 1:
            return self._all_output_sample
        else:
            return self._all_output_sample[:, self._output_ID]

    @output_sample_.setter
    def output_sample_(self, value):
        raise EnvironmentError("You cannot set this variable")

    @property
    def input_sample_(self):
        return self._input_sample

    @input_sample_.setter
    def input_sample_(self, value):
        raise EnvironmentError("You cannot set this variable")

    @property
    def all_params_(self):
        # There is a compromise between speed and memory efficiency.
        params = np.zeros((self._n_sample, self._corr_dim))
        n = self._n_input_sample
        for i, param in enumerate(self._params):
            params[n*i:n*(i+1), :] = param
        return params

    @property
    def reshaped_output_sample_(self):
        return self.output_sample_.reshape((self._n_param, self._n_input_sample))

    @reshaped_output_sample_.setter
    def reshaped_output_sample_(self, value):
        raise EnvironmentError("You cannot set this variable")

    @property
    def kendalls_(self):
        kendalls = np.zeros((self._n_param, self._corr_dim))
        for k in self._pairs:
            kendalls[:, k] = self._copula_converters[k].to_Kendall(self._params[:, k])
        return kendalls

    @property
    def fixed_params(self):
        return self._fixed_params

    @fixed_params.setter
    def fixed_params(self, value):
        """
        Setter of the matrix of fixed params.
        """
        if value is None:
            # There is no fixed pairs
            matrix = np.zeros((self._input_dim, self._input_dim), dtype=float)
            matrix[:] = None
        elif isinstance(value, str):
            # It should be a path to a csv file
            matrix = pd.read_csv(value, index_col=0).values
        else:
            matrix = value
            
        # The matrix should be checked
        check_matrix(matrix)
            
        # The lists only contains the fixed pairs informations
        self._fixed_pairs = []
        self._fixed_params = matrix
        self._fixed_params_list = []
        k = 0
        for i in range(1, self._input_dim):
            for j in range(i):
                if self._families[i, j] > 0:
                    if matrix[i, j] == 0.:
                        warnings.warn('The parameter of the pair %d-%d is set to 0. Check if this is correct.' % (i, j))
                    if not np.isnan(matrix[i, j]):
                        # The pair is fixed we add it in the list
                        self._fixed_pairs.append(k)
                        self._fixed_params_list.append(matrix[i, j])
                        # And we remove it from the list of dependent pairs
                        self._pairs.remove(k)
                        self._pairs_ij.remove([i, j])
                        self._n_pairs -= 1
                k += 1


class ListDependenceResult(list):
    """The result from the Conservative Estimation.
    
    Parameters
    ----------
    """
    def __init__(self, dep_params=None, output_samples=None, input_samples=None, 
                 q_func=None, run_type=None, n_evals=None,
                 families=None, random_state=None, saved_nevals=0):
        
        self.rng = check_random_state(random_state)
        # TODO: add a setter in the class
        self.q_func = q_func

        if dep_params is not None:
            for k, dep_param in enumerate(dep_params):
                input_sample = None if input_samples is None else input_samples[k]
    
                result = DependenceResult(dep_param=dep_param, input_sample=input_sample, 
                                          output_sample=output_samples[k], q_func=q_func, families=families, random_state=self.rng)
                self.append(result)

        self.families = families
        self.input_dim = self.families.shape[0]
        self.corr_dim = self.input_dim * (self.input_dim - 1) / 2
        self._bootstrap_samples = None
        self.saved_nevals = saved_nevals

    def extend(self, value):
        super(ListDependenceResult, self).extend(value)
        self.families = value.families

    @property
    def pairs(self):
        """
        """
        if self.families is None:
            print('Family matrix was not defined')
        else:
            return to_list(self.families)
    
    @property
    def dep_params(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return [result.dep_param for result in self]
        
    @property
    def kendalls(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return [result.dep_param for result in self]
        
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
            # TODO: Must be changed if the number of sample is different for each param
            return np.asarray([result.output_sample for result in self])
    
    @property
    def n_input_sample(self):
        if self.n_params == 0:
            return 0
        else:  
            return self.output_samples.shape[1]

    @property
    def n_evals(self):
        return self.n_params*self.n_input_sample - self.saved_nevals

    @property
    def n_params(self):
        return len(self)
    
    @property
    def quantities(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.quantity for result in self])

    @property
    def min_result(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return self[self.quantities.argmin()]

    @property
    def min_quantity(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return self.quantities.min()

    @property
    def argmin_quantity(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return self[self.quantities.argmin()].dep_param
    
    @property
    def full_dep_params(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return np.asarray([result.full_dep_params for result in self])

    @property
    def bootstrap_samples(self):
        if self._bootstrap_samples is None:
            print('Bootstrap not computed')
        else:
            return self._bootstrap_samples

    def compute_bootstraps(self, n_bootstrap=1000):
        if self.n_params == 0:
            print("There is no data...")
        else:
            self._bootstrap_samples = np.asarray([bootstrap(result.output_sample, n_bootstrap, self.q_func) for result in self])


class DependenceResult(object):
    """Result from conservative estimate.
    """
    def __init__(self, dep_param=None, input_sample=None, output_sample=None, 
                 q_func=None, run_type=None, families=None, random_state=None):
        self.dep_param = dep_param
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.q_func = q_func
        self.run_type = run_type
        self.families = families
        self.rng = check_random_state(random_state)
        self.bootstrap_sample = None

    def compute_bootstrap(self, n_bootstrap=1000):
        """
        """
        self.bootstrap_sample = bootstrap(self.output_sample, n_bootstrap, self.q_func)

    @property
    def quantity(self):
        return self.q_func(self.output_sample, axis=0)

    @property
    def full_dep_params(self):
        dim = self.families.shape[0]
        corr_dim = dim * (dim - 1) / 2
        full_params = np.zeros((corr_dim, ))
        pairs = to_list(self.families)
        full_params[pairs] = self.dep_param
        return full_params

    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, value):
        assert isinstance(value, dict), \
            TypeError("It should be a dictionnary")

        self._quantity_name = value['Quantity Name']
        self._confidence_level = value['Confidence Level']
        self._estimation_method = value['Estimation Method']

        if self._quantity_name == 'Probability':
            self._threshold = value['Threshold']
            self._operator = value['Operator']
        elif self._quantity_name == 'Quantile':
            self._alpha = value['Quantile Probability']

        self._configs = value

    def __str__(self):
        to_print = '%s: %s\n' % (self._quantity_name, self.quantity)
        if self.confidence_interval is not None:
            to_print += '%s confidence interval at %d %%: %s' % \
                (self._quantity_name, self._confidence_level *
                 100, self.confidence_interval)
        return to_print

    def draw_bounds(self, indep_quant=None, figsize=(10, 6)):
        """
        """
        min_quantile = self.quantity.min(axis=1)
        max_quantile = self.quantity.max(axis=1)

        fig, ax = plt.subplots()
        if indep_quant is not None:
            ax.plot(indep_quant._alpha, indep_quant.quantity, 'b')
        ax.plot(self._alpha, min_quantile, '--b')
        ax.plot(self._alpha, max_quantile, '--b')

        ax.set_ylabel(self._quantity_name)
        ax.set_xlabel('$\\alpha$')
        ax.axis('tight')
        fig.tight_layout()

    def draw(self, id_alpha=0, dep_meas="KendallTau", figsize=(10, 6), 
             color_map="jet", max_eps=1.E-1,
             savefig=False, figpath='.'):
        """Draw the quantity with the dependence measure or copula
        parameter
        """
        obj = self._dependence
        copula_params = self._cond_params
        n_param, n_pairs = copula_params.shape

        assert n_pairs in [1, 2, 3],\
            EnvironmentError("Cannot draw the quantity for dim > 3")

        if dep_meas == "KendallTau":
            params = np.zeros((n_param, n_pairs))
            for i, k in enumerate(obj._pairs):
                params[:, i] = obj._copula_converters[k].to_Kendall(copula_params[:, i])
            param_name = "\\tau"
        elif dep_meas == "CopulaParam":
            params = copula_params
            param_name = "\\rho"
        elif dep_meas == "PearsonRho":
            raise NotImplementedError("Still not done")
            params = None
            param_name = "\\rho^{Pearson}"
        else:
            raise AttributeError("Undefined param")

        
        # Output quantities of interest
        quantity = self.quantity[id_alpha, :].reshape(n_param, -1)
        if self.confidence_interval is not None:
            interval = self.confidence_interval[id_alpha, :]
        else:
            interval = None
        quantity_name = self._quantity_name

        # Find the "almost" independent configuration
        if n_pairs == 1:
            id_indep = (np.abs(params)).argmin()
        else:
            id_indep = np.abs(params).sum(axis=1).argmin()

        # Independent parameter and quantile
        indep_param = params[id_indep]
        indep_quant = quantity[id_indep]

        # If we have confidence interval
        if interval is not None:
            low_bound = quantity - interval
            up_bound = quantity + interval
            indep_quant_l_bound = low_bound[id_indep]
            indep_quant_u_bound = up_bound[id_indep]

        # If it's greater than the tolerence, no need to show it
        if np.sum(indep_param) > max_eps:
            print_indep = False
        else:
            print_indep = True

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if n_pairs == 1:  # One used correlation parameter
            ax = fig.add_subplot(111)  # Create the axis object

            # Ids of the sorted parameters for the plot
            id_sorted_params = np.argsort(params, axis=0).ravel()

            # Plot of the quantile conditionally to the correlation parameter
            ax.plot(params[id_sorted_params], quantity[id_sorted_params, :],
                    'ob', label=quantity_name, linewidth=2)

            # Plot the confidence bounds
            if interval is not None:
                ax.plot(params[id_sorted_params], up_bound[id_sorted_params],
                        '--b', label=quantity_name + " confidence",
                        linewidth=2)
                ax.plot(params[id_sorted_params], low_bound[id_sorted_params],
                        '--b', linewidth=2)

            # Print a line to distinguish the difference with the independence
            # case
            if print_indep:
                p_min, p_max = params.min(), params.max()
                ax.plot([p_min, p_max], [indep_quant] * 2, "r-",
                        label="Independence")
                if interval is not None:
                    ax.plot([p_min, p_max], [indep_quant_l_bound] * 2, "r--")
                    ax.plot([p_min, p_max], [indep_quant_u_bound] * 2, "r--")

            i, j = obj._pairs_ij[0][0], obj._pairs_ij[0][1]
            ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
            ax.set_ylabel(quantity_name)
            ax.legend(loc="best")

        elif n_pairs == 2:  # For 2 correlation parameters
            view = "3d"
            if view == "3d":
                # Dependence parameters values
                r1, r2 = params[:, 0], params[:, 1]

                # 3d ax
                ax = fig.add_subplot(111, projection='3d')
                # Draw the point with the colors
                ax.scatter(r1, r2, quantity, s=40)

                # Plot the confidence bounds
                if interval is not None:
                    ax.plot_trisurf(r1, r2, low_bound,
                                    color="red", alpha=0.05, linewidth=1)
                    ax.plot_trisurf(r1, r2, up_bound, color="red",
                                    alpha=0.05, linewidth=1)
                    #ax.plot(r1, r2, up_bound, 'r.')
                    #ax.plot(r1, r2, low_bound, 'r.')

                # Print a line to distinguish the difference with the
                # independence case
                if print_indep:
                    p1_min, p1_max = r1.min(), r1.max()
                    p2_min, p2_max = r2.min(), r2.max()
                    p1_ = np.linspace(p1_min, p1_max, 3)
                    p2_ = np.linspace(p2_min, p2_max, 3)
                    p1, p2 = np.meshgrid(p1_, p2_)
                    q = np.zeros(p1.shape) + indep_quant
                    ax.plot_wireframe(p1, p2, q, color="red")

                    if interval is not None:
                        q_l = np.zeros(p1.shape) + indep_quant_l_bound
                        q_u = np.zeros(p1.shape) + indep_quant_u_bound
                        ax.plot_wireframe(p1, p2, q_l, color="red")
                        ax.plot_wireframe(p1, p2, q_u, color="red")
                    #    ax.plot([p_min, p_max], [indep_quant_l_bound] * 2,
                    #    "r--")
                    #    ax.plot([p_min, p_max], [indep_quant_u_bound] * 2,
                    #    "r--")
                # Labels
                i, j = obj._pairs_ij[0][0], obj._pairs_ij[0][1]
                ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
                i, j = obj._pairs_ij[1][0], obj._pairs_ij[1][1]
                ax.set_ylabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)

                ax.set_zlabel(quantity_name)

        elif n_pairs == 3:  # For 2 correlation parameters
            color_scale = quantity.ravel()
            cm = plt.get_cmap(color_map)
            c_min, c_max = min(color_scale), max(color_scale)
            cNorm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
            scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            x, y, z = params[:, 0], params[:, 1], params[:, 2]

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=scalar_map.to_rgba(color_scale), s=40)
            scalar_map.set_array(color_scale)
            cbar = fig.colorbar(scalar_map)
            if print_indep:
                pos = cbar.ax.get_position()
                cbar.ax.set_aspect('auto')
                ax2 = cbar.ax.twinx()
                ax2.set_ylim([c_min, c_max])
                width = 0.05
                pos.x0 = pos.x1 - width
                ax2.set_position(pos)
                cbar.ax.set_position(pos)
                n_label = 5
                labels_val = np.linspace(c_min, c_max, n_label).tolist()
                labels = [str(round(labels_val[i], 2)) for i in range(n_label)]
                labels_val.append(indep_quant)
                labels.append("Indep=%.2f" % indep_quant)
                ax2.set_yticks([indep_quant])
                ax2.set_yticklabels(["Indep"])

            ax.set_xlabel("$%s_{12}$" % (param_name), fontsize=14)
            ax.set_ylabel("$%s_{13}$" % (param_name), fontsize=14)
            ax.set_zlabel("$%s_{23}$" % (param_name), fontsize=14)
        
        # Other figure stuffs
        title = r"%s - $n = %d$" % (quantity_name, obj._n_input_sample)
        ax.set_title(title, fontsize=18)
        ax.axis("tight")
        fig.tight_layout()
    
        # Saving the figure
        if savefig:
            fname = figpath + '/'
            if isinstance(savefig, str):
                fname += savefig
            else:
                fname += "fig" + quantity_name
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")
            
        return ax


def to_matrix(param, dim):
    """
    """

    matrix = np.zeros((dim, dim))
    k = 0
    for i in range(dim):
        for j in range(i):
            matrix[i, j] = param[k]
            k += 1

    return matrix

def to_list(matrix):
    """
    """
    params = []
    k = 0
    dim = matrix.shape[0]
    for i in range(dim):
        for j in range(i):
            if matrix[i, j] > 0:
                params.append(k)
            k += 1

    return params

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def bootstrap(data, num_samples, statistic):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
    
    Inspired from: http://people.duke.edu/~ccc14/pcfb/analysis.html"""
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, axis=1))
    return stat

def create_bounds_grid(dimensions):
    """
    """
    n_pair = len(dimensions)
    params = list(itertools.product([-1., 1., 0.], repeat=n_pair))
    params.remove((0.,)*n_pair) # remove indepencence
    params = np.asarray(params)
    for p in range(n_pair):
        params_p = params[:, p]
        params_p[params_p == -1.] = dimensions[p][0]
        params_p[params_p == 1.] = dimensions[p][1]
        
    return params

def get_dependence_parameters(dimensions, n_dep_param, grid_type):
    """
    """
    if n_dep_param is None:
        # We take the bounds
        params = create_bounds_grid(dimensions)
        n_dep_param = len(params)
    else:
        # We create the grid
        space = Space(dimensions)
        params = space.rvs(n_dep_param, sampling=grid_type)

    return params
