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
import pyDOE
from scipy.stats import norm, rv_discrete
import nlopt
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from .gridsearch import Space

from sklearn.utils import check_random_state

from .vinecopula import VineCopula, check_matrix
from .conversion import Conversion, get_tau_interval
from .correlation import create_random_kendall_tau, check_params

OPERATORS = {">": operator.gt, ">=": operator.ge,
             "<": operator.lt, "<=": operator.le}


class ConservativeEstimate(object):
    """
    Conservative estimation, toward dependencies, of a quantity of interest at the output of a computational model.

    In a problem with incomplete dependence information, one can try to determine
    the worst case scenario of dependencies according to a certain risk. 
    The dependence structure is described using the copula theory. 
    Vines Copulas can be used to build multidimensional copulas.

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

    vine_structure : :class:`~numpy.ndarray` or None, optional (default=None)
        The Vine copula structure matrix. It describes the construction of the
        vine tree.
        If None, a default matrix is created.

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
        self.fixed_params = fixed_params
        self.bounds_tau = bounds_tau
        self.vine_structure = vine_structure
        self.copula_type = copula_type

        self._forest_built = False
        self._lhs_grid_criterion = 'centermaximin'
        self._grid_folder = './experiment_designs'
        self._dep_measure = None
        self._load_data = False
        self.eps = 1.E-4
        
    @classmethod
    def from_hdf(cls, filepath_or_buffer, id_of_experiment='all', out_ID=0, with_input_sample=True):
        """Load result from HDF5 file.

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
                            random_state=None, use_grid=None, save_grid=None, 
                            done_results=None):
        """
        """
        rng = check_random_state(random_state)
        run_type = 'Classic'
        assert callable(q_func), "Quantity function is not callable"
        
        dimensions = [self._bounds_tau_list[pair] for pair in self.pairs]

        if n_dep_param is None:
            # We take the bounds
            params = create_bounds_grid(dimensions)                
            n_dep_param = len(params)
        else:
            # We create the grid
            space = Space(dimensions)
            params = space.rvs(n_dep_param, sampling=grid_type)
        
        def param_func(param):
            return self.stochastic_function(param, n_input_sample)

        params_not_to_compute = []
        # Some parameters are cancelled has already done the computation
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
            
            print("You saved %d params to compute" % len(params_id_not_to_compute))
            params = np.delete(params, params_id_not_to_compute, axis=0)
            
        # Evaluate the sample
        tmp = map(param_func, params)                
        output_samples = np.asarray(tmp).reshape(n_dep_param, n_input_sample)
        
        # Add back the results
        if len(params_not_to_compute) > 0:
            params = np.r_[params, params_not_to_compute[:, self.pairs]]
            output_samples = np.r_[output_samples, done_results.output_samples]
        
        result = ListDependenceResult(dep_params=params, 
                                      output_samples=output_samples, 
                                      q_func=q_func, run_type=run_type,
                                      families=self.families,
                                      random_state=rng)

        return result

    def stochastic_function(self, param, n_input_sample, random_state=None):
        """This function considers the model output as a stochastic code by 
        taking the dependence parameter as input.
        """
        rng = check_random_state(random_state)

        if isinstance(param, list):
            param = np.asarray(param)
        elif isinstance(param, float):
            param = np.asarray([param])

        assert param.ndim == 1, 'Only one parameter at a time for the moment'

        full_param = np.zeros((self._corr_dim, ))
        full_param[self.pairs] = param
        
        input_sample = self._get_sample(full_param, n_input_sample)
        output_sample = self.model_func(input_sample)

        return output_sample

    def run(self, n_dep_param, n_input_sample, grid='lhs',
            dep_measure="KendallTau", seed=None, use_grid=None, save_grid=None):
        """Generates and evaluates observations of the multiple dependence parameters
        obtained by the discretised space :math:`\Theta_K`.

        The method creates the design space :math:`\Theta_K`, generates
        observations of the input variables for each dependence parameters, and
        evaluate them.

        Parameters
        ----------
        n_param : int
            The number :math:`K` of dependence parameters of :math:`\Theta_K`.

        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf{X}`.

        grid : string, optional (default='lhs')
            The discretisation type. Such as
            - 'lhs': Latin Hypercube Sampling grid,
            - 'fixed': regular grid,
            - 'rand': random Monte-Carlo grid.

        dep_measure : string, optional (default="KendallTau")
            The discretisation can be difficult for some copula, where their parameter
            supports are non-bounded. The use of concordance measure is therefore needed.
            Available dependence measures: =
            - "KendallTau": the Tau Kendall parameter,
            - "PearsonRho": the Pearson Rho parameter,
            - "SpearmanRho": the Spearman Rho parameter.

        out_ID : int, optional (default=0)
            The index of the output if the function output is multidimensional.

        seed : int or None, optional (default=None)
            If int, ``seed`` is the seed used by the random number generator;
            If None, ``seed`` is the seed is random.

        use_grid : string or None, optional (default=None)
            The path to the `csv` file of the grid of dependence parameters.
            If None, the grid is generated.

        save_grid : string or None, optional (default=None)
            The path to the `csv` file to save the grid.
            If None, the grid is not saved.

        Attributes
        ----------
        input_sample_ : :class:`~numpy.ndarray`
            The input sample

        output_sample_ : :class:`~numpy.ndarray`
            The output sample from the model.

        all_params_ : :class:`~numpy.ndarray`
            The dependence parameters associated to each output observation.
        """
        # Set the seed
        if seed is not None:
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)

        # Creates the sample of dependence parameters
        self._generate_dependence_grid(n_dep_param, grid, dep_measure,
                                       use_grid, save_grid)
        
        # If some pairs parameters are fixed
        if self._fixed_pairs:
            self._params[:, self._fixed_pairs] = self._fixed_params_list

        self._build_and_run(n_input_sample)
        self._run_type = 'Classic'

    def minmax_run(self, n_input_sample, seed=None, eps=1.E-8, with_indep=True):
        """Generates and evaluates observations for dependence parameters
        of perfect dependences.

        Parameters
        ----------
        The method creates the design space :math:`\boldsymbol \Theta_K`, generates
        observations of the input variables for each dependence parameters, and
        evaluate them.

        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf{X}`.

        seed : int or None, optional (default=None)
            If int, ``seed`` is the seed used by the random number generator;
            If None, ``seed`` is the seed is random.

        eps : float, optional (default=1.E-8)
            How far we are from the perfect dependence.
        """
        # Set the seed
        if seed is not None:
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)
            
        p = self._n_pairs
        if with_indep:
            grid = [-1. + eps, 1. - eps, 0.]
        else:            
            grid = [-1. + eps, 1. - eps]

        self._n_param = len(grid)**p
        self._params = np.zeros((self._n_param, self._corr_dim))

        tmp = tuple(itertools.product(grid, repeat=p))
        self._params[:, self._pairs] = np.asarray(tmp, dtype=float)

        # If some pairs parameters are fixed
        if self._fixed_pairs:
            self._params[:, self._fixed_pairs] = self._fixed_params_list

        self._build_and_run(n_input_sample)
        self._run_type = 'Perfect Dependence'

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

    def run_custom_param(self, param, n_input_sample, seed=None):
        """Generates and evaluates observations for custom dependence
        parameters.

        Parameters
        ----------
        param : :class:`~numpy.ndarray`
            The dependence parameters.

        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf{X}`.

        seed : int or None, optional (default=None)
            If int, ``seed`` is the seed used by the random number generator;
            If None, ``seed`` is the seed is random.

        """
        if seed is not None:  # Initialises the seed
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)
            
        if isinstance(param, list):
            param = np.asarray(param).reshape(1, -1)

        if isinstance(param, float):
            param = np.asarray([param]).reshape(1, -1)
            
        if param.ndim == 1:
            param = param.reshape(1, -1)
            
        self._n_param = param.shape[0]

        self._params = param
        self._build_and_run(n_input_sample)
        self._run_type = 'Custom'

    def run_gradient_descent(self, alpha, n, init_param=None, max_iter=100, step_g=0.1, gamma_order=0.5):
        """Gradient descent method using KW algorithm.
        """
        # Independence is taking if no starting point given
        if init_param is None:
            init_param = np.zeros((self._n_pairs, ))

        n_pairs = self._n_pairs
        boundaries = self._bounds_tau
        #TODO : make it work with multi dimensionnal boundaries

        # if 'normal', the boundaries are for the moment, the set of correlation matrix
        boundaries_grad = 'normal' if self._copula_type == 'normal' else boundaries
        boundaries_grad = boundaries if n_pairs == 1 else boundaries_grad

        np.random.seed(0)

        all_theta = np.zeros((max_iter, n_pairs))
        quant_theta = np.zeros((max_iter, ))
        all_theta[0] = theta[impact.pairs]
        quant_theta[0] = compute_quantile(impact, theta, n, alpha)

        for i in range(max_iter):
            gamma = (i+1)**gamma_factor
            grad = compute_gradient(impact, theta, n, eps, alpha, boundaries_grad, families)
            theta = theta - gamma * grad # New theta
            theta = get_in_bound(theta, boundaries_grad, families, axis='all') # Get in bound
    
            all_theta[i] = theta[impact.pairs]
            quant_theta[i] = compute_quantile(impact, theta, n, alpha)
            if not (max_iter % (i+1)):
                print 'Iteration: %d' % (i+1)

    def run_iterative(self, alpha, n, K=None, max_n_pairs=5, seed=None):
        """Iteratively construct the worst case dependence structure.
        """
        if seed is not None:  # Initialises the seed
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)

        dim = self._input_dim
        init_family = self.families
        # First run for independence
        self.run_independence(n, seed)
        families = np.zeros((dim, dim))
        indep_quantile = self.compute_quantiles(alpha).quantity.ravel()
        selected_pairs = []
        worst_quantiles = []
        indiv_pair_quantiles = {}
        for k in range(max_n_pairs):
            quantiles = {}
            for i in range(1, dim):
                for j in range(i):
                    if [i, j] not in selected_pairs:
                        tmp_families = np.copy(families)
                        tmp_families[i, j] = init_family[i, j]
                        self.families = tmp_families
                        #print k, i, j, families
                        if K is None:
                            self.minmax_run(n)
                        else:
                            self.run(K, n)
                        quantiles[i, j] = self.compute_quantiles(alpha).quantity.max()
                        if k == 0:
                            indiv_pair_quantiles[i, j] = quantiles[i, j]
            selected_pair = max(quantiles, key=quantiles.get)
            i, j = selected_pair[0], selected_pair[1]
            families[i, j] = init_family[i, j]
            selected_pairs.append(selected_pair)
            worst_quantiles.append(quantiles[selected_pair])

        return selected_pairs, worst_quantiles, indiv_pair_quantiles, indep_quantile

    def run_tree_minimisation(self, alpha, n=1, K=20, n_iters=30, 
                              coeff=0.5, ratio=10, 
                              leaf_selection='probabilistic', seed=None, 
                              verbose=True, with_plot=False, with_true_quant=None):
        """Gets the minimum quantile of the domain with an iterative sampling using quantile regression trees.
        """
        # Initialising the seed
        if seed is not None:
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)

        if with_plot and self._n_pairs in [1, 2]:
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            ax_quant = axes[0]
            ax_quant.set_ylabel('$Q (\\alpha)$', color='r')
            for tl in ax_quant.get_yticklabels():
                tl.set_color('r')
            ax_utility = ax_quant.twinx()
            ax_utility.set_ylabel('Utility', color='b')
            for tl in ax_utility.get_yticklabels():
                tl.set_color('b')
            ax_density = axes[1]
            ax_density.set_ylabel('Density of new points')

            ax_hist = axes[2]
            ax_hist.set_xlabel('$\\theta$')
            ax_hist.set_ylabel('Density of all points')

        self.run(K, n, grid='lhs')
        params = self.params_
        out_sample = self.reshaped_output_sample_

        all_params = np.concatenate([params]*n)
        all_out_sample = out_sample.ravel()

        # TODO: a problem can occurs if some params are in double
        dict_out = {}
        for k, param in enumerate(params):
            dict_out[str(param)] = out_sample[k].tolist()

        size_leaf = K*n/ratio
        use_gradient_boosting = False

        for k in range(n_iters):
            # Create and fit the forest
            if not use_gradient_boosting:
                forest = DecisionTreeRegressor(min_samples_leaf=size_leaf)
            else:
                forest = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=100, min_samples_leaf=size_leaf, max_leaf_nodes=K)
            forest.fit(all_params, all_out_sample)

            # The unique params
            params_k = unique_rows(all_params)

            # Estimate quantiles
            if not use_gradient_boosting:
                quantiles_k = forest.predict_quantile(params_k, alpha)
            else:
                quantiles_k = forest.predict(params_k)

            # Get zones by looking to the unique quantiles
            unique_quantiles = np.unique(quantiles_k)

            # Find the min quantile
            id_min = np.where(quantiles_k == quantiles_k.min())[0]

            # Shape of the zones
            ids_in_zones = [np.where(quant == quantiles_k)[0].tolist() for quant in unique_quantiles]

            # Compute the utility
            n_zones = unique_quantiles.shape[0]

            zone_shapes = []
            for ids_zone in ids_in_zones:
                min_zone = params_k[ids_zone].min(axis=0)
                max_zone = params_k[ids_zone].max(axis=0)
                zone_shapes.append(np.c_[min_zone, max_zone])

            if n_zones > 1:
                utilities = compute_utility(coeff, quantiles_k, params_k, dict_out, unique_quantiles, ids_in_zones)
            else:
                utilities = np.ones((1,))

            # Posterior distribution
            density = utilities/utilities.sum()

            # Random variable of the density
            # TODO: make it works for p > 1
            if leaf_selection == 'deterministic':
                used_utility = utilities.argmax()
                zone_shape = zone_shapes[used_utility]
                new_params = np.zeros((K, self._corr_dim))
                for i in self._pairs:
                    shape = zone_shape[i, :]
                    new_params[:, i] = np.random.uniform(shape[0], shape[1], K)
            elif leaf_selection == 'probabilistic':
                rv = rv_discrete(values=(range(n_zones), density))
                used_utility = int(rv.rvs(size=1))
                zone_shape = zone_shapes[used_utility]
                new_params = np.zeros((K, self._corr_dim))
                for i in self._pairs:
                    shape = zone_shape[i, :]
                    new_params[:, i] = np.random.uniform(shape[0], shape[1], K)
            elif leaf_selection == 'full-probabilistic':
                rv = rv_discrete(values=(range(n_zones), density))
                new_params = np.zeros((K, self._corr_dim))
                for j, used_utility in enumerate(rv.rvs(size=K)):
                    zone_shape = zone_shapes[used_utility]
                    for i in self._pairs:
                        shape = zone_shape[i, :]
                        new_params[j, i] = np.random.uniform(shape[0], shape[1])
            else:
                raise NotImplementedError('Unknow method')

            self.run_custom_param(new_params, n)
            best_out_sample = self.output_sample_

            for i, param in enumerate(new_params):
                if str(param.tolist()) in dict_out:
                    dict_out[str(param)].append(best_out_sample[i])
                else:
                    dict_out[str(param)] = best_out_sample[i].reshape(-1, 1).tolist()

            p_all_params = all_params
            all_params = np.r_[all_params, np.repeat(new_params, n, axis=0)]
            all_out_sample = np.r_[all_out_sample, best_out_sample]
            size_leaf = max(all_out_sample.shape[0]/ratio, 1)

            min_params = params_k[id_min]
            if verbose:
                print 'Min quantile: %.3f in leaf [%.2f, %.2f]' % (quantiles_k.min(), min_params.min(), min_params.max())
                print 'Selected leaf in [%.2f, %.2f]' % (new_params.min(), new_params.max())
                print

            if with_plot and self._n_pairs in [1, 2]:
                ax_quant.lines = []
                ax_quant.set_ylim(quantiles_k.min(), quantiles_k.max())
                ax_quant.plot(params_k, quantiles_k, 'r.')
                if with_true_quant is not None:
                    ax_quant.plot(with_true_quant[0], with_true_quant[1], 'g-')

                ax_utility.lines = []
                #ax_utility.set_ylim(0, 1.05)
                for i, utility in enumerate(density):
                    x_utility = [zone_shapes[i][:, 0], zone_shapes[i][:, 1]]
                    y_utility = [utility]*2
                    ax_utility.plot(x_utility, y_utility, 'b-', linewidth=2)

                ax_density.cla()
                ax_density.set_ylabel('Density of new points')
                ax_density.hist(new_params, bins=ratio*3, color='blue')

                ax_hist.cla()
                ax_hist.set_xlabel('$\\theta$')
                ax_hist.set_ylabel('Density of all points')
                ax_hist.hist(p_all_params, bins=100, color='blue')
                fig.canvas.draw()

#            ratio += 1
#            coeff -= 0.01
            coeff = max(0., coeff)
            print('N evals : %d' % (all_out_sample.shape[0]))
            print 'Min quantile : %.3f at ' % (unique_quantiles.min()), zone_shapes[unique_quantiles.argmin()][self._pairs, :]
            self._all_output_sample = all_out_sample
            self._all_params= all_params

    def _build_and_run(self, n_input_sample):
        """Creates the input sample of each dependence parameters
        and evaluates the observations.

        Parameters
        ----------
        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf{X}`.
        """
        # Creates the sample of input parameters
        self._build_input_sample(n_input_sample)

        # Evaluates the input sample
        self._all_output_sample = self.model_func(self._input_sample)

        # Get output dimension
        self._output_info()

    def func_quant(self, param, alpha, n_input_sample):
        """
        """
        self._n_param = 1
        self._params = np.zeros((1, self._corr_dim), dtype=float)
        self._params[self._pairs] = param

        if self._fixed_pairs is not None:
            self._params[:, self._fixed_pairs] = self._fixed_params_list
        
        self._build_and_run(n_input_sample)
        self._run_type = 'Minimising'
        
        return np.percentile(self.output_sample_, alpha * 100.)

    def minimise_quantile(self, alpha, n_input_sample, theta_init=None, eps=1.E-5):
        """
        """
        if theta_init is not None:
            theta_init = [0.]*self._n_pairs
        def func(param, grad):
            if grad.size > 0:  
                print grad
            return self.func_quant(param, alpha, n_input_sample)
            
        algorithm = nlopt.GN_DIRECT
        opt = nlopt.opt(algorithm, self._n_pairs)
        opt.set_lower_bounds([-0.9])
        opt.set_upper_bounds([0.9])
        opt.maxeval=10
        opt.set_min_objective(func)
        theta_opt = opt.optimize(theta_init)
        return theta_opt

    def _generate_dependence_grid(self, n_param, grid, dep_measure, use_grid, 
                                  save_grid):
        """Generates the dependence parameters.

        The method discretises the dependence parameter support :math:` \Theta`
        into a design of experiment :math:` \Theta_K` of cardinality :math:`K`.
        The discretisation can be made by a regular grid, an LHS design or a 
        random Monte-Carlo. It is also made using a concordance measure such as
        the Kendall Tau, but in some cases, it can also be made using the 
        copula dependence parameter. 

        Parameters
        ----------
        n_param : int
            The number :math:`K` of dependence parameters of
            :math:`\boldsymbol \Theta_K`.

        grid : string
            The discretisation type. Such as
            - 'lhs': Latin Hypercube Sampling grid,
            - 'fixed': regular grid,
            - 'rand': random Monte-Carlo grid.

        dep_measure : string
            The measure used to describe the dependence between variables
            - "KendallTau": the Tau Kendall parameter,
            - "PearsonRho": the Pearson Rho parameter,
            - "SpearmanRho": the Spearman Rho parameter.
            
        use_grid : string or None, optional (default=None)
            The path to the `csv` file of the grid of dependence parameters.
            If None, the grid is generated.

        save_grid : string or None, optional (default=None)
            The path to the `csv` file to save the grid.
            If None, the grid is not saved.
        """
        grid_filename = None
        p = self._n_pairs
        if grid == 'lhs':
            gridname = '%s_crit_%s' % (grid, self._lhs_grid_criterion)
        else:
            gridname = grid

        # If the design is loaded
        if use_grid is not None:
            # Load the sample from file and get the filename
            sample, grid_filename = self._load_grid(n_param, dep_measure, use_grid, gridname)

            meas_param = np.zeros((n_param, self._corr_dim))
            for k, i in enumerate(self._pairs):
                tau_min, tau_max = self._bounds_tau_list[k]
                meas_param[:, i] = sample[:, k]*(tau_max - tau_min) + tau_min
        else:
            if dep_measure == "KendallTau":
                if grid == 'fixed':
                    # Number of configurations per dimension
                    n_p = int((n_param) ** (1./p))
                    if n_p < 3:
                        warnings.warn('There is only %d configuration per dimension' % (n_p))
                        
                    # Creates the p-dim grid
                    v = []
                    for k, i in enumerate(self._pairs):
                        tau_min, tau_max = self._bounds_tau_list[k]
                        v.append(np.linspace(tau_min, tau_max, n_p+1, endpoint=False)[1:])
                    tmp = np.vstack(np.meshgrid(*v)).reshape(p, -1).T

                    # The final total number is not always the initial one.
                    n_param = n_p ** p
                    meas_param = np.zeros((n_param, self._corr_dim))
                    for k, i in enumerate(self._pairs):
                        meas_param[:, i] = tmp[:, k]

                elif grid == 'rand':  # Random grid
                    if self._copula_type == "vine":
                        meas_param = np.zeros((n_param, self._corr_dim))
                        for k, i in enumerate(self._pairs):
                            tau_min, tau_max = self._bounds_tau_list[k]
                            meas_param[:, i] = np.random.uniform(tau_min, tau_max, n_param)
                    elif self._copula_type == "normal":
                        # TODO: correct this
                        meas_param = create_random_kendall_tau(self._families, n_param)
                    else:
                        raise AttributeError('Unknow copula type:', self._copula_type)

                elif grid == 'lhs':
                    meas_param = np.zeros((n_param, self._corr_dim))
                    sample = pyDOE.lhs(p, samples=n_param, 
                                       criterion=self._lhs_grid_criterion)
                    for k, i in enumerate(self._pairs):
                        tau_min, tau_max = self._bounds_tau_list[k]
                        meas_param[:, i] = sample[:, k]*(tau_max - tau_min) + tau_min
                else:
                    raise AttributeError('%s is unknow for DOE type' % grid)
            elif dep_measure == "PearsonRho":
                raise NotImplementedError("Work in progress.")
            elif dep_measure == "SpearmanRho":
                raise NotImplementedError("Not yet implemented")
            else:
                raise AttributeError("Unkown dependence parameter")

        # The grid is save if it was asked and if it does not already exists
        if save_grid is not None and use_grid is None:
            # The grid is saved
            if save_grid is True: # No filename provided
                dirname = self._grid_folder
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                # The sample variable exist only for lhs sampling
                if grid != 'lhs':
                    sample = np.zeros((n_param, p))
                    for k, i in enumerate(self._pairs):
                        tau_min, tau_max = self._bounds_tau_list[k]
                        sample[:, k] = (meas_param[:, i] - tau_min) / (tau_max - tau_min)
                k = 0
                do_save = True
                name = '%s_p_%d_n_%d_%s_%d.csv' % (gridname, p, n_param, dep_measure, k)
                grid_filename = os.path.join(dirname, name)
                # If this file already exists
                while os.path.exists(grid_filename):
                    existing_sample = np.loadtxt(grid_filename).reshape(n_param, -1)
                    # We check if the build sample and the existing one are equivalents
                    if np.allclose(np.sort(existing_sample, axis=0), np.sort(sample, axis=0)):
                        do_save = False
                        print 'The DOE already exist in %s' % (name)
                        break
                    k += 1
                    name = '%s_p_%d_n_%d_%s_%d.csv' % (gridname, p, n_param, dep_measure, k)
                    grid_filename = os.path.join(dirname, name)
                
            # It is saved
            if do_save:
                np.savetxt(grid_filename, sample)
            
        self._grid_filename = grid_filename
        self._dep_measure = dep_measure
        self._grid = grid
        self._n_param = n_param
        self._params = np.zeros((n_param, self._corr_dim))

        from conversion import Conversion
        # Convert the dependence measure to copula parameters
        for i in self._pairs:
            self._params[:, i] = self._copula_converters[i].to_copula_parameter(meas_param[:, i], dep_measure)
            u =  Conversion.NormalCopula.fromKendallToParam(self._params[:, i])

    def _build_input_sample(self, n):
        """Creates the observations of each dependence measure of :math:`\Theta_K`.

        Parameters
        ----------
        n : int
            The number of observations.
        """
        n_sample = n * self._n_param  # Total number of observations
        self._n_sample = n_sample
        self._n_input_sample = n
        res_list = map(lambda param: self._get_sample(param, n), self._params)
        self._input_sample = np.concatenate(res_list)

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

    def _output_info(self):
        """Information about the output dimension.
        """
        # If the output dimension is one
        if self._all_output_sample.shape[0] == self._all_output_sample.size:
            self._output_dim = 1
        else:
            self._output_dim = self._all_output_sample.shape[1]

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
        print 'loading file %s' % name        
        sample = np.loadtxt(filename).reshape(n_param, -1)
        assert n_param == sample.shape[0], 'Wrong grid size'
        assert p == sample.shape[1], 'Wrong dimension'
        return sample, filename

    def build_forest(self, quant_forest=None):
        """Build a Quantile Random Forest to estimate conditional quantiles.

        Parameters
        ----------
        quant_forest : :class:`~QuantileForest`
            A Quantile Regression Forest instance.
        """
        # We take only used params (i.e. the zero cols are taken off)
        # Actually, it should not mind to RF, but it's better for clarity.
        used_params = self.all_params_[:, self._pairs]
        quant_forest.fit(used_params, self.output_sample_)
        self._quant_forest = quant_forest
        self._forest_built = True

    def compute_quantity(self, quantity_func, options, boostrap=False):
        """Compute the output quantity of interest of each dependence parameters.

        Parameters
        ----------
        quantity_func : string or callable
            The function of the quantity of interest. It can be:
            - 'quantile' to compute the quantiles,
            -'"probability' to compute the probability,
            - a custom function that compute the quantity of interest
            given the output sample.

        options : list
            Aditionnal arguments of `quantity_func`.

        bootstrap : bool
            If True, a bootrap is made on the quantity of interest.
        """
        if isinstance(quantity_func, str):
            if quantity_func == "quantile":
                self.compute_quantiles(*options)
                self._quantity = self._quantiles
                if boostrap:
                    raise Exception("Not implemented yet")
                else:
                    self._quantity_interval = None
                    self._quantity_up_bound = None
                    self._quantity_low_bound = None
            if quantity_func == "probability":
                self.compute_probability(*options)
                self._quantity = self._probability
                self._quantity_interval = self._probability_interval
                self._quantity_up_bound = self._probability + self._probability_interval
                self._quantity_low_bound = self._probability - \
                    self._probability_interval
        elif callable(quantity_func):
            out_sample = self.reshaped_output_sample_
            result = quantity_func(out_sample, *options)
            if isinstance(result, tuple):
                self._quantity = result[0]
                self._quantity_interval = result[1]
                self._quantity_up_bound = result[0] + result[1] / 2.
                self._quantity_low_bound = result[0] - result[1] / 2.
            else:
                self._quantity = result[0]
                self._quantity_interval = None
                self._quantity_up_bound = None
                self._quantity_low_bound = None
        else:
            raise TypeError("Unknow input variable quantity_func")

    def compute_probability(self, threshold, estimation_method='empirical',
                            confidence_level=0.95, operator='>', bootstrap=False,
                            output_ID=0):
        """Computes conditional probabilities for each parameters.

        Compute the probability of the current sample for each dependence
        parameter.

        Parameters
        ----------
        threshold : float, int
            The threshold :math:`s` of the output probability :math:`\mathbb P[Y \geq s]`
        estimation_method : string, optional (default='empirical')
            The probability estimation method. Available methods: 
            - 'empirical': Count the number of sample greater of lower to the threshold.
            - 'randomforest': Not yet implemented.
        confidence_level : float, optional (default=0.95)
            The confidence level probability.
        operator : string, optional (default='>')
            The operator of the probability :math:`\mathbb P[Y \geq s]`.
        """
        assert isinstance(threshold, (float, int)), \
            TypeError("Threshold should be a number.")
        assert isinstance(confidence_level, float), \
            TypeError("Confidence Level should be a float")
        assert 0. < confidence_level < 1., \
            ValueError("Confidence level should be a probability")
        assert isinstance(estimation_method, str), \
            TypeError("Method name should be a string")

        self._output_ID = output_ID
        configs = {'Quantity Name': 'Probability',
                  'Threshold': threshold,
                  'Confidence Level': confidence_level,
                  'Estimation Method': estimation_method,
                  'Operator': operator
                 }

        op_func = OPERATORS[operator]

        if estimation_method == "empirical":
            out_sample = self.reshaped_output_sample_
            probability = (op_func(out_sample, threshold).astype(float)).mean(axis=1)
            tmp = np.sqrt(probability * (1. - probability) /
                          self._n_input_sample)
            # Quantile of a Gaussian distribution
            q_normal = norm.ppf((1 + confidence_level) / 2.)
            interval = q_normal * tmp  # Confidence interval
            cond_params = self._params[:, self._pairs]
        elif estimation_method == 'randomforest':
            raise NotImplementedError('Not Yet Done...')
        else:
            raise AttributeError("Method does not exist")

        return DependenceResult(configs, self, probability, interval, cond_params)

    def compute_quantiles(self, alpha, estimation_method='empirical',
                          confidence_level=0.95, grid_size=None, bootstrap=False,
                          output_ID=0):
        """Computes quantiles of each dependence parameters.

        Compute the alpha-quantiles of the current sample for each dependence
        parameter.

        Parameters
        ----------
        alpha : float or :class:`~numpy.ndarray`
            Probability of the quantile.

        estimation_method : string, optional (default='empirical')
            The quantile estimation method. Available methods: 
            - 'empirical': The percentile of the output sample.
            - 'randomforest': Not yet implemented.

        confidence_level : float, optional (default=0.95)
            The confidence level probability.
        """
        assert isinstance(estimation_method, str), \
            TypeError("Method name should be a string")
        if isinstance(alpha, (list, np.ndarray)):
            for alphai in alpha:
                assert 0. < alphai < 1., \
                    ValueError("alpha should be a probability")
            alpha = np.asarray(alpha)
        elif isinstance(alpha, float):
            assert 0. < alpha < 1., \
                ValueError("alpha should be a probability")
        else:
            TypeError("Method name should be a float")
            
        self._output_ID = output_ID
        configs = {'Quantity Name': 'Quantile',
                  'Quantile Probability': alpha,
                  'Confidence Level': confidence_level,
                  'Estimation Method': estimation_method
                 }

        interval = None
        # TODO: correct the bootstrap for the quantile.
        cond_params = self._params[:, self._pairs]
        if estimation_method == 'empirical':
            out_sample = self.reshaped_output_sample_
            if not bootstrap:
                quantiles = np.percentile(out_sample, alpha * 100., axis=1).reshape(1, -1)
            else:
                quantiles, interval = bootstrap_quantile(out_sample, alpha, 20, 0.01)                
            # TODO: think about using the check function instead of the percentile.

        elif estimation_method == "randomforest":
            if not self._forest_built:
                warnings.warn(""""RandomForest has not been manually built. 
                Please use the build_forest method before computing the 
                quantiles.\nThe forest is build with default options for now.""",
                DeprecationWarning)
                self.build_forest()

            if grid_size:
                # Create the grid of params
                grid = np.zeros((self._n_pairs, grid_size))
                for i, k in enumerate(self._pairs):
                    # The bounds are taken from data, no need to go further.
                    p_min = self._copula[k].to_Kendall(self._params[:, k].min())
                    p_max = self._copula[k].to_Kendall(self._params[:, k].max())
                    # The space is filled for Kendall Tau params
                    tmp = np.linspace(p_min, p_max, grid_size+1, endpoint=False)[1:]
                    # Then, it is converted to copula parameters
                    grid[i, :] = self._copula[k].to_copula_parameter(tmp, 'KendallTau')

                # This is all the points from the grid.
                cond_params = np.vstack(np.meshgrid(*grid)).reshape(self._n_pairs,-1).T

            # Compute the quantiles
            quantiles = self._quant_forest.compute_quantile(cond_params, alpha)
        else:
            raise AttributeError(
                "Unknow estimation method: %s" % estimation_method)

        return DependenceResult(configs, self, quantiles, interval, cond_params)

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
                print 'File %s already has different configurations' % (file_name)
                file_name = '%s_%d.hdf5' % (init_file_name[:-5], k)
                k += 1

        print 'Data saved in %s' % (file_name)

        return file_name

    def draw_matrix_plot(self, corr_id=None, copula_space=False, figsize=(10, 10),
                         savefig=False):
        """
        """
        if corr_id is None:
            id_corr = np.ones(self._n_sample, dtype=bool)
        else:
            id_corr = np.where((self.all_params_ == self._params[corr_id]).all(axis=1))[0]

        data = self._input_sample[id_corr]

        if copula_space:
            x = np.zeros(data.shape)
            for i, marginal in enumerate(self._margins):
                for j, ui in enumerate(data[:, i]):
                    x[j, i] = marginal.computeCDF(ui)
        else:
            x = data

        fig, axes = plt.subplots(self._input_dim, self._input_dim, figsize=figsize, sharex='col')

        for i in range(self._input_dim):
            for j in range(self._input_dim):
                ax = axes[i, j]
                xi = x[:, i]
                xj = x[:, j]
                if i != j:
                    ax.plot(xj, xi, '.')
                if i == j:
                    ax.hist(xi, bins=30, normed=True)
                    
                if copula_space:
                    ax.set_xticks([])
                    ax.set_yticks([])

        fig.tight_layout()
        
        if savefig:
            if isinstance(savefig, str):
                fname = savefig + '/'
            else:
                fname = "./"
            fname += "matrix_plot"
            fig.savefig(fname + ".png", dpi=200)

        return fig

    def draw_design_space(self, corr_id=None, figsize=(10, 6),
                          savefig=False, color_map="jet", output_name=None,
                          input_names=None, return_fig=False, color_lims=None):
        """
        """
        assert self._input_dim in [2, 3], "Cannot draw quantiles for dim > 3"

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if corr_id is None:
            id_corr = np.ones(self._n_sample, dtype=bool)
        else:
            id_corr = np.where((self.all_params_ == self._params[corr_id]).all(axis=1))[0]

        if input_names:
            param_name = input_names
        else:
            param_name = ["$x_{%d}$" % (i + 1) for i in range(self._input_dim)]

        if output_name:
            output_label = output_name
        else:
            output_label = "Output value"

        x = self._input_sample[id_corr]
        y = self.output_sample_[id_corr]
        color_scale = y
        cm = plt.get_cmap(color_map)
        if color_lims is None:
            c_min, c_max = min(color_scale), max(color_scale)
        else:
            c_min, c_max = color_lims[0], color_lims[1]
        cNorm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        if self._input_dim == 2:  # If input dimension is 2
            ax = fig.add_subplot(111)  # Creat the ax object
            x1 = x[:, 0]
            x2 = x[:, 1]
            ax.scatter(x1, x2, c=scalarMap.to_rgba(color_scale))
            scalarMap.set_array(color_scale)
            cbar = fig.colorbar(scalarMap)
            cbar.set_label(output_label)
            ax.set_xlabel(param_name[0], fontsize=14)
            ax.set_ylabel(param_name[1], fontsize=14)

        if self._input_dim == 3:  # If input dimension is 3
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x1, x2, x3, c=scalarMap.to_rgba(color_scale), s=40)
            scalarMap.set_array(color_scale)
            cbar = fig.colorbar(scalarMap)
            cbar.set_label(output_label)

            ax.set_xlabel(param_name[0], fontsize=14)
            ax.set_ylabel(param_name[1], fontsize=14)
            ax.set_zlabel(param_name[2], fontsize=14)

        title = "Design Space with $n = %d$ observations" % len(id_corr)
        if corr_id is not None:
            title += "\n$\\rho = "
            p = self._params[corr_id]
            if self._corr_dim == 1:
                title += "%.1f$" % (p)
            elif self._corr_dim == 3:
                p = self._params[corr_id]
                title += "[%.1f, %.1f, %.1f]$" % (p[0], p[1], p[2])
        ax.set_title(title, fontsize=18)
        ax.axis("tight")
        fig.tight_layout()

        if savefig:
            if isinstance(savefig, str):
                fname = savefig + '/'
            else:
                fname = "./"
            fname += "sample_space"
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")

        if return_fig:
            return fig
        else:
            return None

# =============================================================================
# Setters
# =============================================================================
    @property
    def model_func(self):
        """The model function. Must be a callable.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        """
        """
        if callable(func):
            self._model_func = func
        else:
            raise TypeError("The model function must be callable.")

    @property
    def margins(self):
        """The PDF margins. List of :class:`~openturns.Distribution` objects.
        """
        return self._margins

    @margins.setter
    def margins(self, list_margins):
        assert isinstance(list_margins, list), \
            TypeError("It should be a list of margins distribution objects.")

        self._margins_inv_CDF = []
        for marginal in list_margins:
            if isinstance(marginal, ot.DistributionImplementation):
                self._margins_inv_CDF.append(marginal.computeQuantile)
            else:
                TypeError("Must be an OpenTURNS distribution objects.")

        self._margins = list_margins
        self._input_dim = len(list_margins)
        self._corr_dim = self._input_dim * (self._input_dim - 1) / 2

    @property
    def families(self):
        """The copula families.
        """
        return self._families

    @families.setter
    def families(self, value):
        """
        """
        if isinstance(value, str):
            # It should be a path to a csv file
            matrix = pd.read_csv(value, index_col=0).values
        else:
            matrix = value
        matrix = matrix.astype(int)
        check_matrix(matrix)
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
    def pairs(self):
        return self._pairs

    @property
    def n_pairs(self):
        return self._n_pairs

    @property
    def corr_dim(self):
        return self._corr_dim

    @property
    def copula_type(self):
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
    def params_(self):
        return self._params
    @property
    def n_dep_params_(self):
        return self._n_param
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
    """
    """
    def __init__(self, dep_params=None, output_samples=None, input_samples=None, 
                 q_func=None, run_type=None, n_evals=None,
                 families=None, random_state=None):
        
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
        self._bootstrap_samples = None
            
    def extend(self, value):
        super(ListDependenceResult, self).extend(value)
        self.families = value.families
             
    @property
    def pairs(self):
        if self.families is None:
            print('Family matrix was not defined')
        else:
            return to_list(self.families)
        
    @property
    def corr_dim(self):
        dim = self.families.shape[0]
        return dim * (dim - 1) /2
    
    @property
    def dep_params(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            return [result.dep_param for result in self]
    
    @property
    def n_pairs(self):
        if self.n_params == 0:
            return 0
        else:
            return (self.families > 0).sum()
    
    @property
    def output_samples(self):
        if self.n_params == 0:
            print("There is no data...")
        else:
            # TODO: Must be changed is the number of sample is different for each param
            return np.asarray([result.output_sample for result in self])
    
    @property
    def n_input_sample(self):
        if self.n_params == 0:
            return 0
        else:  
            return self.output_samples.shape[1]

    @property
    def n_evals(self):
        return self.n_params*self.n_input_sample
    
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
        self.rng = check_random_state(random_state)
        self.families = families
        self.bootstrap_sample = None

    def compute_bootstrap(self, n_bootstrap=1000):
        """
        """
        self.bootstrap_sample = bootstrap(self.output_sample, n_bootstrap, self.q_func)

    @property
    def quantity(self):
        return self.q_func(self.output_sample, axis=0)

    @property
    def cond_params(self):
        return self._cond_params

    @cond_params.setter
    def cond_params(self, value):
        self._cond_params = value
        
    @property
    def full_dep_params(self):
        dim = self.families.shape[0]
        corr_dim = dim * (dim - 1) / 2
        full_params = np.zeros((corr_dim, ))
        pairs = to_list(self.families)
        full_params[pairs] = self.dep_param
        return full_params

    @property
    def dependence(self):
        return self._dependence

    @dependence.setter
    def dependence(self, obj):
        # TODO: there is a bug sometimes. I don't know why it makes a difference 
        # between ImpactOfDependence and dependence.dependence.ImpactOfDependence
        #assert isinstance(obj, ImpactOfDependence), \
        #    TypeError('Variable must be an ImpactOfDependence object. Got instead:', type(obj))
        self._dependence = obj

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


def bootstrap_quantile(data, alpha, n_boot, alpha_boot):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = np.random.randint(0, n, (n_boot, n))
    samples = data[idx]
    stat = np.sort(np.percentile(samples, alpha * 100., axis=1))

    return stat.mean(axis=0), np.c_[stat[int((alpha_boot*0.5) * n_boot)],
            stat[int((1. - alpha_boot*0.5) * n_boot)]]


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

def compute_volume(params):
    n, dim = params.shape
    return (params.max(axis=0) - params.min(axis=0)).prod()
    
def sample_density(values):
    """
    """
    u = np.random.rand()
    return rv_discrete(values=(range(values.shape[0]), values/values.sum())).ppf(u)
    
def compute_utility(coeff, quantiles_k, params_k, dict_out, unique_quantiles, ids_in_zones):
    """
    """                    
    # The zones
    n_zones = len(unique_quantiles)
    info_sample = np.zeros((n_zones, ))
    info_quant = np.zeros((n_zones, ))
    utilities = np.zeros((n_zones, ))
    
    # For each zone
    for i in range(n_zones):
        quant = unique_quantiles[i]
        if ids_in_zones is None:
            ids_zone = np.where(quant == quantiles_k)[0]
        else:
            ids_zone = ids_in_zones[i]

        # The number of observation in the leaf
        n_obs_leaf = sum([len(dict_out[str(params_k[id_p])]) for id_p in ids_zone])
        
        # Volume of the zone
        volume = compute_volume(params_k[ids_zone])
        
        # Compute informations 
        # TODO: the volume is neglected with the distribution problem
        info_sample[i] = n_obs_leaf
        info_quant[i] = (quantiles_k.max() - quant ) / (quantiles_k.max() - quantiles_k.min())
        
    info_sample = (info_sample.max() - info_sample) / (info_sample.max() - info_sample.min())
    utilities = coeff*info_sample + (1 - coeff)*info_quant
    return utilities

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