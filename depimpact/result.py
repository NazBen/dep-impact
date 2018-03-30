import numpy as np
import os
import h5py
import json

from .utils import asymptotic_error_quantile, bootstrap,\
    dict_to_margins, margins_to_dict, matrix_to_list

from sklearn.utils import check_random_state
from scipy.stats import gaussian_kde, norm

from .vinecopula import Conversion

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
            density = kde_estimate(self.quantity)[0]
            error = asymptotic_error_quantile(
                self.n_sample, density, quantity_param)
        elif quantity_name == 'probability':
            error = asymptotic_error_quantile(
                self.n_sample, quantity, quantity_param)
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

