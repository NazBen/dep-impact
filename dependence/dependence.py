# -*- coding: utf-8 -*-
import numpy as np
import openturns as ot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from itertools import combinations
from scipy.stats import rv_continuous

from .vinecopula import VineCopula, check_matrix
from .conversion import Conversion, get_tau_interval
from .correlation import get_grid_rho, create_random_correlation_param, create_random_kendall_tau


class ImpactOfDependence(object):
    """
    Quantify the impact of dependencies.

    This class study the impact of eventual dependencies of random variables on
    a quantity of interest of the output distribution. Different forms of
    dependencies, using the notion of copulas, are availaible to study the
    relation between the output and the dependencies.

    Parameters
    ----------
    model_func : callable
        The evaluation model such as :math:`Y = g(\mathbf X)`.

    margins : list of :class:`~openturns.Distribution`
        The probabilistic distribution :math:`f_{X_i}` for :math:`i \in 1 \dots d` of the margins.

    copula_name : string, optional (default="NormalCopula")
        The copula name $C$ to describes the dependence structure.
        The copula name must be supported and in the list of available copulas.

    Attributes
    ----------
    rand_vars : :class:`~openturns.ComposedDistribution`
        The random variable :math:`\mathbf X` describing the marginals and
        the copula.
    """
    _load_data = False

    def __init__(self, model_func, margins, families):
        self.model_func = model_func
        self.margins = margins
        self.families = families
        
    @classmethod
    def from_data(cls, data_sample, dim, out_ID=0):
        """Load from data.

        This method initialise the class using built data from previous simulations.

        Parameters
        ----------
        data_sample : :class:`~numpy.ndarray`
            The sample of built data.
        dim : int
            The input dimension.
        out_ID : int, optional (default=0)
            The output ID to take care of when multiple output are available.
        """
        # Creates the class object from ghost parameters.
        def foo(): return None
        obj = cls(foo, ot.ComposedDistribution())

        corr_dim = dim * (dim - 1) / 2
        obj._corr_dim = corr_dim
        obj._input_dim = dim

        obj._list_param = data_sample[:, :corr_dim]
        obj._n_sample = obj._list_param.shape[0]
        obj._input_sample = data_sample[:, corr_dim:corr_dim + dim]
        obj._all_output_sample = data_sample[:, corr_dim + dim:]
        obj._output_sample = obj._all_output_sample[:, out_ID]
        obj._params = pd.DataFrame(obj._list_param).drop_duplicates().values
        obj._n_param = obj._params.shape[0]
        obj._n_input_sample = obj._n_sample / obj._n_param
        obj._load_data = True

        return obj

    @classmethod
    def from_structured_data(cls, loaded_data="full_structured_data.csv"):
        """
        Load from structured with labels.

        loaded_data: a string, a DataFrame, a list of strings.
        """
        if isinstance(loaded_data, str):
            data = pd.read_csv(loaded_data)
        elif isinstance(loaded_data, pd.DataFrame):
            data = loaded_data
        elif isinstance(loaded_data, list):
            labels = ""
            data = None  # Init the data
            # For each data sample
            for i, load_dat in enumerate(loaded_data):
                # If it's a string
                if isinstance(load_dat, str):
                    # Load it in a DataFrame
                    dat_i = pd.read_csv(load_dat)
                # If it's a DataFrame
                elif isinstance(load_dat, pd.DataFrame):
                    dat_i = load_dat
                else:
                    raise TypeError("Uncorrect type for data")

                if i == 0:  # For the first file
                    data = dat_i  # Data
                    labels = data.columns.values  # Labels
                else:  # For the other files
                    # Check if the data have the sample features
                    assert (data.columns.values == labels).all(), \
                        "Different data files"
                    # Append all
                    data = data.append(dat_i)
        else:
            raise TypeError("Uncorrect type for loaded_data")

        c_names = data.columns.values  # Column names
        # We count the number of correlation parameters
        corr_dim = ["r_" in name for name in c_names].count(True)

        # Compute the problem dimension
        dim = int(np.roots([1, -1, -2 * corr_dim])[0])

        return cls.from_data(data.values, dim)

    def run(self, n_dep_param, n_input_sample, fixed_grid=False,
            dep_measure="PearsonRho", output_ID=0, seed=None):
        """Run the problem. It creates and evaluates the sample from different
        dependence parameter values.

        Parameters
        ----------
        n_dep_param : int
            The number of dependence parameters.

        n_input_sample : int
            The number of observations in the sampling of :math:`\mathbf X`.

        fixed_grid : bool, optional (default=False)
            The sampling of :math:`\mathbf X` is fixed or random.

        dep_measure : string, optional (default="PearsonRho")
            The dependence measure used in the problem to explore the dependence 
            structures. Available dependence measures: 
            - "PearsonRho": The Pearson Rho parameter. Also called linear correlation parameter.
            - "KendallTau": The Tau Kendall parameter.

        output_ID : int, optional (default=0)
            The index of the output if the output is multidimensional.

        seed : int or None, optional (default=None)
            If int, ``seed`` is the seed used by the random number generator;
            If None, ``seed`` is the seed is random.

        Attributes
        ----------
        input_sample_ : :class:`~numpy.ndarray`
            The input sample

        output_sample_ : :class:`~numpy.ndarray`
            The output sample from the model.

        all_params_ : :class:`~numpy.ndarray`
            The dependence parameters associated to each output observation.
        """
        if seed:  # Initialises the seed
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)

        # Creates the sample of dependence parameters
        self._build_corr_sample(n_dep_param, fixed_grid, dep_measure)

        # Creates the sample of input parameters
        self._build_input_sample(n_input_sample)

        # Evaluates the input sample
        self._all_output_sample = self.model_func(self._input_sample)

        # If the output dimension is one
        if self._all_output_sample.shape[0] == self._all_output_sample.size:
            self._output_sample = self._all_output_sample
        else:
            self._output_sample = self._all_output_sample[:, output_ID]

    def _build_corr_sample(self, n_param, fixed_grid, dep_measure):
        """Creates the sample of dependence parameters.

        Parameters
        ----------
        n_param : int
            The number of dependence parameters.

        fixed_grid : bool
            The sampling of :math:`\mathbf X` is fixed or random.

        dep_measure : string, optional (default="PearsonRho")
            The dependence measure used in the problem to explore the dependence 
            structures. Available dependence measures: 
            - "PearsonRho": The Pearson Rho parameter. Also called linear correlation parameter.
            - "KendallTau": The Tau Kendall parameter.
        """
        corr_dim = self._corr_dim

        if dep_measure == "KendallTau":
            if fixed_grid:
                raise NotImplementedError("Not done yet")
                meas_param = get_grid_rho(self._corr_matrix_bool, n_param)                        
            else:  # Random grid
                meas_param = np.zeros((n_param, self._corr_dim))
                for i in self._corr_vars:
                    tau_min, tau_max = get_tau_interval(self._family_list[i])
                    meas_param[:, i] = np.random.uniform(tau_min, tau_max, n_param)

        elif dep_measure == "PearsonRho":
            if fixed_grid:
                assert self._n_corr_vars == 1, \
                    NotImplementedError(
                        "Fixed Grid does not work for high dim")
                # TODO: fix that
                meas_param = get_grid_rho(self._corr_matrix_bool, n_param)
            else:  # Random grid
                meas_param = create_random_correlation_param(
                    self._corr_matrix_bool, n_param)
        elif dep_measure == "SpearmanRho":
            raise NotImplementedError("Not yet implemented")
        else:
            raise AttributeError("Unkown dependence parameter")
            
        self._meas_param = meas_param
        self._n_param = n_param
        self._params = np.zeros((n_param, self._corr_dim))

        for i in self._corr_vars:
            self._params[:, i] = self._copula[i].to_copula_parameter(meas_param[:, i], dep_measure)

    def _build_input_sample(self, n):
        """Creates the observations for differents dependence parameters.

        Parameters
        ----------        
        n : int
            The number of observations in the sampling of :math:`\mathbf X`.

        from_init_sample : bool, optional (default=None)
            Not yet functionable. 
        """
        
        n_sample = n * self._n_param
        self._n_sample = n_sample
        self._n_input_sample = n
        self._input_sample = np.empty((n_sample, self._input_dim))
        self._all_params = np.empty((n_sample, self._corr_dim))

        # We loop for each copula param and create observations for each
        for i, param in enumerate(self._params):  # For each copula parameter
            tmp = self._get_sample(param, n)

            # We save the input sample
            self._input_sample[n*i:n*(i+1), :] = tmp

            # As well for the dependence parameter
            self._all_params[n*i:n*(i+1), :] = param

    def _get_sample(self, param, n_obs):
        """
        must be the copula parameter
        """
        dim = self._input_dim

        # TODO: The structure is standard, think about changing it.
        structure = np.zeros((dim, dim), dtype=int)
        for i in range(dim):
            structure[i, 0:i+1, ] = i + 1

        matrix_param = to_matrix(param, dim)
        # TODO: We only use one param. Do it for two parameters copulas.

        vine_copula = VineCopula(structure, self._families, matrix_param)

        # Sample from the copula
        cop_sample = vine_copula.get_sample(n_obs)

        # Applied to the inverse transformation
        joint_sample = np.zeros((n_obs, dim))
        for i, inv_CDF in enumerate(self._margins_inv_CDF):
            joint_sample[:, i] = np.asarray(inv_CDF(cop_sample[:, i])).ravel()

        return joint_sample

    def buildForest(self, n_jobs=8):
        """Build a Quantile Random Forest to estimate conditional quantiles.
        """
        self._quantForest = QuantileForest(self._all_params,
                                           self._output_sample, n_jobs=n_jobs)

    def compute_quantity(self, quantity_func, options, boostrap=False):
        """Compute the output quantity of interest.
        quantity_func: can be many things
            - a callable function that compute the quantity of interest
            given the output sample,
            - "quantile" to compute the quantiles,
            - "probability" to compute the probability,
            - "variance" to compute the output variance,
            - "mean" to compute the output mean,
            - "super-quantile" to compute the super quantile.
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

    def compute_probability(self, threshold, confidence_level=0.95,
                            estimation_method="empirical", operator="greater"):
        """Compute the probability of the current sample for each dependence
        parameter.
        """
        assert isinstance(threshold, float), \
            TypeError("Threshold should be a float")
        assert isinstance(confidence_level, float), \
            TypeError("Confidence Level should be a float")
        if confidence_level <= 0. or confidence_level >= 1.:
            raise ValueError("Confidence level should be a probability")
        assert isinstance(estimation_method, str), \
            TypeError("Method name should be a string")

        out_sample = self.reshaped_output_sample_
        params = {'Quantity Name': 'Probability',
                  'Threshold': threshold,
                  'Confidence Level': confidence_level,
                  'Estimation Method': estimation_method,
                  'Operator': operator
                  }

        if estimation_method == "empirical":
            # Computes the empirical probability of the sample
            if operator == "greater":
                probability = (
                    (out_sample > threshold).astype(float)).mean(axis=1)
            elif operator == "lower":
                probability = (
                    (out_sample <= threshold).astype(float)).mean(axis=1)

            tmp = np.sqrt(probability * (1. - probability) /
                          self._n_input_sample)
            # Quantile of a Gaussian distribution
            q_normal = np.asarray(ot.Normal().computeQuantile(
                (1 + confidence_level) / 2.))
            interval = q_normal * tmp  # Confidence interval
        else:
            raise AttributeError("Method does not exist")

        return DependenceResult(params, self, probability, interval)

    def compute_quantiles(self, alpha, confidence_level=0.95,
                          estimation_method="empirical"):
        """Computes conditional quantiles.

        Compute the alpha-quantiles of the current sample for each dependence
        parameter.

        Parameters
        ----------
        alpha : float
            Probability of the quantile.
        """
        assert isinstance(estimation_method, str), \
            TypeError("Method name should be a string")
        assert isinstance(alpha, float), \
            TypeError("Method name should be a float")
        if alpha <= 0. or alpha >= 1.:
            raise ValueError("Quantile probability should be a probability")

        params = {'Quantity Name': 'Quantile',
                  'Quantile Probability': alpha,
                  'Confidence Level': confidence_level,
                  'Estimation Method': estimation_method
                  }
        interval = None

        if estimation_method == "empirical":
            if self._load_data:
                out_sample = np.zeros((self._n_param, self._n_input_sample))
                for i, param in enumerate(self._params):
                    id_param = np.where(
                        (self._all_params == param).all(axis=1))[0]
                    out_sample[i, :] = self._output_sample[id_param]
            else:
                out_sample = self.reshaped_output_sample_

            quantiles = np.percentile(out_sample, alpha * 100., axis=1)
        elif estimation_method == "randomforest":
            self.buildForest()
            self._quantiles = self._quantForest.compute_quantile(
                self._params, alpha)
        else:
            raise AttributeError(
                "Unknow estimation method: %s" % estimation_method)

        return DependenceResult(params, self, quantiles, interval)

    def save_input_data(self, path=".", fname="inputSampleCop", ftype=".csv"):
        """
        """
        full_fname = path + '/' + fname + ftype
        np.savetxt(full_fname, self._all_params)

    def save_all_data(self, path=".", fname="full_data", ftype=".csv"):
        """

        """
        full_fname = path + '/' + fname + ftype
        out = np.c_[self._all_params, self._input_sample, self._output_sample]
        np.savetxt(full_fname, out)

    def save_structured_all_data(self, input_names=[], output_names=[],
                                 path=".", fname="full_structured_data",
                                 ftype=".csv"):
        """
        """
        labels = []
        for i in range(self._input_dim):
            for j in range(i + 1, self._input_dim):
                labels.append("r_%d%d" % (i + 1, j + 1))
        if input_names:
            assert len(input_names) == self._input_dim,\
                "Dimension problem for input_names"
            labels.extend(input_names)
        else:
            for i in range(self._input_dim):
                labels.append("x_%d" % (i + 1))

        output_dim = self._all_output_sample.shape[1]
        if output_names:
            assert len(output_names) == output_dim,\
                "Dimension problem for output_names"
            labels.extend(output_names)
        else:
            for i in range(output_dim):
                labels.append("y_%d" % (i + 1))

        full_fname = path + '/' + fname + ftype
        out = np.c_[self._all_params, self._input_sample,
                    self._all_output_sample]
        out_df = pd.DataFrame(out, columns=labels)
        out_df.to_csv(full_fname, index=False)

    def get_corresponding_sample(self, corr_value):
        """
        """
        id_corr = np.where((self._all_params == corr_value).all(axis=1))[0]
        x = self._input_sample[id_corr]
        y = self._output_sample[id_corr]
        return x, y

    def draw_design_space(self, corr_id=None, figsize=(10, 6),
                          savefig=False, color_map="jet", output_name=None,
                          input_names=None, return_fig=False, color_lims=None,
                          display_quantile_value=None):
        """
        """
        assert self._input_dim in [2, 3], "Cannot draw quantiles for dim > 3"

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if corr_id is None:
            id_corr = np.ones(self._n_sample, dtype=bool)
        else:
            id_corr = np.where((self._all_params == self._params[corr_id]).all(axis=1))[0]

        if input_names:
            param_name = input_names
        else:
            param_name = ["$x_{%d}$" % (i + 1) for i in range(self._input_dim)]

        if output_name:
            output_label = output_name
        else:
            output_label = "Output value"

        x = self._input_sample[id_corr]
        y = self._output_sample[id_corr]
        color_scale = y
        cm = plt.get_cmap(color_map)
        if color_lims is None:
            c_min, c_max = min(color_scale), max(color_scale)
        else:
            c_min, c_max = color_lims[0], color_lims[1]
        cNorm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        if display_quantile_value:
            alpha = display_quantile_value
            self.compute_quantiles(alpha, 1)
            if self._input_dim == 2:
                id_quant = np.where(self._params == corr_value)[0]
            else:
                id_quant = np.where(
                    (self._params == corr_value).all(axis=1))[0]
            quant = self._quantiles[id_quant]

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
            if display_quantile_value:
                title += "\n$q_\\alpha = %.1f$ - $\\rho = " % (quant)
            else:
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
            if type(savefig) is str:
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
    def set_correlated_variables(self, list_vars=None, matrix_bool=None):
        """
        Set of the variable distribution. They must be OpenTURNS distribution
        with a defined copula in it.
        """
        if list_vars:
            assert isinstance(list_vars, (list, np.ndarray)), \
                TypeError('Unsupported type')
            # Matrix of correlated variables
            matrix_bool = np.identity(self._input_dim, dtype=bool)

            # For each couple of correlated variables
            for corr_i in list_vars:
                assert len(corr_i) == 2, \
                    ValueError("Wrong number: %d obtained" % len(corr_i))

                # Make it true in the matrix
                matrix_bool[corr_i[0], corr_i[1]] = True
                matrix_bool[corr_i[1], corr_i[0]] = True

            k = 0
            corr_vars = []
            for i in range(self._input_dim):
                for j in range(i + 1, self._input_dim):
                    if matrix_bool[i, j]:
                        corr_vars.append(k)
                    k += 1

            self._corr_matrix_bool = matrix_bool
            self._corr_vars = corr_vars
            self._corr_vars_ids = list_vars
            self._n_corr_vars = len(corr_vars)
        elif matrix_bool:
            assert isinstance(matrix_bool, np.ndarray), \
                TypeError('Unsupported type')
            k = 0
            corr_vars = []
            list_vars = []
            for i in range(self._input_dim):
                for j in range(i + 1, self._input_dim):
                    if matrix_bool[i, j]:
                        corr_vars.append(k)
                        list_vars.append([i, j])
                    k += 1

            self._corr_matrix_bool = matrix_bool
            self._corr_vars = corr_vars
            self._corr_vars_ids = list_vars
            self._n_corr_vars = len(corr_vars)
        else:  # All random variables are correlated
            self._corr_matrix_bool = np.ones(
                (self._input_dim, self._input_dim), dtype=bool)
            self._corr_vars = range(self._corr_dim)
            self._corr_vars_ids = list(combinations(range(self._input_dim), 2))
            self._n_corr_vars = self._corr_dim

    @property
    def model_func(self):
        """The model function. Must be a callable.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
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
            elif isinstance(marginal, rv_continuous):
                self._margins_inv_CDF.append(marginal.ppf)
            else:
                TypeError("Must be scipy or OpenTURNS distribution objects.")

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
        check_matrix(value)
        self._families = value
        self._family_list = []
        self._n_corr_vars = 0
        self._corr_vars = []
        k = 0
        list_vars = []
        for i in range(self._input_dim):
            for j in range(i):
                self._family_list.append(value[i, j])
                if value[i, j] > 0:
                    self._corr_vars.append(k)
                    self._n_corr_vars += 1
                    list_vars.append([i, j])
                k += 1

        self._copula = [Conversion(family) for family in self._family_list]
        self._corr_vars_ids = list_vars

    @property
    def rand_vars(self):
        """The random variable describing the input joint density of the problem.
        """
        return self._rand_vars

    @rand_vars.setter
    def rand_vars(self, rand_vars):
        assert isinstance(rand_vars, ot.ComposedDistribution), \
            TypeError("The variables must be OpenTURNS Distributions.")

        self._rand_vars = rand_vars

    @property
    def output_sample_(self):
        return self._output_sample

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
        return self._all_params

    @all_params_.setter
    def all_params_(self, value):
        raise EnvironmentError("You cannot set this variable")

    @property
    def reshaped_output_sample_(self):
        return self._output_sample.reshape((self._n_param, self._n_input_sample))

    @reshaped_output_sample_.setter
    def reshaped_output_sample_(self, value):
        raise EnvironmentError("You cannot set this variable")


class DependenceResult(object):

    def __init__(self, params, dependence_object, quantity,
                 confidence_interval=None):
        self.params = params
        self.quantity = quantity
        self.confidence_interval = confidence_interval
        self.dependence = dependence_object

        assert isinstance(dependence_object, ImpactOfDependence), \
            'Variable must be an ImpactOfDependence object'

    @property
    def dependence(self):
        return self._dependence

    @dependence.setter
    def dependence(self, obj):
        assert isinstance(obj, ImpactOfDependence), \
            'Variable must be an ImpactOfDependence object'
        self._dependence = obj

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
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

        self._params = value

    def __str__(self):
        to_print = '%s: %s\n' % (self._quantity_name, self.quantity)
        if self.confidence_interval is not None:
            to_print += '%s confidence interval at %d %%: %s' % \
                (self._quantity_name, self._confidence_level *
                 100, self.confidence_interval)
        return to_print

    def draw(self, dep_meas="CopulaParam", figsize=(10, 6),
             savefig=False, color_map="jet", max_eps=1.E-1):
        """
        The quantity must be compute before
        """
        obj = self._dependence
        n_corr_vars = obj._n_corr_vars

        assert n_corr_vars in [1, 2, 3],\
            EnvironmentError("Cannot draw quantiles for dim > 3")
            
        # Dependence parameters
        copula_params = obj._params[:, obj._corr_vars]

        if dep_meas == "KendallTau":
            params = np.zeros((obj._n_param, n_corr_vars))
            for i, k in enumerate(obj._corr_vars):
                params[:, i] = obj._copula[k].to_Kendall(copula_params[:, i])
            param_name = "\\tau"
        elif dep_meas == "PearsonRho":
            params = obj._copula_converter.to_Pearson(copula_params)
            param_name = "\\rho^{Pearson}"
        elif dep_meas == "CopulaParam":
            params = copula_params
            param_name = "\\rho"
        else:
            raise("Undefined param")


        # Output quantities of interest
        quantity = self.quantity
        interval = self.confidence_interval
        quantity_name = self._quantity_name

        # Find the "almost" independent configuration
        if n_corr_vars == 1:
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

        if n_corr_vars == 1:  # One used correlation parameter
            ax = fig.add_subplot(111)  # Create the axis object

            # Ids of the sorted parameters for the plot
            id_sorted_params = np.argsort(params, axis=0).ravel()

            # Plot of the quantile conditionally to the correlation parameter
            ax.plot(params[id_sorted_params], quantity[id_sorted_params],
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

            i, j = obj._corr_vars_ids[0][0], obj._corr_vars_ids[0][1]
            ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
            ax.set_ylabel(quantity_name)
            ax.legend(loc="best")

        elif n_corr_vars == 2:  # For 2 correlation parameters
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
                # independence
                # case
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
                i, j = obj._corr_vars_ids[0][0], obj._corr_vars_ids[0][1]
                ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
                i, j = obj._corr_vars_ids[1][0], obj._corr_vars_ids[1][1]
                ax.set_ylabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)

                ax.set_zlabel(quantity_name)

        elif n_corr_vars == 3:  # For 2 correlation parameters
            color_scale = quantity
            cm = plt.get_cmap(color_map)
            c_min, c_max = min(color_scale), max(color_scale)
            cNorm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            x, y, z = params[:, 0], params[:, 1], params[:, 2]

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=scalarMap.to_rgba(color_scale), s=40)
            scalarMap.set_array(color_scale)
            cbar = fig.colorbar(scalarMap)
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
        plt.show(block=False)

        # Saving the figure
        if savefig:
            fname = './'
            if type(savefig) is str:
                fname += savefig
            else:
                fname += "fig" + quantity_name
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")


def bootstrap(data, num_samples, statistic, alpha, args):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1, *args))

    return (stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)])

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