# -*- coding: utf-8 -*-
import numpy as np
import openturns as ot
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from conversion import Conversion
from correlation import get_grid_rho, create_random_correlation_param
from pyquantregForest import QuantileForest

COPULA_LIST = ["Normal", "Clayton", "Gumbel"]

def bootstrap(data, num_samples, statistic, alpha, args):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = np.random.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1, *args))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])
            
class ImpactOfDependence(object):
    """
    """
    _load_data = False

    def __init__(self, model_function=None, variable=None, correlation=None):
        """
        """
        if model_function:
            self.set_model_function(model_function)
        if variable:
            self.set_input_variables(variable, correlation)

        # Initialize the variables
        self._all_output_sample = None
        self._output_sample = None
        self._output_quantity = None
        self._output_quantity_interval = None

    @classmethod
    def from_data(cls, data_sample, dim, out_ID=0):
        """
        Load from raw data.
        """
        obj = cls()
        corr_dim = dim * (dim - 1) / 2
        obj._input_dim = dim
        obj._corr_dim = corr_dim
        obj._list_param = data_sample[:, :corr_dim]
        obj._n_sample = obj._list_param.shape[0]
        obj._input_sample = data_sample[:, corr_dim:corr_dim + dim]
        obj._all_output_sample = data_sample[:, corr_dim + dim:]
        obj._output_sample = obj._all_output_sample[:, out_ID]
        obj._params = pd.DataFrame(obj._list_param).drop_duplicates().values
        obj._n_param = obj._params.shape[0]
        obj._n_obs_sample = obj._n_sample / obj._n_param
        obj._load_data = True
        return obj

    @classmethod
    def from_structured_data(cls, loaded_data="full_structured_data.csv"):
        """
        Load from structured with labels.

        loaded_data: a string, a DataFrame, a list of strings.
        """
        # If its a string
        if isinstance(loaded_data, str):
            data = pd.read_csv(loaded_data)
        # If it's a DataFrame
        elif isinstance(loaded_data, pd.DataFrame):
            data = loaded_data
        # If it's a list of data to load
        elif isinstance(loaded_data, list):
            labels = ""  # Init labels
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

    def run(self, n_sample, fixed_grid=True, dep_meas="PearsonRho",
            n_obs_sample=1, output_ID=0, seed=None, from_init_sample=False):
        """
            Run the problem. It creates the sample and evaluate it.
        """
        # Set the seed for numpy and openturns
        if seed:
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)

        # Creation of the sample for all the correlation parameters
        self._create_sample(n_sample, fixed_grid, dep_meas, n_obs_sample,
                            from_init_sample)

        # Evaluation of the input sample
        self._all_output_sample = self._modelFunction(self._input_sample)

        # If the output dimension is one
        if self._all_output_sample.shape[0] == self._all_output_sample.size:
            self._output_sample = self._all_output_sample
        else:
            self._output_sample = self._all_output_sample[:, output_ID]

    def get_reshaped_output_sample(self):
        """
        """
        # Load the output sample and reshape it in a matrix
        return self._output_sample.reshape((self._n_param, self._n_obs_sample))

    def _to_copula_params(measure_param, dep_measure):
        """
        """
        if dep_measure == "KendallTau":
            if self._copula_name == "NormalCopula":
                copula_param = Conversion.\
                    NormalCopula.fromKendallToPearson(measure_param)

            elif self._copula_name == "InverseClaytonCopula":
                copula_param = Conversion.\
                    ClaytonCopula.fromKendallToPearson(measure_param)

        return copula_param
    def _create_sample(self, n_sample, fixed_grid, dep_measure, n_obs_sample,
                       from_init_sample):
        """
            Create the observations for differents dependence parameters
        """
        dim = self._input_dim  # Dimension of input variables
        corr_dim = self._corr_dim  # Dimension of correlation parameters

        # We need the same number of observation for each sample
        n_sample -= n_sample % n_obs_sample  # We take off the rest
        n_param = n_sample / n_obs_sample  # Number of correlation parameters

        # We convert the dependence measure to the copula parameter
        if dep_measure == "KendallTau":
            if fixed_grid:  # Fixed grid
                grid_tau = get_grid_tau(n_param, corr_dim, tauMin, tauMax)
                list_tau = np.vstack(grid_tau).reshape(corr_dim, -1).T
            else:  # Random grid
                tmp = [ot.Uniform(tauMin, tauMax)] * corr_dim
                tmpVar = ot.ComposedDistribution(tmp)
                list_tau = np.array(tmpVar.getSample(n_param)).ravel()

            list_param = self._to_copula_params(list_tau, dep_measure)


        elif dep_measure == "PearsonRho":

            if fixed_grid:  # Fixed grid
                assert self._n_corr_vars == 1,  "Fixed Grid does not work for high dim"
                # TODO: fix that shit!
                list_rho = get_grid_rho(self._corr_matrix_bool, n_param)
                # Once again, we change the number of param to have a grid
                # with the same number of parameters in each dim
                n_param = list_rho.shape[0]
                # TODO : The total number of param may certainly be different
                # than the initial one.  Find a way to adapt it...
                n_sample = n_param * n_obs_sample
            else:  # Random grid
                list_rho = create_random_correlation_param(
                    self._corr_matrix_bool, n_param)

            if self._copula_name == "NormalCopula":
                params = list_rho

            elif self._copula_name == "InverseClaytonCopula":
                # TODO : check that shit
                params = Conversion.\
                    ClaytonCopula.fromPearsonToKendall(list_rho)
        else:
            raise Exception("Other dependence measures not yet implemented")

        self._n_param = n_param
        self._n_sample = n_sample
        self._n_obs_sample = n_obs_sample
        self._params = params
        self._input_sample = np.empty((n_sample, dim))  # Input sample
        self._list_param = np.empty((n_sample, corr_dim))  # Input Corr sample

        if from_init_sample:
            #            var_unif = ot.ComposedDistribution([ot.Uniform(0, 1)]*dim)
            #            unif_sample = np.asarray(var_unif.getSample(n_obs_sample))
            #            for i, var in enumerate(self._input_variables):
            #                var.compute
            init_sample = self._input_variables.getSample(n_obs_sample)
            self._sample_init = init_sample

        # We loop for each copula param and create observations for each
        for i, param in enumerate(params):  # For each copula parameter
            # TODO : do it better
            if from_init_sample:
                if self._corr_dim == 1:
                    self._corr_matrix[0, 1] = param
                elif self._corr_dim == 3:
                    self._corr_matrix[0, 1] = param[0]
                    self._corr_matrix[0, 2] = param[1]
                    self._corr_matrix[1, 2] = param[2]
                B = self._corr_matrix.computeCholesky()
                tmp = np.asarray(init_sample * B)
            else:
                tmp = self._get_sample(param, n_obs_sample)

            # TODO : find a way to create a "real" class InverseClayton...
            if self._copula_name == "InverseClaytonCopula":
                tmp[:, 0] = -tmp[:, 0]

            # We save the input sample
            self._input_sample[n_obs_sample *
                               i:n_obs_sample * (i + 1), :] = tmp

            # As well for the parameters (for dependence measure)
            self._list_param[n_obs_sample * i:n_obs_sample * (i + 1), :] = \
                params[i]

    def buildForest(self, n_jobs=8):
        """
        Build a Quantile Random Forest to estimate conditional quantiles.
        """
        self._quantForest = QuantileForest(self._list_param,
                                           self._output_sample, n_jobs=n_jobs)

    def compute_quantity(self, quantity_func, options, boostrap=False):
        """
        Compute the output quantity of interest.
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
                self._output_quantity = self._quantiles
                if boostrap:
                    raise Exception("Not implemented yet")
                else:
                    self._output_quantity_interval = None
                    self._output_quantity_up_bound = None
                    self._output_quantity_low_bound = None
            if quantity_func == "probability":
                self.compute_probability(*options)
                self._output_quantity = self._probability
                self._output_quantity_interval = self._probability_interval
                self._output_quantity_up_bound = self._probability + self._probability_interval
                self._output_quantity_low_bound = self._probability - \
                    self._probability_interval
        elif callable(quantity_func):
            out_sample = self._output_sample.reshape((self._n_param,
                                                      self._n_obs_sample))
            result = quantity_func(out_sample, *options)
            if isinstance(result, tuple):
                self._output_quantity = result[0]
                self._output_quantity_interval = result[1]
                self._output_quantity_up_bound = result[0] + result[1] / 2.
                self._output_quantity_low_bound = result[0] - result[1] / 2.
            else:
                self._output_quantity = result[0]
                self._output_quantity_interval = None
                self._output_quantity_up_bound = None
                self._output_quantity_low_bound = None
        else:
            raise TypeError("Unknow input variable quantity_func")

    def compute_probability(self, threshold, confidence_level=0.95,
                            operator="greater"):
        """
        Compute the probability of the current sample for each dependence
        parameter.
        """
        # Load the output sample and reshape it in a matrix
        out_sample = self._output_sample.reshape((self._n_param,
                                                  self._n_obs_sample))

        # Compute the empirical probability of the sample
        if operator == "greater":
            probability = ((out_sample >= threshold) *
                           1.).sum(axis=1) / self._n_obs_sample
        elif operator == "lower":
            probability = ((out_sample < threshold) *
                           1.).sum(axis=1) / self._n_obs_sample

        # Confidence interval by TCL theorem.
        tmp = np.sqrt(probability * (1. - probability) / self._n_obs_sample)
        # Quantile of a Gaussian distribution
        q_normal = np.asarray(
            ot.Normal().computeQuantile( (1 + confidence_level) / 2.))

        # Half interval
        interval = q_normal * tmp

        self._probability = probability
        self._probability_interval = interval

    def compute_quantiles(self, alpha, estimation_method):
        """
        Compute the alpha-quantiles of the current sample for each dependence
        parameter.
        """
        # Empirical quantile
        if estimation_method == 1:
            # Loaded data
            if self._load_data:
                out_sample = np.zeros((self._n_param, self._n_obs_sample))
                for i, param in enumerate(self._params):
                    id_param = np.where(
                        (self._list_param == param).all(axis=1))[0]
                    out_sample[i, :] = self._output_sample[id_param]
            else:
                out_sample = self._output_sample.reshape((self._n_param,
                                                          self._n_obs_sample))
            self._quantiles = np.percentile(out_sample, alpha * 100., axis=1)
        # Quantile Random Forest
        elif estimation_method == 2:
            self.buildForest()
            self._quantiles = self._quantForest.compute_quantile(
                self._params, alpha)
        else:
            raise Exception("Not done yet")

    def save_input_data(self, path=".", fname="inputSampleCop", ftype=".csv"):
        """
        """
        full_fname = path + '/' + fname + ftype
        np.savetxt(full_fname, self._list_param)

    def save_all_data(self, path=".", fname="full_data", ftype=".csv"):
        """

        """
        full_fname = path + '/' + fname + ftype
        out = np.c_[self._list_param, self._input_sample, self._output_sample]
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
        out = np.c_[self._list_param, self._input_sample,
                    self._all_output_sample]
        out_df = pd.DataFrame(out, columns=labels)
        out_df.to_csv(full_fname, index=False)

    def _get_sample(self, param, n_obs):
        """
        must be the copula parameter
        """
        # TODO: change it when the new ot released is out
        if self._copula_name == "NormalCopula":
            k = 0
            for i in range(self._input_dim):
                for j in range(i + 1, self._input_dim):
                    self._corr_matrix[i, j] = param[k]
                    k += 1
            self._copula = ot.NormalCopula(self._corr_matrix)
        else:
            self._copula.setParametersCollection([param])

        self._input_variables.setCopula(self._copula)
        return np.asarray(self._input_variables.getSample(n_obs))

    def get_corresponding_sample(self, corr_value):
        """
        """
        id_corr = np.where((self._list_param == corr_value).all(axis=1))[0]
        x = self._input_sample[id_corr]
        y = self._output_sample[id_corr]
        return x, y

    def draw_design_space(self, corr_value=None, figsize=(10, 6),
                          savefig=False, color_map="jet", output_name=None,
                          input_names=None, return_fig=False, color_lims=None,
                          display_quantile_value=None):
        """
        """
        assert self._input_dim in [2, 3], "Cannot draw quantiles for dim > 3"

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if corr_value is None:
            id_corr = np.ones(self._n_sample, dtype=bool)

        else:
            id_corr = np.where((self._list_param == corr_value).all(axis=1))[0]

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
        if corr_value is not None:
            if display_quantile_value:
                title += "\n$q_\\alpha = %.1f$ - $\\rho = " % (quant)
            else:
                title += "\n$\\rho = "

            if self._corr_dim == 1:
                title += "%.1f$" % (corr_value)
            elif self._corr_dim == 3:
                title += "[%.1f, %.1f, %.1f]$" % (corr_value[0], corr_value[1],
                                                  corr_value[2])
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

    def draw_quantity(self, quantity_name="Quantity",
                      dep_meas="PearsonRho", figsize=(10, 6),
                      savefig=False, color_map="jet"):
        """
        The quantity must be compute before
        """
        assert self._n_corr_vars in [1, 2, 3],\
            EnvironmentError("Cannot draw quantiles for dim > 3")
        assert self._output_quantity is not None,\
            Exception("Quantity must be computed first")

        if dep_meas == "KendallTau":
            param_name = "\\tau"

        elif dep_meas == "PearsonRho":
            param_name = "\\rho"
        else:
            raise("Undefined param")

        # Dependence parameters
        params = self._params[:, self._corr_vars]
        # Output quantities of interest
        quantity = self._output_quantity
        # Output quantities of interest confidence intervals
        low_bound = self._output_quantity_low_bound
        up_bound = self._output_quantity_up_bound

        # Find the almost independent configuration
        max_eps = 1.E-1  # Tolerence for the independence param
        if self._n_corr_vars == 1:
            id_indep = (np.abs(params)).argmin()
        else:
            id_indep = np.abs(params).sum(axis=1).argmin()

        # Independent parameter and quantile
        indep_param = params[id_indep]
        indep_quant = quantity[id_indep]
        if low_bound is not None:
            indep_quant_l_bound = low_bound[id_indep]
            indep_quant_u_bound = up_bound[id_indep]

        # If it's greater than the tolerence, no need to show it
        if np.sum(indep_param) > max_eps:
            print_indep = False
        else:
            print_indep = True

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if self._n_corr_vars == 1:  # One used correlation parameter
            ax = fig.add_subplot(111)  # Create the axis object

            # Ids of the sorted parameters for the plot
            id_sorted_params = np.argsort(params, axis=0).ravel()

            # Plot of the quantile conditionally to the correlation parameter
            ax.plot(params[id_sorted_params], quantity[id_sorted_params],
                    'ob', label=quantity_name, linewidth=2)

            # Plot the confidence bounds
            if low_bound is not None:
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
                if low_bound is not None:
                    ax.plot([p_min, p_max], [indep_quant_l_bound] * 2, "r--")
                    ax.plot([p_min, p_max], [indep_quant_u_bound] * 2, "r--")

            i, j = self._corr_vars_ids[0][0], self._corr_vars_ids[0][1]
            ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
            ax.set_ylabel(quantity_name)
            ax.legend(loc="best")

        elif self._n_corr_vars == 2:  # For 2 correlation parameters
            view = "3d"
            if view == "3d":
                # Dependence parameters values
                r1, r2 = params[:, 0], params[:, 1]

                # 3d ax
                ax = fig.add_subplot(111, projection='3d')
                # Draw the point with the colors
                ax.scatter(r1, r2, quantity, s=40)

                # Plot the confidence bounds
                if low_bound is not None:
                    ax.plot_trisurf(r1, r2, low_bound,
                                    color="red", alpha=0.05, linewidth=1)
                    ax.plot_trisurf(r1, r2, up_bound, color="red",
                                    alpha=0.05, linewidth=1)
                    #ax.plot(r1, r2, up_bound, 'r.')
                    #ax.plot(r1, r2, low_bound, 'r.')

                # Print a line to distinguish the difference with the independence
                # case
                if print_indep:
                    p1_min, p1_max = r1.min(), r1.max()
                    p2_min, p2_max = r2.min(), r2.max()
                    p1_ = np.linspace(p1_min, p1_max, 3)
                    p2_ = np.linspace(p2_min, p2_max, 3)
                    p1, p2 = np.meshgrid(p1_, p2_)
                    q = np.zeros(p1.shape) + indep_quant
                    ax.plot_wireframe(p1, p2, q, color="red")

                    if low_bound is not None:
                        q_l = np.zeros(p1.shape) + indep_quant_l_bound
                        q_u = np.zeros(p1.shape) + indep_quant_u_bound
                        ax.plot_wireframe(p1, p2, q_l, color="red")
                        ax.plot_wireframe(p1, p2, q_u, color="red")
                    #    ax.plot([p_min, p_max], [indep_quant_l_bound] * 2, "r--")
                    #    ax.plot([p_min, p_max], [indep_quant_u_bound] * 2, "r--")
                # Labels
                i, j = self._corr_vars_ids[0][0], self._corr_vars_ids[0][1]
                ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
                i, j = self._corr_vars_ids[1][0], self._corr_vars_ids[1][1]
                ax.set_ylabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)

                ax.set_zlabel(quantity_name)

        elif self._n_corr_vars == 3:  # For 2 correlation parameters
            # Colormap configuration
            color_scale = quantity
            cm = plt.get_cmap(color_map)
            c_min, c_max = min(color_scale), max(color_scale)
            c_norm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
            scalarMap = cmx.ScalarMappable(norm=c_norm, cmap=cm)

            # Dependence parameters values
            r1, r2, r3 = params[:, 0], params[:, 1], params[:, 2]

            # 3d ax
            ax = fig.add_subplot(111, projection='3d')

            # Draw the point with the colors
            ax.scatter(r1, r2, r3, c=scalarMap.to_rgba(color_scale), s=40)

            # Create colorbar
            scalarMap.set_array(color_scale)
            cbar = fig.colorbar(scalarMap)

            # If we print the independence values value on the colorbar
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

            # Labels
            i, j = self._corr_vars_ids[0][0], self._corr_vars_ids[0][1]
            ax.set_xlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
            i, j = self._corr_vars_ids[1][0], self._corr_vars_ids[1][1]
            ax.set_ylabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)
            i, j = self._corr_vars_ids[2][0], self._corr_vars_ids[2][1]
            ax.set_zlabel("$%s_{%d%d}$" % (param_name, i, j), fontsize=14)

        # Other figure stuffs
        title = r"%s - $n = %d$" % (quantity_name, self._n_obs_sample)
        ax.set_title(title, fontsize=18)
        ax.axis("tight")
        fig.tight_layout()
        plt.show(block=False)

        # Saving the figure
        if savefig:
            if type(savefig) is str:
                fname = savefig + '/'
            else:
                fname = "./"
            fname += "fig" + quantity_name
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")

    def draw_quantiles(self, alpha, estimation_method, n_per_dim=10,
                       dep_meas="KendallTau", with_sample=False,
                       figsize=(10, 6), savefig=False, color_map="jet"):
        """

        """
        assert self._n_corr_vars in [1, 3], "Cannot draw quantiles for dim > 3"

        self.compute_quantiles(alpha, estimation_method)

        if dep_meas == "KendallTau":
            param_name = "\\tau"
            param_min, param_max = self._tauMin, self._tauMax
            grid_func = get_grid_tau

        elif dep_meas == "PearsonRho":
            param_name = "\\rho"
            param_min, param_max = self._rhoMin, self._rhoMax
            grid_func = get_grid_rho
        else:
            raise("Undefined param")

        # If t's empirical
        if estimation_method == 1:
            # We take only the used correlated variables
            listParam = self._params[:, self._corr_vars]
            # And all the associated quantiles
            quantiles = self._quantiles

        # Quantile Random forest
        elif estimation_method == 2:
            # We recreate a grid for the regression
            listParam = grid_func(n_per_dim, self._corr_dim, param_min,
                                  param_max, all_sample=False)
            # And we compute the quantiles for each point of the grid
            quantiles = self._quantForest.computeQuantile(listParam, alpha)

        # Find the almost independent configuration
        max_eps = 1.E-1  # Tolerence for the independence param
        if self._n_corr_vars == 1:
            id_indep = (np.abs(listParam)).argmin()
        else:
            id_indep = np.abs(listParam).sum(axis=1).argmin()

        # Independent parameter and quantile
        indep_param = listParam[id_indep]
        indep_quant = quantiles[id_indep]

        # If it's greater than the tolerence, no need to show it
        if np.sum(indep_param) > max_eps:
            print_indep = False
        else:
            print_indep = True

        fig = plt.figure(figsize=figsize)  # Create the fig object

        if self._n_corr_vars == 1:  # One used correlation parameter
            ax = fig.add_subplot(111)  # Create the axis object

            # Ids of the sorted parameters for the plot
            id_sorted_params = np.argsort(listParam, axis=0).ravel()

            # Plot of the quantile conditionally to the correlation parameter
            ax.plot(listParam[id_sorted_params], quantiles[id_sorted_params],
                    'b', label="Conditional %.2f %% quantiles" % (alpha),
                    linewidth=2)

            # Add the sample (warning: it can be costly if too many points)
            if with_sample:
                ax.plot(self._list_param, self._output_sample, 'k.')

            # Print a line to distinguish the difference with the independence
            # case
            if print_indep:
                ax.plot([listParam.min(), listParam.max()], [indep_quant] * 2,
                        "r-")

            ax.set_xlabel("$%s_{12}$" % (param_name), fontsize=14)
            ax.set_ylabel("Quantile")
            ax.legend(loc="best")
        elif self._n_corr_vars == 2:  # For 2 correlation parameters
            raise Exception("Not yet implemented")

        elif self._n_corr_vars == 3:  # For 2 correlation parameters
            color_scale = quantiles
            cm = plt.get_cmap(color_map)
            c_min, c_max = min(color_scale), max(color_scale)
            cNorm = matplotlib.colors.Normalize(vmin=c_min, vmax=c_max)
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            x, y, z = listParam[:, 0], listParam[:, 1], listParam[:, 2]

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
        title = r"$\hat q_{\alpha, n}(%s)$ with $\alpha=%.1e$, $n = %d$" % \
            (param_name, alpha, self._n_sample)
        ax.set_title(title, fontsize=18)
        ax.axis("tight")
        fig.tight_layout()
        plt.show(block=False)

        # Saving the figure
        if savefig:
            if type(savefig) is str:
                fname = savefig + '/'
            else:
                fname = "./"
            fname += "condQuantiles"
            fig.savefig(fname + ".pdf")
            fig.savefig(fname + ".png")

# =============================================================================
# Setters
# =============================================================================
    def set_model_function(self, modelFunction):
        """

        """
        assert callable(modelFunction),\
            TypeError("The model function is not callable")
        self._modelFunction = modelFunction

    def set_input_variables(self, variables, correlation=None):
        """
        Set of the variable distribution. They must be OpenTURNS distribution
        with a defined copula in it.
        """
        assert isinstance(variables, (ot.Distribution,
                                      ot.ComposedDistribution)),\
            TypeError("The variables must be openturns Distribution objects")
        self._input_dim = variables.getDimension()
        self._copula = variables.getCopula()
        self._copula_name = self._copula.getName()
        self._input_variables = variables
        self._corr_dim = self._input_dim * (self._input_dim - 1) / 2

        # All variables are correlated
        if not correlation:
            self._n_corr_vars = self._corr_dim
        # Some variables are correlated
        else:
            if isinstance(correlation, np.ndarray):
                corr_bool = correlation
                k = 0
                corr_vars = []
                corr_vars_id = []
                for i in range(self._input_dim):
                    for j in range(i + 1, self._input_dim):
                        if corr_bool[i, j]:
                            corr_vars.append(k)
                            corr_vars_id.append([i, j])
                        k += 1
            elif isinstance(correlation, list):
                n_corr = len(correlation)  # Number of correlated variables
                corr_vars_id = correlation
                # Matrix of correlated variables
                corr_bool = np.identity(self._input_dim, dtype=bool)
                # For each couple of correlated variables
                for corr_i in correlation:
                    # Verify if it's a couple
                    assert len(corr_i) == 2,\
                        ValueError("Correlation is between 2 variables...")
                    # Make it true in the matrix
                    corr_bool[corr_i[0], corr_i[1]] = True
                    corr_bool[corr_i[1], corr_i[0]] = True

                k = 0
                corr_vars = []
                for i in range(self._input_dim):
                    for j in range(i + 1, self._input_dim):
                        if corr_bool[i, j]:
                            n_corr += 1
                            corr_vars.append(k)
                        k += 1
            self._corr_matrix_bool = corr_bool
            self._corr_vars = corr_vars
            self._corr_vars_ids = corr_vars_id
            self._n_corr_vars = len(corr_vars)

        if self._copula_name == "NormalCopula":
            self._corr_matrix = ot.CorrelationMatrix(self._input_dim)