"""
Using a Vine Copula construction, we aim to determine the bounds of
a quantity of interest.
"""

import numpy as np
import openturns as ot

class DependenceBounding(object):    
    def __init__(self, model_func, margins, vine_structure=None):
        self.model_func = model_func
        self.margins = margins
        self.vine_structure = vine_structure

        self._n_configs = 2**self._input_dim

    def run(self, n_input_sample, output_ID=0, seed=None):
        """
        """
        if seed:  # Initialises the seed
            np.random.seed(seed)
            ot.RandomGenerator.SetSeed(seed)


        # Creates the sample of input parameters
        self._build_input_sample(n_input_sample)

        # Evaluates the input sample
        self._all_output_sample = self.model_func(self._input_sample)

        # If the output dimension is one
        if self._all_output_sample.shape[0] == self._all_output_sample.size:
            self._output_dim = 1
            self._output_sample = self._all_output_sample
        else:
            self._output_dim = self._all_output_sample.shape[1]
            self._output_sample = self._all_output_sample[:, output_ID]

    def _build_input_sample(self, n):
        """Creates the observations for differents dependence parameters.

        Parameters
        ----------
        n : int
            The number of observations in the sampling of :math:`\mathbf X`.
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

    def _get_sample(self, param, n_obs, param2=None):
        """Creates the sample from the Vine Copula.

        Parameters
        ----------
        param : :class:`~numpy.ndarray`
            The copula parameters.
        n_obs : int
            The number of observations.
        param2 : :class:`~numpy.ndarray`, optional (default=None)
            The 2nd copula parameters. Usefull for certain copula families like Student.
        """
        dim = self._input_dim
        structure = self._vine_structure

        matrix_param = to_matrix(param, dim)

        if self._copula_type == 'vine':
            # TODO: One param is used. Do it for two parameters copulas.
            vine_copula = VineCopula(structure, self._families, matrix_param)

            # Sample from the copula
            # The reshape is in case there is only one sample (for RF tests)
            cop_sample = vine_copula.get_sample(n_obs).reshape(n_obs, self._input_dim)
        elif self._copula_type == 'normal':
            # Create the correlation matrix
            cor_matrix = matrix_param + matrix_param.T + np.identity(dim)
            cop = ot.NormalCopula(ot.CorrelationMatrix(cor_matrix))
            cop_sample = np.asarray(cop.getSample(n_obs))
        else:
            raise AttributeError('Unknown type of copula.')

        # Applied to the inverse transformation to get the sample of the joint distribution
        joint_sample = np.zeros((n_obs, dim))
        for i, inv_CDF in enumerate(self._margins_inv_CDF):
            joint_sample[:, i] = np.asarray(inv_CDF(cop_sample[:, i])).ravel()

        return joint_sample
        
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
 