[![Build Status](https://travis-ci.org/NazBen/impact-of-dependence.svg?branch=master)](https://travis-ci.org/NazBen/impact-of-dependence)
[![CircleCI](https://circleci.com/gh/NazBen/impact-of-dependence.svg?style=svg)](https://circleci.com/gh/NazBen/impact-of-dependence)
[![codecov](https://codecov.io/gh/NazBen/impact-of-dependence/branch/master/graph/badge.svg)](https://codecov.io/gh/NazBen/impact-of-dependence)
# Impact of Dependencies

A python package to study the influence of dependencies between random variables in a probabilistic study. 

For the moment, the class `ConservativeEstimate` creates a probabilistic model with an incomplete description of the dependencies between the input variables. The class can therefore estimates, through a Monte-Carlo sampling, a quantity of interest of a model output distribution. It can also give a conservative estimation of the output quantity by determining a dependence structure that minimize the quantity.

An iterative algorithm is also available.

## Installation

The package is still in development and is not yet availaible on [PyPi](https://pypi.python.org/pypi) or [Anaconda](https://anaconda.org/).

Unfortunatly, the package needs many python library dependencies:

- [numpy](http://www.numpy.org/),
- [scipy](https://www.scipy.org/),
- [pandas](http://pandas.pydata.org/),
- [openturns](http://www.openturns.org/),
- [scikit-learn](http://scikit-learn.org/),
- [scikit-optimize](https://github.com/scikit-optimize),
- [pyDOE](https://pythonhosted.org/pyDOE/),
- [matplotlib](https://matplotlib.org/),
- [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/).

Also, the software [R](https://www.r-project.org/) is needed with the package:

- [VineCopula](https://cran.r-project.org/web/packages/VineCopula/index.html).

However, we are still working on diminishing the number of dependencies. Especially the dependencies with R, by replacing the VineCopula R package with [vinecopulib](https://github.com/vinecopulib/vinecopulib). 

### Using Anaconda

## Examples

Several notebook examples are available in the directory [examples](https://github.com/NazBen/impact-of-dependence/tree/master/examples).
