[![Build Status](https://travis-ci.org/NazBen/impact-of-dependence.svg?branch=master)](https://travis-ci.org/NazBen/impact-of-dependence)
[![CircleCI](https://circleci.com/gh/NazBen/impact-of-dependence.svg?style=svg)](https://circleci.com/gh/NazBen/impact-of-dependence)
[![codecov](https://codecov.io/gh/NazBen/impact-of-dependence/branch/master/graph/badge.svg)](https://codecov.io/gh/NazBen/impact-of-dependence)
# Impact of Dependencies

A python package to study the influence of dependencies between random variables in a probabilistic study. 

For the moment, the class `ConservativeEstimate` create a probabilistic model with an incomplete description of the dependence structure between the input variables. 

estimates a quantity of intereset of a model output

## Package Dependencies

Several external modules are required:

    - numpy,
    - scipy,
    - pandas,
    - openturns,
    - rpy2,
    - pyDOE.
    
The software R is also needed to execture the 

And some are optional:

    - Matplotlib,
    - futures.
    

## Install

```
  python setup.py install
```

or 

```
pip install -e.
```

## Examples

Several notebook examples are available in the directory [examples](https://github.com/NazBen/impact-of-dependence/tree/master/examples).
