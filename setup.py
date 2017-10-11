#!/usr/bin/env python
"""An OpenTURNS module for conservative estimation of risk quantities for
incomplete dependence structure.

See:
https://github.com/NazBen/impact-of-dependence
"""

# Always prefer setuptools over distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
from codecs import open
from os import path

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='impact-of-dependence',
    version='0.3',
    description='A Python package to compute the impact of dependencies in \
        probabilistic studies.',
    url='https://github.com/NazBen/impact-of-dependence',
    author='Nazih Benoumechiara',
    author_email='nazih.benoumechiara@gmail.com',
    license='MIT',
    keywords='copula reliability openturns',
    packages=['dependence'],
    install_requires=required
)
