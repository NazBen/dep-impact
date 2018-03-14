#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='dep-impact',
    version='0.1',
    description='A Python package on the influence of dependencies in a structural reliability problem.',
    url='https://github.com/NazBen/dep-impact',
    author='Nazih Benoumechiara',
    author_email='nazih.benoumechiara@gmail.com',
    license='MIT',
    keywords='copula reliability regular-vines',
    packages=['depimpact'],
    install_requires=required
)
