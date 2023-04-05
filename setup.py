#! /usr/bin/env python

from setuptools import setup


setup(name='landing_stiefel',
      install_requires=['torch', 'geoopt', 'scipy', 'geotorch', 'matplotlib'],
      packages=['landing_stiefel'],
      version='0.0'
      )
