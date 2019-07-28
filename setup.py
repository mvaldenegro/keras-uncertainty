#!/usr/bin/env python

from setuptools import setup, find_packages

long_description='''
Keras-Uncertainty is a library to perform uncertainty quantification of Machine Learning models, focusing on Epistemic Uncertainty.
Basically we need models that know what they don't know. This has a variety of real world applications.

This library is only compatible with Python 3.x.
'''

setup(name='Keras-Uncertainty',
      version='0.0.1',
      description='Uncertainty Quantification for Keras models',
      long_description=long_description,
      author='Matias Valdenegro-Toro',
      author_email='matias.valdenegro@gmail.com',
      url='https://github.com/mvaldenegro/keras-uncertainty',
      download_url='https://github.com/mvaldenegro/keras-uncertainty/releases',
      license='LGPLv3',
      install_requires=['keras>=2.2.0', 'numpy'],
      packages=find_packages()
     )