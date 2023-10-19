#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:41:15 2019
@author: aurelien
"""

from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()
    
setup(name='fluct_prof',
      version='1.0',
      description='Fluctuometry profiler',
      install_requires=['tk','matplotlib','lmfit', 'ttkwidgets', 'scipy', 'seaborn', 'tifffile', 'xlsxwriter'],
      long_description = readme(),
      url='',
      packages = find_packages(),
      author='Taras Sych',
      author_email='taras.sych@ki.se',
      package_data = {},
      include_package_data = False,
      license='MIT',
      zip_safe=False)