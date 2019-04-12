#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt.
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com.

"""
Brief:   Image processing based QA library for Cybersickness susceptibility testing

Author:  Brion Mario
"""

import os
from setuptools import setup, find_packages

REQUIREMENTS = [line.strip() for line in
                open(os.path.join("requirements.txt")).readlines()]

setup(name='cssi',
      version='0.1.0',
      url='https://github.com/brionmario/cssi-core',
      description='Image processing based QA library for Cybersickness susceptibility testing',
      author='Brion Mario',
      author_email='brion@apareciumlabs.com',
      packages=find_packages(exclude=('tests', 'docs')),
      package_data={'cssi': ['Readme.rst']},
      install_requires=REQUIREMENTS,
      include_package_data=True,
      license="The MIT License (MIT)"
      )
