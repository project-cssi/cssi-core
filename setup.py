#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""
Brief:   Image processing based python library for Cybersickness susceptibility testing

Author:  Brion Mario
"""

import os

from setuptools import setup, find_packages

PKG = "cssi"
VERSION_FILE_PATH = os.path.join(os.path.split(__file__)[0], "{0}/version.py".format(PKG))
VERSION = "0.0.0"  # default fallback
REPO_URL = "https://github.com/brionmario/cssi-core"
DESCRIPTION = "Image processing based QA library for Cybersickness susceptibility testing"
AUTHOR = "Brion Mario"
AUTHOR_EMAIL = "brion@apareciumlabs.com"
LICENSE = "The MIT License (MIT)"

# Search the `cssi/version.py` file and extract the version
try:
    with open(VERSION_FILE_PATH) as file:
        content = {}
        exec(file.read(), content)
        VERSION = content["__version__"]
except FileNotFoundError:
    raise RuntimeError("Unable to read version file on path {0}".format(VERSION_FILE_PATH,))

REQUIREMENTS = [line.strip() for line in
                open(os.path.join("requirements.txt")).readlines()]

setup(name=PKG,
      version=VERSION,
      url=REPO_URL,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=find_packages(exclude=('tests', 'docs')),
      package_data={PKG: ['Readme.rst']},
      install_requires=REQUIREMENTS,
      include_package_data=True,
      license=LICENSE
      )
