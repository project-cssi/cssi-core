#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""The main access point for the CSSI library

Authors:
    Brion Mario

"""

from abc import ABC, abstractmethod

from cssi.config import read_cssi_config


class CSSI(object):

    def __init__(self, config_file=None):
        if config_file is None:
            self.config_file = "config.cssi"
        self.config_file = config_file
        self.config = read_cssi_config(filename=self.config_file)


class CSSIContributor(ABC):
    """An abstract class for all the CSSI contributors

    All the contributors of CSSI score generation should extend this
    class and must implement the `generate_score` function.

    """

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug

    @abstractmethod
    def generate_score(self, *args):
        """"""
        pass
