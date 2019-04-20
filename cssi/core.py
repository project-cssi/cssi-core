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


class CSSI(object):

    def __init__(self, config_file=True):
        pass


class CSSIContributor(ABC):
    """Interface for all the CSSI contributors

    All the contributors of CSSI score generation should implement this
    class.

    """

    @abstractmethod
    def score(self):
        """"""
        pass
