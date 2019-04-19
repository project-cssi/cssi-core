#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""Interface for all the CSSI contributors

All the contributors of CSSI score generation should implement this
class.

Authors:
    Brion Mario

"""

from abc import ABC, abstractmethod


class CSSIContributor(ABC):

    @abstractmethod
    def score(self):
        pass
