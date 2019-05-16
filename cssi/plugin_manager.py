#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""This module provides plugin support for the CSSI Library

Authors:
    Brion Mario

"""

import sys

from cssi.exceptions import CSSIException


class Plugins:
    """Manages all the CSSI plugins"""

    def __init__(self):
        pass

    @classmethod
    def init_plugins(cls, modules, config):
        """Load the plugins"""
        plugins = cls()

        for module in modules:
            plugins.current_module = module
            __import__(module)
            mod = sys.modules[module]

            cssi_init = getattr(mod, "cssi_init", None)
            if not cssi_init:
                raise CSSIException(
                    "The plugin module {0} doesn't contain a cssi_init function".format(module)
                )

            options = config.get_plugin_options(module)
            cssi_init(plugins, options)

        plugins.current_module = None
        return plugins



