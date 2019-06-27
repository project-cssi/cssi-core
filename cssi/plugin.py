#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""Plug-in interfaces for CSSI.

CSSI Library is built to support plugins to extend it's behaviour.
Following are a list of plugin types currently supported by CSSI.

Contributor Plugins
============
Contributor Plugins adds more contributing factors to the CSSI library
and extends the CSSI core algorithm.

To write a plugin for CSSI, create a module and implement the subclass
:class:`cssi.CSSIPlugin`. The abstract methods need to be implemented
specially the :meth:`cssi.CSSIPlugin.get_info` which provides plugin
metadata.

Example:
    Your module must contain a ``cssi_init`` function that can be used
    to register the plugin.

        from cssi.plugin import CSSIPlugin, ContributorPlugin, PluginType

        class CSSIPluginSample(CSSIPlugin, ContributorPlugin):
            ...
        def cssi_init(plugins, options):
            plugins.add_contributor_plugin(CSSIPluginSample())

Authors:
    Brion Mario

"""

from enum import Enum
from abc import ABC, abstractmethod


class CSSIPlugin(ABC):
    """Base class for all the CSSI plugins"""

    @abstractmethod
    def get_info(self, *args, **kwargs):
        """Get information about the plugin.

        Plugins must provide a plugin name and the plugin type through this
        function. This information will be used to store the plugin scores etc.
        It is advised to use the `_cssi_plugin` token in the name and better to
        use the same name given in the `setup.py` module. The function should
        return a dictionary in the following format.

        {
            "name": "<<unique_name>>_cssi_plugin",
            "type": <<plugin_type>>
        }

        Returns:
            dict: Dictionary containing useful information about the plugin.
        """
        return {}

    @abstractmethod
    def generate_final_score(self, *args, **kwargs):
        """Generate the final score of the plugin

        All plugins must generate a final score and expose it using
        this function.

        Returns:
            float: Final score for the plugin
        """
        return 0


class ContributorPlugin(ABC):
    """Base class for CSSI contributor plugins."""

    @abstractmethod
    def generate_unit_score(self, *args, **kwargs):
        """Generate the unit score for the plugin

        Contributor plugins can expose a unit score using this function
        and usually the head frame and the camera frame are passed in.
        Therefore, with those assets, a plugin can generate a score for
        a iteration.

        Returns:
            float: Unit score for the plugin.
         """
        return 0


class PluginType(Enum):
    """Used to identify different plugin types.

    These types can be used to distinguish between the different plugin
    types supported by the CSSI library.
    """
    CONTRIBUTOR = "CONTRIBUTOR"
