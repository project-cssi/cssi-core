#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""The config module for the CSSI library

Authors:
    Brion Mario

"""

import configparser

from cssi.exceptions import CSSIException

DEFAULT_INTERNAL_CONFIG = "default.config.cssi"


class CSSIConfig(object):
    """CSSI library configuration."""

    def __init__(self):
        # init [latency] section options
        self.latency_weight = 0.0
        self.latency_boundary = 0.0

        # init [sentiment] section options
        self.sentiment_weight = 0.0

        # init [questionnaire] section options
        self.questionnaire_weight = 0.0

        # init [plugins] list
        self.plugins = []

        # init plugin options
        self.plugin_options = {}

    def read_from_file(self, filename):
        """Read configuration from a file.

        A filename can be passed in to load the configurations.

        """

        parser = CustomCSSIConfigParser()
        try:
            parser.read(filename)
        except configparser.Error as error:
            raise CSSIException("Couldn't read supplied configuration file {0}: {1}".format(filename, error))

        status = False
        try:
            for option_spec in self.CONFIG_FILE_OPTIONS:
                was_set = self._set_config_attribute_from_option(parser, *option_spec)
                if was_set:
                    status = True
        except ValueError as error:
            raise CSSIException("Couldn't read supplied configuration file {0}: {1}".format(filename, error))

        # cssi plugin options
        for plugin in self.plugins:
            if parser.has_section(plugin):
                self.plugin_options[plugin] = parser.get_section(plugin)
                status = True

        return status

    CONFIG_FILE_OPTIONS = [
        # Arguments for _set_config_attribute_from_option function

        # [run]
        ('plugins', 'run:plugins', 'list'),

        # [latency]
        ('latency_weight', 'latency:latency_weight', 'float'),
        ('latency_boundary', 'latency:latency_boundary', 'float'),

        # [sentiment]
        ('sentiment_weight', 'sentiment:sentiment_weight', 'float'),

        # [questionnaire]
        ('questionnaire_weight', 'questionnaire:questionnaire_weight', 'float'),
    ]

    def _set_config_attribute_from_option(self, parser, attr, where, type_=''):
        """Sets an attribute on self if it exists in the ConfigParser."""
        section, option = where.split(":")
        if parser.has_option(section, option):
            method = getattr(parser, 'get' + type_)
            setattr(self, attr, method(section, option))
            return True
        return False

    def get_plugin_options(self, plugin):
        """Returns a list of options for the plugins named `plugin`."""
        return self.plugin_options.get(plugin, {})

    def set_option(self, option_name, value):
        """Sets an option in the configuration."""

        # Check all the default options.
        for option_spec in self.CONFIG_FILE_OPTIONS:
            attr, where = option_spec[:2]
            if where == option_name:
                setattr(self, attr, value)
                return

        # Checks if it is a plugin option.
        plugin_name, _, key = option_name.partition(":")
        if key and plugin_name in self.plugins:
            self.plugin_options.setdefault(plugin_name, {})[key] = value
            return

        raise CSSIException("The option was not found {0}".format(option_name))

    def get_option(self, option_name):
        """Get an option from the configuration."""

        # Check all the default options.
        for option_spec in self.CONFIG_FILE_OPTIONS:
            attr, where = option_spec[:2]
            if where == option_name:
                return getattr(self, attr)

        # Checks if it is a plugin option.
        plugin_name, _, key = option_name.partition(":")
        if key and plugin_name in self.plugins:
            return self.plugin_options.get(plugin_name, {}).get(key)

        raise CSSIException("The option was not found {0}".format(option_name))


class CustomCSSIConfigParser(configparser.RawConfigParser):
    """Custom parser for CSSI configs

    This class extends the functionality of the `RawConfigParser` from `configparser`
    module. And it is useful for parsing lists and custom configuration options.

    """

    def __init__(self):
        configparser.RawConfigParser.__init__(self)

    def read(self, filenames, encoding=None):
        """Read a file name as UTF-8 configuration data."""
        return configparser.RawConfigParser.read(self, filenames=filenames, encoding=encoding)

    def has_option(self, section, option):
        """Checks if the config has a section"""
        has = configparser.RawConfigParser.has_option(self, section, option)
        if has:
            return has
        return False

    def has_section(self, section):
        """Checks if the config has a section"""
        has = configparser.RawConfigParser.has_section(self, section)
        if has:
            return section
        return False

    def options(self, section):
        """Checks if the config has options in a given section"""
        if configparser.RawConfigParser.has_section(self, section):
            return configparser.RawConfigParser.options(self, section)
        raise configparser.NoSectionError

    def get_section(self, section):
        """Returns the contents of a section, as a dictionary.

        Important for providing plugin options.
        """
        sections = {}
        for option in self.options(section):
            sections[option] = self.get(section, option)
        return sections

    def get(self, section, option, *args, **kwargs):
        """Get an option value for a given section."""
        val = configparser.RawConfigParser.get(self, section, option, *args, **kwargs)
        return val

    def getlist(self, section, option):
        """Read a list of strings."""
        _list = self.get(section, option)
        values = []
        for line in _list.split('\n'):
            for value in line.split(','):
                value = value.strip()
                if value:
                    values.append(value)
        return values


def read_cssi_config(filename):
    """Read the CSSI configuration file.

    Returns:
        config: CSSIConfig object

    """
    # load the defaults
    config = CSSIConfig()

    # read from the file
    is_read = config.read_from_file(filename=filename)

    # TODO: Log these messages
    if not is_read:
        config.read_from_file(filename="default.config.cssi")
        print("Configurations couldn't be loaded from file: {0}. Rolling back to internal defaults.".format(filename))
    else:
        print("Configuration was successfully read from file: {0}".format(filename))

    return config
