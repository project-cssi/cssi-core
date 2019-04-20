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
