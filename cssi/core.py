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
import os
import logging
# `Keras` was hanging when trying to use the model.predict
# and swapping the backend from `Tensorflow` to `Theano`
# fixed the issue.
os.environ['KERAS_BACKEND'] = 'theano'

from cssi.config import read_cssi_config
from cssi.latency import Latency
from cssi.sentiment import Sentiment
from cssi.questionnaire import SSQ

logger = logging.getLogger(__name__)


class CSSI(object):
    """The main access point for the CSSI library"""

    def __init__(self, shape_predictor, debug=False, config_file=None):
        """ Initializes all the core modules in the CSSI Library.

        :param shape_predictor: Path to the landmark detector.
        :param debug: Boolean indicating if debug mode should be activated or not.
        :param config_file: A file containing all the configurations for CSSI.
        """
        # If no config file name is passed in, defaults to `config.cssi`
        if config_file is None:
            self.config_file = "config.cssi"
        self.config_file = config_file
        # Sets the debug mode
        self.debug = debug
        # Tries to read the config file.
        self.config = read_cssi_config(filename=self.config_file)
        # Initialize the latency capturing module
        self.latency = Latency(config=self.config, debug=self.debug, shape_predictor=shape_predictor)
        # Initializing the Sentiment capturing module
        self.sentiment = Sentiment(config=self.config, debug=self.debug)
        # Initializing the questionnaire module.
        self.questionnaire = SSQ(config=self.config, debug=self.debug)
        logger.debug("CSSI library initialized......")
