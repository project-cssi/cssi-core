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

logger = logging.getLogger(__name__)


class CSSI(object):

    def __init__(self, shape_predictor, debug=False, config_file=None):
        if config_file is None:
            self.config_file = "config.cssi"
        self.config_file = config_file
        self.debug = debug
        self.config = read_cssi_config(filename=self.config_file)
        self.latency = Latency(config=self.config, debug=self.debug, shape_predictor=shape_predictor)
        self.sentiment = Sentiment(config=self.config, debug=self.debug)
        logger.debug("Initialized CSSI library")



