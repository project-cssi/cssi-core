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
from cssi.plugin_manager import Plugins
from cssi.exceptions import CSSIException

logger = logging.getLogger(__name__)


class CSSI(object):
    """The main access point for the CSSI library"""

    def __init__(self, shape_predictor, debug=False, config_file=None):
        """Initializes all the core modules in the CSSI Library.

        Args:
            shape_predictor (str): Path to the landmark detector.
            debug (bool): Boolean indicating if debug mode should be activated or not.
            config_file (str): A file containing all the configurations for CSSI.

        Examples:
            >>> cssi = CSSI(shape_predictor="hmd_face_landmarks.dat", debug=True, config_file="config.cssi")
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
        # Load the plugins specified in the configuration file
        self.plugins = Plugins.load_plugins(modules=self.config.plugins, config=self.config, debug=self.debug)
        logger.debug("CSSI library initialized......")

    def generate_cssi_score(self, tl, ts, tq, ps=None):
        """Generators the final CSSI score.

        This is the core function of the CSSI library and it takes in the scores and
        generates the final CSSI score for the test session.

        Args:
            tl (float): Total latency score
            ts (float): Total sentiment score
            tq (float): Total questionnaire score
            ps (dict): A dictionary containing plugin scores.
                ex: {"heartrate.plugin": 40.00, "ecg.plugin": 55.90}
        Returns:
            float: The CSSI score.
        Raises:
            CSSIException: If the calculations couldn't be completed successfully
                this exception will be thrown.
        Examples:
            >>> cssi.generate_cssi_score(tl, ts, tq, plugin_scores)
        """
        tot_ps = 0.0  # Variable to store the sum of the plugin scores
        tot_pw = 0  # Variable to keep track total plugin weight

        # Checks if any plugins are provided for score calculation.
        if ps is not None:
            for key, value in ps.items():
                plugin_name = key
                # Checks if the plugin is registered in the configuration file
                # If not, raises an exception.
                if plugin_name not in self.config.plugins:
                    raise CSSIException("The plugin {0} appears to be invalid.".format(plugin_name))
                else:
                    plugin_weight = float(self.config.plugin_options[plugin_name]["weight"]) / 100
                    plugin_score = value

                    # Checks if the passed in plugin score is less than 100.
                    # If not an exception will be thrown.
                    if plugin_score > 100:
                        raise CSSIException("Invalid score provided for the plugin: {0}.".format(plugin_name))

                    # Ads the current plugin score to the total plugin score.
                    tot_ps += plugin_score * plugin_weight
                    # Ads the current plugin weight to the total plugin weight percentage.
                    tot_pw += plugin_weight

        lw = float(self.config.latency_weight) / 100  # latency weight percentage
        sw = float(self.config.sentiment_weight) / 100  # sentiment weight percentage
        qw = float(self.config.questionnaire_weight) / 100  # questionnaire weight percentage

        # Checks if the total weight is less than 100 percent.
        if (lw + sw + qw + tot_pw) > 1:
            raise CSSIException("Invalid weight configuration. Please reconfigure and try again")

        # Calculating the CSSI score
        cssi = (tl * lw) + (ts * sw) + (tq * qw) + tot_ps

        # Double checks if the generated CSSI score is less than 100.
        if cssi > 100:
            raise CSSIException("Invalid CSSI score was generated. Please try again")

        return cssi

    def generate_contributor_plugin_unit_scores(self, head_frame, scene_frame):
        """Generate the unit scores of all the plugins

        Contributor plugins can generate a unit score by implementing
        the :meth:`ContributorPlugin.generate_unit_score` method.

        Returns:
            List: A List of dictionaries containing the plugin names and scores.
        """
        plugin_scores = []

        for plugin in self.plugins.contributor_plugins:
            plugin_name = plugin.get_info()["name"]
            plugin_score = plugin.generate_unit_score(head_frame=head_frame, scene_frame=scene_frame)
            plugin_scores.append({"name": plugin_name, "score": plugin_score})

        return plugin_scores

    def generate_plugin_final_scores(self, **kwargs):
        """Generate the final scores of the plugins

        All the plugins implement the :meth:`CSSIPlugin.generate_final_score`
        and this function will generate the final scores of all the plugins.

        Returns:
            List: A list of plugin final scores.
        """
        scores = {}
        final_scores = {}

        if kwargs is not None:
            for key, value in kwargs.items():
                if key == "scores":
                    scores = value

        # Calculate total score of the contributor plugins
        for plugin in self.plugins.contributor_plugins:
            plugin_name = plugin.get_info()["name"]
            if plugin_name in scores:
                final_score = plugin.generate_final_score(scores=scores[plugin_name])
                final_scores[plugin_name] = final_score

        return final_scores
