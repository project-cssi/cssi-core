#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (c) Copyright 2019 Brion Mario.
# (c) This file is part of the CSSI Core library and is made available under MIT license.
# (c) For more information, see https://github.com/brionmario/cssi-core/blob/master/LICENSE.txt
# (c) Please forward any queries to the given email address. email: brion@apareciumlabs.com

"""Questionnaire Module

This modules contains all the functions and classes related to the questionnaire
score generation.

Authors:
    Brion Mario

"""

import os
import json
import numpy as np

from cssi.exceptions import QuestionnaireMetaFileNotFoundException


# noinspection PyPep8Naming
class SSQ(object):
    QUESTIONNAIRE_MAX_TOTAL_SCORE = 235.62
    PRE_QUESTIONNAIRE_META_FILE_NAME = "ssq.pre.meta.json"
    POST_QUESTIONNAIRE_META_FILE_NAME = "ssq.post.meta.json"

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug

    def generate_score(self, pre, post):
        pre_N, pre_O, pre_D, pre_TS = self._calculate_pre_score(pre=pre)
        post_N, post_O, post_D, post_TS = self._calculate_post_score(post=post)
        return np.array(
            [max(0, (post_N - pre_N)), max(0, (post_O - pre_O)), max(0, (post_D - pre_D)), max(0, (post_TS - pre_TS)),
             np.array([pre_N, pre_O, pre_D, pre_TS]), np.array([post_N, post_O, post_D, post_TS])])

    def _calculate_pre_score(self, pre):
        return self._calculate_ssq_total_score(questionnaire=pre, filename=self.PRE_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_post_score(self, post):
        return self._calculate_ssq_total_score(questionnaire=post, filename=self.POST_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_ssq_total_score(self, questionnaire, filename):
        N = 0.0
        O = 0.0
        D = 0.0
        try:
            with open(self._get_meta_file_path(filename)) as meta_file:
                meta = json.load(meta_file)
                # Iterate through the symptoms and generate the
                # populate the `N`, `O` & `D` symptom scores.
                for s in meta["symptoms"]:
                    if s["weight"]["N"] == 1:
                        N += questionnaire[s["symptom"]]
                    if s["weight"]["O"] == 1:
                        O += questionnaire[s["symptom"]]
                    if s["weight"]["D"] == 1:
                        D += questionnaire[s["symptom"]]

                # Calculate the `N`, `O` & `D` weighted scores.
                # and finally compute the total score.
                N *= meta["conversion_multipliers"]["N"]
                O *= meta["conversion_multipliers"]["O"]
                D *= meta["conversion_multipliers"]["D"]
                TS = (N + O + D) * meta["conversion_multipliers"]["TS"]

                return np.array([N, O, D, TS])
        except FileNotFoundError as error:
            raise QuestionnaireMetaFileNotFoundException(
                "Questionnaire meta file couldn't not be found at %s" % (self._get_meta_file_path())
            ) from error

    @staticmethod
    def _get_meta_file_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta", filename)
