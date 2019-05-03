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

from cssi.contributor_base import CSSIContributor
from cssi.exceptions import CSSIException


class SSQ(CSSIContributor):
    QUESTIONNAIRE_MAX_TOTAL_SCORE = 235.62
    PRE_QUESTIONNAIRE_META_FILE_NAME = "ssq.pre.meta.json"
    POST_QUESTIONNAIRE_META_FILE_NAME = "ssq.post.meta.json"

    def generate_final_score(self, pre, post):
        """Generates the final questionnaire score.
        Args:
            pre (dict): Pre questionnaire results.
            post (dict): Post questionnaire results.
        Returns:
            float: The total questionnaire score.
        Examples:
            >>> cssi.questionnaire.generate_final_score(pre, post)
        """
        # Calculate the pre and post questionnaire scores.
        pre_n, pre_o, pre_d, pre_ts = self._calculate_pre_score(pre=pre)
        post_n, post_o, post_d, post_ts = self._calculate_post_score(post=post)

        # Calculating the total questionnaire score.
        tq = ((post_ts - pre_ts) / self.QUESTIONNAIRE_MAX_TOTAL_SCORE) * 100

        # check if score is less than 0, if yes, moderate it to 0
        if tq < 0:
            tq = 0

        return tq

    def _calculate_pre_score(self, pre):
        return self._calculate_ssq_total_score(questionnaire=pre, filename=self.PRE_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_post_score(self, post):
        return self._calculate_ssq_total_score(questionnaire=post, filename=self.POST_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_ssq_total_score(self, questionnaire, filename):
        _n = 0.0
        _o = 0.0
        _d = 0.0
        try:
            with open(self._get_meta_file_path(filename)) as meta_file:
                meta = json.load(meta_file)
                # Iterate through the symptoms and generate the
                # populate the `N`, `O` & `D` symptom scores.
                for s in meta["symptoms"]:
                    if s["weight"]["N"] == 1:
                        _n += int(questionnaire[s["symptom"]])
                    if s["weight"]["O"] == 1:
                        _o += int(questionnaire[s["symptom"]])
                    if s["weight"]["D"] == 1:
                        _d += int(questionnaire[s["symptom"]])

                # Calculate the `N`, `O` & `D` weighted scores.
                # and finally compute the total score.
                n = _n * meta["conversion_multipliers"]["N"]
                o = _o * meta["conversion_multipliers"]["O"]
                d = _d * meta["conversion_multipliers"]["D"]
                ts = (_n + _o + _d) * meta["conversion_multipliers"]["TS"]

                return n, o, d, ts
        except FileNotFoundError as error:
            raise CSSIException(
                "Questionnaire meta file couldn't not be found at %s" % (self._get_meta_file_path(filename))
            ) from error

    @staticmethod
    def _get_meta_file_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta", filename)
