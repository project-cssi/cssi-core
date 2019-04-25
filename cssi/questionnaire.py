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

from cssi.contributor import CSSIContributor
from cssi.exceptions import QuestionnaireMetaFileNotFoundException


# noinspection PyPep8Naming
class SSQ(CSSIContributor):
    QUESTIONNAIRE_MAX_TOTAL_SCORE = 235.62
    PRE_QUESTIONNAIRE_META_FILE_NAME = "ssq.pre.meta.json"
    POST_QUESTIONNAIRE_META_FILE_NAME = "ssq.post.meta.json"

    def generate_final_score(self, pre_total, post_total):
        TQ = ((post_total - pre_total) / self.QUESTIONNAIRE_MAX_TOTAL_SCORE) * 100
        return TQ

    def generate_unit_score(self, pre, post):
        pre_N, pre_O, pre_D, pre_TS = self._calculate_pre_score(pre=pre)
        post_N, post_O, post_D, post_TS = self._calculate_post_score(post=post)
        return pre_TS, post_TS, [pre_N, pre_O, pre_D, pre_TS], [post_N, post_O, post_D, post_TS]

    def _calculate_pre_score(self, pre):
        return self._calculate_ssq_total_score(questionnaire=pre, filename=self.PRE_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_post_score(self, post):
        return self._calculate_ssq_total_score(questionnaire=post, filename=self.POST_QUESTIONNAIRE_META_FILE_NAME)

    def _calculate_ssq_total_score(self, questionnaire, filename):
        _N = 0.0
        _O = 0.0
        _D = 0.0
        try:
            with open(self._get_meta_file_path(filename)) as meta_file:
                meta = json.load(meta_file)
                # Iterate through the symptoms and generate the
                # populate the `N`, `O` & `D` symptom scores.
                for s in meta["symptoms"]:
                    if s["weight"]["N"] == 1:
                        _N += questionnaire[s["symptom"]]
                    if s["weight"]["O"] == 1:
                        _O += questionnaire[s["symptom"]]
                    if s["weight"]["D"] == 1:
                        _D += questionnaire[s["symptom"]]

                # Calculate the `N`, `O` & `D` weighted scores.
                # and finally compute the total score.
                N = _N * meta["conversion_multipliers"]["N"]
                O = _O * meta["conversion_multipliers"]["O"]
                D = _D * meta["conversion_multipliers"]["D"]
                TS = (_N + _O + _D) * meta["conversion_multipliers"]["TS"]

                return np.array([N, O, D, TS])
        except FileNotFoundError as error:
            raise QuestionnaireMetaFileNotFoundException(
                "Questionnaire meta file couldn't not be found at %s" % (self._get_meta_file_path())
            ) from error

    @staticmethod
    def _get_meta_file_path(filename):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta", filename)
