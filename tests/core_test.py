import pytest
from cssi.core import CSSI

class TestCSSI(object):
    def test_init(self):
        self.make_file("config.cssi", """\
           [run]
            plugins =
                heartrate.plugin

            [latency]
            latency_weight = 40
            latency_boundary = 3

            [sentiment]
            sentiment_weight = 30

            [questionnaire]
            questionnaire_weight = 20

            [heartrate.plugin]
            weight = 10
            """)
        print(self.make_file)
