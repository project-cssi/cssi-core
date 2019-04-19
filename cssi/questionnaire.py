class Questionnaire(object):

    MAX_QUESTIONNAIRE_SCORE = 100

    def __init__(self, pre, post):
        self.pre = pre
        self.post = post

    def _calculate_pre_score(self):
        pass

    def _calculate_post_score(self):
        pass
