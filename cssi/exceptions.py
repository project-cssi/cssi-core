class BaseCSSIException(Exception):
    """The base of all CSSI library exceptions."""
    pass


class CSSIException(BaseCSSIException):
    """An exception specific to CSSI library"""
    pass


class QuestionnaireMetaFileNotFoundException(CSSIException):
    """The questionnaire meta file couldn't be found."""
    pass


class LandmarkDetectorFileNotFoundException(CSSIException):
    """The landmark detector file couldn't be found."""
    pass
