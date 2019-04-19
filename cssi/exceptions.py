class BaseCSSIException(Exception):
    """The base of all CSSI library exceptions."""
    pass


class CSSIException(BaseCSSIException):
    """An exception specific to CSSI library"""
    pass


class QuestionnaireMetaFileNotFoundError(FileNotFoundError):
    """The questionnaire couldn't be parsed."""
    pass
