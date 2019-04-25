from abc import ABC, abstractmethod


class CSSIContributor(ABC):
    """An abstract class for all the CSSI contributors

    All the contributors of CSSI score generation should extend this
    class and must implement the `generate_score` function.

    """

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug

    @abstractmethod
    def generate_final_score(self, *args):
        """"""
        pass

    @abstractmethod
    def generate_unit_score(self, *args):
        """"""
        pass
