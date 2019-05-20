from abc import ABC, abstractmethod


class CSSIPlugin(ABC):
    """Base class for all the CSSI plugins"""

    @abstractmethod
    def get_info(self, *args, **kwargs):
        """Get information about the plugin.

        Plugins must provide a plugin name and the plugin type through this
        function. This information will be used to store the plugin scores etc.
        It is advised to use the `_cssi_plugin` token in the name and better to
        use the same name given in the `setup.py` module. The function should
        return a dictionary in the following format.

        {
            "name": "<<unique_name>>_cssi_plugin",
            "type": <<plugin_type>>
        }

        Returns:
            dict: Dictionary containing useful information about the plugin.
        """
        return {}

    @abstractmethod
    def generate_final_score(self, *args, **kwargs):
        """Generate the final score of the plugin

        All plugins must generate a final score and expose it using
        this function.

        Returns:
            float: Final score for the plugin
        """
        return 0
