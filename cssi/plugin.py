class CSSIPlugin(object):
    """Base class for all the CSSI plugins"""

    def get_info(self):
        """Get information about the plugin.

        Every plugin needs to provide a valid plugin name i.e. "<<unique_name>>_cssi_plugin" and
        should specify the plugin type.

        Returns:
            dict: Dictionary containing useful information about the plugin.
                ex: {"name": "heart_rate_cssi_plugin", "type": "contributor"}
        """
        return {}
