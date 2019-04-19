def _must_be_implemented(that, func_name):
    """Helper function to raise NotImplementedError in interfaces."""
    if hasattr(that, '_cssi_plugin_name'):
        item = 'Plugin'
        name = that._cssi_plugin_name
    else:
        item = 'Class'
        _class = that.__class__
        name = '{_class.__module__}.{_class.__name__}'.format(_class=_class)

    raise NotImplementedError(
        '{item} {name!r} needs to implement the {func_name}() function'.format(
            item=item, name=name, func_name=func_name
            )
        )
