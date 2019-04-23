import logging
from logging import NullHandler

# Add NullHandler to avoid errors if the host application
# doesn't have logging configured.
default_logger = logging.getLogger("cssi.core")
default_logger.addHandler(NullHandler())

# Set the default level to WARN
if default_logger.level == logging.NOTSET:
    default_logger.setLevel(logging.WARN)
