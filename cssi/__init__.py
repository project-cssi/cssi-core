import os
import logging.config

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG_PATH = os.path.join(os.path.split(BASE_DIR)[0], 'logging.conf')

print(LOGGER_CONFIG_PATH)

logging.config.fileConfig(LOGGER_CONFIG_PATH, disable_existing_loggers=False)
