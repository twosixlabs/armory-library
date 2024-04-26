# The logging level defaults to INFO, rather than WARNING because logs are the main way
# that armory communicates to the user.
#
# Console log messages are currently sent to stderr.
#
# This module creates the stderr sink at initialization, so `log.debug()` and friends are available at import.


import logging
import sys

LOGGING_LINE_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'

LOGGING_DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S'

ARMORY_LOGGING_STREAM = sys.stderr


def _configure_armory_loggers(root_module_name):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "armory_formatter": {
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "armory_handler": {
                    "formatter": "armory_formatter",
                    "class": "logging.StreamHandler",
                    "stream": ARMORY_LOGGING_STREAM,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["armory_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
