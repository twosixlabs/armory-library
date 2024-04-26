"""This namespace package contains Armory Library."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .logs import _configure_armory_loggers

# Configure Armory root logger
_configure_armory_loggers(root_module_name=__name__)
