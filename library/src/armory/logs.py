"""
Instantiate our package's logger
"""


import atexit
from collections import Counter
import logging

from rich.logging import RichHandler

modules = Counter(a=1, b=2)


def show_module_counts():
    print(modules.most_common())


atexit.register(show_module_counts)


IGNORE_PACKAGES = (
    "botocore tensorflow s3transfer botocode urllib3 h5py git fspec".split()
)


def origin_filter(record):
    package, _, _ = record.name.partition(".")
    if package in IGNORE_PACKAGES:
        return False
    modules[record.name] += 1
    return True


def configure_root_logger():
    # does what basicConfig() does, but more explicitly
    logging.root.setLevel(logging.NOTSET)
    handler = RichHandler(rich_tracebacks=True)
    handler.addFilter(origin_filter)
    formatter = logging.Formatter("%(message)s %(name)s", datefmt="[%X]")
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    return logging.root


rt = configure_root_logger()
print(rt)


# use `armory` as the logger name for any module that imports this one
log = logging.getLogger(__package__)

if __name__ == "__main__":
    for module in ("armory", "reticulating", "splines"):
        logging.getLogger(module).info("hello")
