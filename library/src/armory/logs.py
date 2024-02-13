"""
Instantiate our package's logger
"""


import logging

from rich.logging import RichHandler

# TODO: a different possible strategy would be to set up handlers for each
# module that wants to log to the console such that messages from, for example,
# h5py could have its own level set and logging package would handle this natively

IGNOREABLE_PACKAGES = """botocore tensorflow s3transfer botocode urllib3 h5py
    git fsspec boto3 filelock hooks awsrequesti auth""".split()


def origin_filter(record) -> bool:
    """discard log messages if they originate from ignorable packages"""
    package, _, _ = record.name.partition(".")
    return package not in IGNOREABLE_PACKAGES


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
