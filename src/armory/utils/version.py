"""
ARMORY Versions use "Semantic Version" scheme where stable releases will have versions
like `0.14.6`.  Armory uses the most recent git tag for versioning. For example if the
most recent git tag is `v0.14.6`, then the version will be `0.14.6`.

If you are a developer, the version will be constructed from the most recent tag plus a
suffix of gHASH where HASH is the short hash of the most recent commit. For example,
if the most recent git tag is v0.14.6 and the most recent commit hash is 1234567 then
the version will be 0.14.6.g1234567. This scheme does differ from the scm strings
which also have a commit count and date in them like 1.0.1.dev2+g0c5ffd9.d20220314181920
which is a bit ungainly.
"""

import functools
from importlib import metadata
from pathlib import Path

import setuptools_scm

from armory.logs import log

PYPI_PACKAGE_NAME = "armory-testbed"


def get_metadata_version(package: str = PYPI_PACKAGE_NAME) -> str:
    """Retrieve the version from the package metadata"""
    return str(metadata.version(package))


def get_tag_version(git_dir: Path = None) -> str:
    """Retrieve the version from the most recent git tag, return empty string on
    failure"""
    project_root = Path(__file__).parent.parent.parent
    scm_config = {
        "root": project_root,
        "version_scheme": "post-release",
    }
    if not Path(project_root / ".git").is_dir():
        raise LookupError("Unable to find `.git` directory!")
    return setuptools_scm.get_version(**scm_config)


def get_build_hook_version() -> str:
    """Retrieve the version from the build hook"""
    try:
        from armory.__about__ import __version__

        return __version__
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Unable to extract armory version from __about__.py")


@functools.lru_cache(maxsize=1, typed=False)
def get_version(package_name=PYPI_PACKAGE_NAME) -> str:
    errors = []
    try:
        version = get_tag_version()
        log.debug(f"version {version} found via git tag")
        return version
    except LookupError as e:
        error_str = f"version not found via git tag: {e}"
        log.debug(error_str)
        errors.append(error_str)

    try:
        version = get_build_hook_version()
        log.debug(f"version {version} found via build hook at armory/__about__.py")
        return version
    except ModuleNotFoundError as e:
        error_str = f"version not found via build hook at armory/__about__.py: {e}"
        log.debug(error_str)
        errors.append(error_str)

    try:
        version = get_metadata_version()
        log.debug(f"version {version} found via package metadata")
        return version
    except metadata.PackageNotFoundError as e:
        error_str = f"version not found via package metadata: Package {e} not installed"
        log.debug(error_str)
        errors.append(error_str)

    errors.append("Unable to determine version number!")
    verbose_errors = "\n".join(errors)
    log.error(verbose_errors)
    raise RuntimeError(verbose_errors)
