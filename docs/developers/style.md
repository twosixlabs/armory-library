Armory Style Guide
=====================
The following describes the stlying approach used during `armory` development.  If you plan to make
contributions to armory, please follow these style guidelines.  Note: Some of these are enforced by
our CI process and we have provided some `git hooks` to help with the formatting.  For more information
see [Pre-commit Hooks](#pre-commit-hooks) below.


# Formatting
All contributions to the repository must be formatted with [black](https://github.com/psf/black).
```
pip install black==22.*
black .
```
We will update black versioning annually following their [Stability Policy](https://black.readthedocs.io/en/stable/the_black_code_style/index.html#stability-policy).

As of version 0.16.1 `tools/format_json.py` no longer exists. Instead the built-in [json.tool](https://docs.python.org/3/library/json.html#module-json.tool) is used along with the `--sort-keys` and `--indent=4` flags.

We use [Flake8](https://flake8.pycqa.org/) for non-formatting PEP style enforcement.

    flake8

Our repo-specific Flake8 configuration is detailed in `.flake8`.

We use [isort](https://pycqa.github.io/isort/) to sort Python imports.

    isort --profile black *


### Pre-commit Hooks

The above tools can be ran automatically, prior to each commit, by installing
the Python `pre-commit` package and setting up git hooks.

```bash
python -m pip install pre-commit
python -m pre_commit install
```

Note: the `pre-commit` package is already installed as part of the `developer`
flavor dependencies.

```bash
pip install .[developer]
```

The pre-commit tool can be invoked manually as well.

```bash
# run all tools on currently-staged files
python -m pre_commit run
# run just the black hook
python -m pre_commit run black
# run all tools on all files in the repo
python -m pre_commit run --all-files
```
For more information about how to contribute to armory, see [Contributing to Armory](/CONTRIBUTING.md).


# Import Style

Imports in python files should be organized into three blocks, lexically ordered, after the docstring, and before other code:
* Block 1: built-in package imports
* Block 2: external package imports
* Block 3: internal package imports
These blocks should be separated by a single empty line. Here is an example:
```python
"""
Docstring
"""

import json
import os

from art import defences
import numpy as np
import requests

from armory.docker.management import ManagementInstance
from armory.utils.external_repo import download_and_extract_repos
from armory.logs import log
# ...
```

Exceptions are allowed for import error handling, required import ordering, or in-class/function imports.

## Additional Import Block for Downloaded GitHub Repos

A fourth import block may be added for external package imports that require downloading an external github repo via armory.
This is typically only used for some baseline models and art experimental attacks.
This must use the `armory.errors.ExternalRepoImport` context manager as follows (one `with` statement per external repo):
```
from armory.errors import ExternalRepoImport

with ExternalRepoImport(
    repo="colour-science/colour@v0.3.16",
    experiment="carla_obj_det_dpatch_undefended.json",
):
    import colour
```

`repo` refers to the GitHub repo name (optionally with `@tag`).
`experiment` refers to the `.json` file in the `scenario_config` directory that uses this repo.
These repos are specifically *NOT* installed in the armory-supported docker containers and conda environments, and are downloaded at runtime.
The rationale is that they are not a core part of the library and are meant to mirror usage of individuals evaluating their own models.
