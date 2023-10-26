![charmory logo](docs/assets/charmory.png)

---

[![CI][ci-badge]][ci-url]
[![PyPI Status Badge][pypi-badge]][pypi-url]
[![PyPI - Python Version][python-badge]][python-url]
[![License: MIT][license-badge]][license-url]
[![Docs][docs-badge]][docs-url]
[![Code style: black][style-badge]][style-url]
[![DOI](https://zenodo.org/badge/673882087.svg)](https://zenodo.org/doi/10.5281/zenodo.10041829)

# Overview

Charmory is a scaffolding name as we rework code coming from the `armory.` namespace.
It is slated to be renamed to `armory` once we adapt all legacy code that needs
to be adapted. We expect the `charmory.` namespace to be disappear by the end of 2023.

Presently, working use of armory-library, as shown in the `examples/` directory
imports symbols from both `armory` and `charmory` namespaces. Soon a global substitution
in user code from `charmory` to simply `armory` will be needed. We'll announce
in the release notes when this is needed.



# Installation & Configuration

```bash
pip install armory-library
```

Will make the `armory` and `charmory` namespaces available to your Python environment.


# Usage
See the documentation in the [armory-library docs](https://armory-library.readthedocs.io/en/latest/).

# Acknowledgment
This material is based upon work supported by the Defense Advanced Research Projects
Agency (DARPA) under Contract No. HR001120C0114. Any opinions, findings and
conclusions or recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Defense Advanced Research Projects
Agency (DARPA).



<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[ci-badge]: https://github.com/twosixlabs/armory-library/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/twosixlabs/armory-library/actions/
[pypi-badge]: https://badge.fury.io/py/armory-library.svg
[pypi-url]: https://pypi.org/project/armory-library
[python-badge]: https://img.shields.io/pypi/pyversions/armory-library
[python-url]: https://pypi.org/project/armory-library
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs-badge]: https://readthedocs.org/projects/armory/badge/
[docs-url]: https://readthedocs.org/projects/armory/
[style-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/ambv/black
