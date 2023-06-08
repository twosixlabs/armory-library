![charmory logo](docs/assets/charmory.png)

---

[![CI][ci-badge]][ci-url]
[![PyPI Status Badge][pypi-badge]][pypi-url]
[![PyPI - Python Version][python-badge]][python-url]
[![License: MIT][license-badge]][license-url]
[![Docs][docs-badge]][docs-url]
[![Code style: black][style-badge]][style-url]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7561756.svg)](https://doi.org/10.5281/zenodo.7561756)


# Overview

Charmory is the temporary name for the JATIC armory replacement. This name is
mostly for the use of the developers as we move function and code from Armory to
Charmory. Interim releases will be provided under this scaffolding name, although
most users will still `import armory` as there is no Python package "charmory".



# Installation & Configuration

```bash
pip install charmory
```

Will make the `armory` namespace available to your Python environment. In the open-source
version, the installation name was "armory-testbed" which provided also provided
the `armory` namespace.

# Usage
See the documentation [here](https://jatic.pages.jatic.net/twosix/armory/).

# Examples
## Patch Attack
![Benign Image](docs/assets/patch_attack_example_benign.png)
![Adversarial Image](docs/assets/patch_attack_example_adversarial.png)

# Acknowledgment
This material is based upon work supported by the Defense Advanced Research Projects
Agency (DARPA) under Contract No. HR001120C0114. Any opinions, findings and
conclusions or recommendations expressed in this material are those of the author(s)
and do not necessarily reflect the views of the Defense Advanced Research Projects
Agency (DARPA).

# Points of Contact
POC: Matt Wartell @matt.wartell
DPOC: Christopher Woodall @christopher.woodall


<!-- TODO: repoint to JATIC CI or drop the badges -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[ci-badge]: https://github.com/twosixlabs/armory/workflows/GitHub%20CI/badge.svg
[ci-url]: https://github.com/twosixlabs/armory/actions/
[pypi-badge]: https://badge.fury.io/py/armory-testbed.svg
[pypi-url]: https://pypi.org/project/armory-testbed
[python-badge]: https://img.shields.io/pypi/pyversions/armory-testbed
[python-url]: https://pypi.org/project/armory-testbed
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs-badge]: https://readthedocs.org/projects/armory/badge/
[docs-url]: https://readthedocs.org/projects/armory/
[style-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/ambv/black
