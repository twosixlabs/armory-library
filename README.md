![armory logo](docs/assets/armory-logo.png)

---

[![CI][ci-badge]][ci-url]
[![PyPI Status Badge][pypi-badge]][pypi-url]
[![PyPI - Python Version][python-badge]][python-url]
[![License: MIT][license-badge]][license-url]
[![Docs][docs-badge]][docs-url]
[![Code style: black][style-badge]][style-url]
[![DOI](https://zenodo.org/badge/673882087.svg)](https://zenodo.org/doi/10.5281/zenodo.10041829)

# Overview

Armory-library is a pure Python library which allows the measurement of ML systems in
the face of adversarial attacks. It takes the years of experience gained and techniques
discovered under the [DARPA GARD program][gardproject] and makes it available to the
general ML user.


# Installation & Configuration

```bash
pip install armory-library
```

This is all that is needed to get a working Armory installation. However, Armory-library
is a library and does not contain any sample code. We provide examples in the
`armory-examples` repository which is released concurrently with Armory-library.

## Example programs

To install the examples, run:

```bash
pip install armory-examples
```

The [example source code][example-src], along with the [Armory-library
documentation](docs/index.md) is a good place to learn how to construct your own
evaluations using armory-library.


# Quick Look

We have provided an sample notebook using Armory to evaluate a food101 classifier
in the presence of a Project Gradient Descent (PGD) attack. The notebook can be
run for free on Google Colab to get a taste of how Armory works.

[![Open In Colab][colab-badge]][colab-url]

# Documentation

The Armory-library documentation is [published on Read the Docs][docs-url] or
can be viewed directly in [the docs directory](docs/index.md) of this repository.

# The historic GARD-Armory

Armory-library is the successor to the [GARD-Armory research program run under
DARPA][GARD-Armory]. As that program is nearing its conclusion, that repository
will be archived sometime in 2024 and there will be no further development in
GARD-Armory by the time you are reading this sentence. The development teams
for both GARD-Armory and Armory-library can be reached at <armory@twosixtech.com>

# Acknowledgment

This material is based upon work supported by the Defense Advanced Research Projects
Agency (DARPA) under Contract No. HR001120C0114 and US Army (JATIC) Contract No.
W519TC2392035. Any opinions, findings and conclusions or recommendations expressed in
this material are those of the author(s) and do not necessarily reflect the views of the
Defense Advanced Research Projects Agency (DARPA) or JATIC.



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
[style-url]: https://github.com/psf/black
[gardproject]: https://www.gardproject.org
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/twosixlabs/armory-library/blob/master/examples/notebooks/image_classification_food101.ipynb
[example-src]: https://github.com/twosixlabs/armory-library/tree/master/examples/src/armory/examples
[GARD-Armory]: https://github.com/twosixlabs/armory
