# becquerel

[![tests](https://github.com/lbl-anp/becquerel/actions/workflows/tests.yaml/badge.svg)](https://github.com/lbl-anp/becquerel/actions/workflows/tests.yaml)
[![Coverage Status](https://coveralls.io/repos/github/lbl-anp/becquerel/badge.svg?branch=main)](https://coveralls.io/github/lbl-anp/becquerel?branch=main)
[![PyPI version](https://img.shields.io/pypi/v/becquerel.svg)](https://pypi.org/project/becquerel)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/becquerel.svg)](https://pypi.org/project/becquerel)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Becquerel is a Python package for analyzing nuclear spectroscopic
measurements. The core functionalities are reading and writing different
spectrum file types, fitting spectral features, performing detector
calibrations, and interpreting measurement results. It includes tools for
plotting radiation spectra as well as convenient access to tabulated nuclear
data, and it will include fits of different spectral features. It relies
heavily on the standard scientific Python stack of numpy, scipy, matplotlib,
and pandas. It is intended to be general-purpose enough that it can be useful
to anyone from an undergraduate taking a laboratory course to the advanced
researcher.

## Installation

```bash
pip install becquerel
```

## Features in development (contributions welcome!)

- Reading additional `Spectrum` file types (N42, CHN, CSV)
- Writing `Spectrum` objects to various standard formats
- Fitting spectral features with Poisson likelihood

If you are interested in contributing or are want to install the package from
source, please see the instructions in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Reporting issues

When reporting issues with `becquerel`, please provide a minimum working example
to help identify the problem and tag the issue as a `bug`.

## Feature requests

For a feature request, please create an issue and label it as a `new feature`.

## Dependencies

External dependencies are listed in `requirements.txt` and will be installed
automatically with the standard `pip` installation. They can also be installed
manually with the package manager of your choice (`pip`, `conda`, etc).
The dependencies `beautifulsoup4`, `lxml` and `html5lib` are necessary for
[`pandas`][1].

Developers require additional requirements which are listed in
`requirements-dev.txt`. We use [`pytest`][2] for unit testing, [`black`][3] for
code formatting and are converting to [`numpydoc`][4].

[1]: https://pandas.pydata.org/pandas-docs/stable/install.html#dependencies
[2]: https://docs.pytest.org/en/latest/
[3]: https://black.readthedocs.io/en/stable/
[4]: https://numpydoc.readthedocs.io/en/latest/format.html

## Copyright Notice

becquerel (bq) Copyright (c) 2017-2021, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to receipt
of any required approvals from the U.S. Dept. of Energy) and University of
California, Berkeley. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
