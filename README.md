# becquerel

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

## Installation instructions

### As a user

```
pip install -r requirements.txt
python setup.py install --user
```

Before reinstalling, be sure to either remove the ```build``` directory
or run:

```
python setup.py clean --all
```

### As a developer

```
pip install -r requirements-dev.txt
python setup.py develop
```

(It is more convenient to use `develop` so that the code is soft-linked
from the installation directory, and the installed package will always use
the current version of code.)

## Running the tests

(Requires `requirements-dev.txt` to be installed)
To run the tests using `pytest`, from the root directory of the repo:

```
pytest
```

(`python setup.py test` is still supported also.)
By default, a code coverage report is printed to the terminal.
Tests marked `webtest` or `plottest` are by default skipped for the sake of
speed. To run all tests, clear the pre-configured markers option:

```
pytest -m ""
```

To produce an HTML code coverage report in directory `htmlcov`
with line-by-line highlighting:

```
pytest --cov-report html:htmlcov
```

## Dependencies

External dependencies are listed in `requirements.txt` and can be installed
with `pip` (see [Installation instructions][0]) or manually. The dependencies
`beautifulsoup4`, `lxml` and `html5lib` are necessary for [`pandas`][1].
Developers additionally need [`pytest`][2] and are encouraged to use
[`pylint`][3], [`pycodestyle`][4], [`pydocstyle`][5] and [`yapf`][6] for
proper code formatting.

[0]: #installation-instructions
[1]: https://pandas.pydata.org/pandas-docs/stable/install.html#dependencies
[2]: https://docs.pytest.org
[3]: https://pylint.readthedocs.io
[4]: http://pycodestyle.pycqa.org
[5]: http://www.pydocstyle.org
[6]: https://github.com/google/yapf

## Code Style Guide

Use [google standards](https://google.github.io/styleguide/pyguide.html)

## Linter

* Use `flake8` in your IDE
* Use `pylint` from command line (as in style guide)

## Features in development (contributions welcome!)

* Reading additional `Spectrum` file types (N42, CHN, CSV)
* Writing `Spectrum` objects to various standard formats
* Fitting spectral features (e.g., gaussian lines with different background models)

If you are interested in contributing, please see the guidelines in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Copyright Notice

Becquerel v. 0.1, Copyright (c) 2017, The Regents of the University of
California (UC), through Lawrence Berkeley National Laboratory, and the UC
Berkeley campus (subject to receipt of any required approvals from the U.S.
Dept. of Energy). All rights reserved. If you have questions about your rights
to use or distribute this software, please contact Berkeley Lab's Innovation &
Partnerships Office at  IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department of
Energy and the U.S. Government consequently retains certain rights.  As such,
the U.S. Government has been granted for itself and others acting on its
behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
to reproduce, distribute copies to the public, prepare derivative works, and
perform publicly and display publicly, and to permit other to do so.
