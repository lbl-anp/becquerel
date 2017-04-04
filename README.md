# becquerel

## Installation instructions

### As a user

```
pip install -r requirements.txt
python setup.py install
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

## Code Style Guide

Use [google standards](https://google.github.io/styleguide/pyguide.html)

## Linter

* Use `flake8` in your IDE
* Use `pylint` from command line (as in style guide)
