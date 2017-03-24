# becquerel

## Installation instructions

To install the package:

```
pip install -r requirements.txt
python setup.py install
```

Before reinstalling, be sure to either remove the ```build``` directory
or run:

```
python setup.py clean --all
```

For development purposes, it is more convenient to install using
```
python setup.py develop
```
so that the code is soft-linked from the installation directory,
and the installed package will always use the current version of code.

## Running the tests

To run the tests using `pytest`:

```
pytest
```

(`python setup.py test` is still supported also.)
By default, a code coverage report is printed to the terminal.
Tests marked `webtest` or `plottest` are by default skipped for the sake of
speed. To run all tests, clear the pre-configured markers option via

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
