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

## Running the tests

To run the tests using `nose`:

```
python setup.py test
```

To run the tests using `nose` with a code coverage report (text report
is printed out stdout and a detailed HTML report is written to the
`htmlcov` directory):

```
python setup.py nosetests
```

## Code Style Guide

Use [google standards](https://google.github.io/styleguide/pyguide.html)

## Linter

* Use `flake8` in your IDE
* Use `pylint` from command line (as in style guide)
* Make a `pylintrc` file in the repo root
