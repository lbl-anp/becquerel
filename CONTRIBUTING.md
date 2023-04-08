# Contribution guidelines

Contributions to `becquerel` are welcome and encouraged, whether it is
reporting bugs, requesting features, or contributing code.
Please follow these guidelines when contributing to this project.

## Developer Instructions

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python setup.py develop

pip install pre-commit
pre-commit install
```

(It is more convenient to use `develop` so that the code is soft-linked
from the installation directory, and the installed package will always use
the current version of code.)

We use [`pre-commit`](https://pre-commit.com/) to automatically run black.

### Running the tests

(Requires `requirements-dev.txt` to be installed)
To run the tests using `pytest`, from the root directory of the repo:

```bash
pytest
```

(`python setup.py test` is still supported also.)
By default, a code coverage report is printed to the terminal.
Tests marked `webtest` or `plottest` are by default skipped for the sake of
speed. To run all tests, clear the pre-configured markers option:

```bash
pytest -m ""
```

To produce an HTML code coverage report in directory `htmlcov`
with line-by-line highlighting:

```bash
pytest --cov-report html:htmlcov
```

## Code Style Guide

Use [google standards](https://google.github.io/styleguide/pyguide.html)

## Linter

The code style/formatting will be checked with
[`black`](https://black.readthedocs.io/en/stable/) and
[`flake8`](https://flake8.pycqa.org/en/latest/) in the CI

## Checklist for code contributions

- [ ] Branch off of `main`
- [ ] Develop the feature or fix
- [ ] Write tests to cover all use cases
- [ ] Ensure all tests pass (`pytest`)
- [ ] Ensure test coverage is >95%
- [ ] Autoformat (`black .`)
- [ ] Spellcheck your code and docstrings
- [ ] Push branch to GitHub and create a pull request to `main`
