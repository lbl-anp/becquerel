## Contribution guidelines

Contributions to `becquerel` are welcome and encouraged, whether it is
reporting bugs, requesting features, or contributing code.
Please follow these guidelines when contributing to this project.

### Developer Instructions

```
pip install -r requirements.txt
pip install -r requirements-dev.txt
python setup.py develop
```

(It is more convenient to use `develop` so that the code is soft-linked
from the installation directory, and the installed package will always use
the current version of code.)

### Running the tests

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

### Code Style Guide

Use [google standards](https://google.github.io/styleguide/pyguide.html)

### Linter

Use the linter of your choice.
The code style will be checked with [`black`](https://black.readthedocs.io/en/stable/) in the testing.
We like to use [`flake8`](https://flake8.pycqa.org/en/latest/) and/or [`pylance`](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance).

### Checklist for code contributions:
  - [ ] Branch off of `main` and name the branch `feature-XX` or `issue-XX`
  - [ ] Develop the feature or fix
  - [ ] Write tests to cover all use cases
  - [ ] Ensure all tests pass (`python setup.py test`)
  - [ ] Ensure test coverage is >95%
  - [ ] Autoformat (`black .`)
  - [ ] Spellcheck your code and docstrings
  - [ ] Push branch to GitHub and create a pull request to `main`
