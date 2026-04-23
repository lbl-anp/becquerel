# Contribution guidelines

Contributions to `becquerel` are welcome and encouraged, whether it is
reporting bugs, requesting features, or contributing code.
Please follow these guidelines when contributing to this project.

## Developer Instructions

```bash
pip install -e ".[dev]"
# or: uv sync --extra dev

pre-commit install
```

(Editable installs keep the source tree linked into the environment, so the
installed package always uses the current checkout.)

## Linting, formatting, and other checks

We use [`pre-commit`](https://pre-commit.com/) to automatically run various
checks and formatting tools in the CI.
If `pre-commit` is installed, it will automatically run when committing
new code, and it can also be run at any time using the following command:

```bash
pre-commit run --all
```

or run on any files not yet committed to the repository using

```bash
pre-commit run --files <filename1> <filename2> ...
```

### Running the tests

(Requires the `dev` extra to be installed.)
To run the tests using `pytest`, from the root directory of the repo:

```bash
pytest
```

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

Use [google standards](https://google.github.io/styleguide/pyguide.html).

## Checklist for code contributions

- [ ] Branch off of `main`
- [ ] Develop the feature or fix
- [ ] Write tests to cover all use cases
- [ ] Ensure all checks pass (`pre-commit`)
- [ ] Ensure all tests pass (`pytest`)
- [ ] Ensure test coverage is >95%
- [ ] Push branch to GitHub and create a pull request to `main`
