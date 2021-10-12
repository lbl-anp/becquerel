#!/usr/bin/env python
"""Becquerel: Tools for radiation spectral analysis."""

import sys
import site
from setuptools import setup, find_packages
import importlib.util

# Enables --editable install with --user
# https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

_spec = importlib.util.spec_from_file_location(
    "__metadata__", "./becquerel/__metadata__.py"
)
METADATA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(METADATA)

# Enables --editable install with --user
# https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# remove package title from description
with open("README.md") as fh:
    README = "\n".join(fh.readlines()[2:])

with open("CONTRIBUTING.md") as fh:
    CONTRIBUTING = fh.read()

with open("requirements.txt") as fh:
    REQUIREMENTS = [_line for _line in fh.readlines() if _line]

with open("requirements-dev.txt") as fh:
    REQUIREMENTS_DEV = [l.strip() for l in fh.readlines() if not l.startswith("-r")]

# make long description from README and CONTRIBUTING
# but move copyright notice to the end
LONG_DESCRIPTION = "{0}\n{2}\n## Copyright Notice\n{1}".format(
    *README.split("## Copyright Notice"), CONTRIBUTING
)

setup(
    name=METADATA.__name__,
    version=METADATA.__version__,
    description=METADATA.__description__,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=METADATA.__url__,
    download_url=f"{METADATA.__url__}/releases",
    maintainer=METADATA.__maintainer__,
    maintainer_email=METADATA.__email__,
    classifiers=METADATA.__classifiers__,
    platforms="any",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=REQUIREMENTS,
    setup_requires=["pytest-runner"],
    license=METADATA.__license__,
    tests_require=REQUIREMENTS_DEV,
)
