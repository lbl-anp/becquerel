#!/usr/bin/env python
"""Becquerel: Tools for radiation spectral analysis."""

import sys
import site
from setuptools import setup, find_packages

# Enables --editable install with --user
# https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

NAME = "becquerel"

MAJOR = 0
MINOR = 3
MICRO = 0
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"

DESCRIPTION = __doc__.split("\n")[0].split(": ")[-1]
URL = "https://github.com/lbl-anp/becquerel"
MAINTAINER = "The Becquerel Development Team"
EMAIL = "becquerel-dev@lbl.gov"

# classifiers from list at https://pypi.org/classifiers/
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: Other/Proprietary License
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3 :: Only
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
"""

with open("README.md") as fh:
    README = fh.read()
# remove package title from description
README = "\n".join(README.split("\n")[2:])

with open("CONTRIBUTING.md") as fh:
    CONTRIBUTING = fh.read()

with open("LICENSE.txt") as fh:
    LICENSE = fh.read()

with open("requirements.txt") as fh:
    REQUIREMENTS = fh.read()

# make long description from README and CONTRIBUTING
# but move copyright notice to the end
LONG_DESCRIPTION, COPYRIGHT = README.split("## Copyright Notice")
LONG_DESCRIPTION += "\n" + CONTRIBUTING
LONG_DESCRIPTION += "\n" + "## Copyright Notice" + COPYRIGHT

# write metadata to a file that will be imported by becquerel
with open("becquerel/__metadata__.py", "w") as f:
    print('"""Becquerel package metadata."""', file=f)
    print("", file=f)
    print(f'__description__ = "{DESCRIPTION}"', file=f)
    print(f'__url__ = "{URL}"', file=f)
    print(f'__version__ = "{VERSION}"', file=f)
    print(f'__license__ = """{LICENSE}"""', file=f)
    print(f'__copyright__ = """{COPYRIGHT}"""', file=f)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=URL + "/releases",
    maintainer=MAINTAINER,
    maintainer_email=EMAIL,
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    platforms="any",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[_f for _f in REQUIREMENTS.split("\n") if _f],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    license="Other/Proprietary License (see LICENSE.txt)",
)
