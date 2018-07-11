#!/usr/bin/env python
"""Becquerel: Tools for radiation spectral analysis."""

from __future__ import print_function
from setuptools import setup, find_packages


MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)

DESCRIPTION = __doc__.split('\n')[0]
URL = 'https://github.com/lbl-anp/becquerel'
MAINTAINER = "The Becquerel Development Team"
EMAIL = "becquerel-dev@lbl.gov"

# classifiers from list at https://pypi.org/classifiers/
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: Other/Proprietary License
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
"""

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

with open('LICENSE.txt', 'r') as fh:
    LICENSE = fh.read()

with open('requirements.txt', 'r') as fh:
    REQUIREMENTS = fh.read()


setup(
    name='becquerel',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url=URL,
    download_url=URL + '/releases',
    maintainer=MAINTAINER,
    maintainer_email=EMAIL,
    license=LICENSE,
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    packages=find_packages(),
    python_requires='>=2.6',
    install_requires=[_f for _f in REQUIREMENTS.split('\n') if _f],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)
