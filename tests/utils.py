"""Helpers for testing."""

from pathlib import Path

import requests

TESTS_PATH = Path(__file__).parent.absolute()
REPO_PATH = TESTS_PATH.parent


def database_is_up(url):
    """Check whether an online database can be reached.

    Parameters
    ----------
    url : str
        URL to query.
    """
    try:
        return requests.post(url).status_code == requests.codes.ok
    except requests.exceptions.ConnectionError:
        return False


def nndc_is_up():
    """Check whether the NNDC databases can be reached."""
    return database_is_up("https://www.nndc.bnl.gov/nudat3/indx_sigma.jsp")


def xcom_is_up():
    """Check whether the NIST XCOM databases can be reached."""
    return database_is_up("https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html")
