"""Helpers for testing."""
from pathlib import Path
import requests

TESTS_PATH = Path(__file__).parent.absolute()
REPO_PATH = TESTS_PATH.parent


def nndc_is_up():
    """Check whether the NNDC databases can be reached."""
    try:
        return (
            requests.post("https://www.nndc.bnl.gov/nudat3/indx_sigma.jsp").status_code
            == requests.codes.ok
        )
    except requests.exceptions.ConnectionError:
        return False


def xcom_is_up():
    """Check whether the NIST XCOM databases can be reached."""
    try:
        return (
            requests.post(
                "https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html"
            ).status_code
            == requests.codes.ok
        )
    except requests.exceptions.ConnectionError:
        return False
