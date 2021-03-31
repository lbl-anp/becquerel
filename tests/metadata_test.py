"""Ensure the metadata is updated and valid."""
from utils import REPO_PATH
import becquerel as bq

COPYRIGHT = (REPO_PATH / "COPYRIGHT.txt").read_text().strip("\n")


def test_copyright_in_license():
    license = (REPO_PATH / "LICENSE.txt").read_text()
    copyright_in_license = (
        license.split("*** Copyright Notice ***")[-1]
        .split("*** License Agreement ***")[0]
        .strip("\n")
    )
    assert COPYRIGHT == copyright_in_license


def test_copyright_in_readme():
    readme = (REPO_PATH / "README.md").read_text().strip("\n")
    assert readme.endswith(COPYRIGHT)


def test_copyright_in_metadata():
    assert bq.__metadata__.__copyright__.strip("\n") == COPYRIGHT
