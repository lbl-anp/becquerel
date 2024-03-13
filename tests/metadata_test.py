"""Ensure the metadata is updated and valid."""

from utils import REPO_PATH

import becquerel as bq

COPYRIGHT = (REPO_PATH / "COPYRIGHT.txt").read_text()
LICENSE = (REPO_PATH / "LICENSE.txt").read_text()
README = (REPO_PATH / "README.md").read_text()


class TestCopyright:
    def test_license(self):
        # The license only contains the first paragraph from the copyright
        assert COPYRIGHT.split("\n\n")[0] == LICENSE.split("\n\n")[1]

    def test_readme(self):
        assert README.endswith(COPYRIGHT)

    def test_metadata(self):
        assert bq.__metadata__.__copyright__ == COPYRIGHT
