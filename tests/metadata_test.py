"""Ensure the metadata is updated and valid."""
from utils import REPO_PATH
import becquerel as bq

COPYRIGHT = (REPO_PATH / "COPYRIGHT.txt").read_text()
LICENSE = (REPO_PATH / "LICENSE.txt").read_text()
README = (REPO_PATH / "README.md").read_text()
COPYRIGHT_IN_LICENSE = (
    LICENSE.split("*** Copyright Notice ***")[-1]
    .split("*** License Agreement ***")[0]
    .strip()
    + "\n"
)


class TestCopyright:
    def test_license(self):
        assert COPYRIGHT == COPYRIGHT_IN_LICENSE

    def test_readme(self):
        assert README.endswith(COPYRIGHT)

    def test_metadata(self):
        assert bq.__metadata__.__copyright__ == COPYRIGHT


class TestVersion:
    def test_copyright(self):
        assert (
            COPYRIGHT.split("Becquerel v. ")[1].split(",")[0]
            == bq.__metadata__.__version__
        )

    def test_license(self):
        assert (
            LICENSE.split("Becquerel v. ")[1].split(",")[0]
            == bq.__metadata__.__version__
        )

    def test_readme(self):
        assert (
            README.split("## Copyright Notice")[1]
            .split("Becquerel v. ")[1]
            .split(",")[0]
            == bq.__metadata__.__version__
        )
