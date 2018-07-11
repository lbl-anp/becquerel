## Release checklist

- [ ] Branch off of `develop` and name the branch `version-X.X.X`
- [ ] Update version in `setup.py`
- [ ] Update version in the copyright notice in `README`
- [ ] Update version in the `LICENSE`
- [ ] Update classifiers in `setup.py`
- [ ] Update `HISTORY` with a list of changes for this version
- [ ] Verify that all tests pass (`python setup.py test`)
- [ ] Commit the changes, push to GitHub, and start a pull request
- [ ] After PR is accepted, create tagged version on GitHub
- [ ] Upload new version to PyPI
