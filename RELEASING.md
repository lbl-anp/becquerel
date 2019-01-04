## 1. Create Tagged Release

We follow the `git flow` [release process](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

- If normal release:
  - [ ] Branch off of `develop` and name the branch `release-X.X.X`
- If hotfix/patch:
  - [ ] Branch off of `master` and name the branch `hotfix-X`
- [ ] Update version number within the repository
  - in `setup.py`
  - in Copyright Notice in `README`
  - in the `LICENSE`
- [ ] Update classifiers in `setup.py`
- [ ] Verify that all tests pass (`python setup.py test`)
- [ ] Commit the changes, push to GitHub, and start a pull request into `master`
- [ ] Approve PR and merge it into master, but do not delete it yet
- [ ] Create tagged version (`X.X.X`) on GitHub
- [ ] Add release notes to the tag on GitHub with a list of changes

## 2. Distribution Creation/Upload

- [ ] Create distribution
  ```bash
  git pull
  git checkout X.X.X
  rm dist/*
  python3 -m pip install --user --upgrade setuptools wheel
  python3 setup.py sdist bdist_wheel --universal
  ```
- [ ] Test distribution
  ```bash
  python3 -m pip install dist/becquerel-X.X.X-py2.py3-none-any.whl
  python3 -m pip install dist/becquerel-X.X.X.tar.gz
  ```
- [ ] Upload new version to PyPI
  ```bash
  python3 -m pip install --user --upgrade twine
  python3 -m twine upload dist/*
  ```
- [ ] Test new version installs from PyPI
  ```bash
  cd ..
  python3 -m pip install becquerel
  python3 -c "import becquerel; print(becquerel.__version__)"
  ```

## 3. Cleanup

- [ ] Create PR from the `release-X.X.X`/`hotfix-X` branch into `develop`
- [ ] After PR is accepted, delete the `release-X.X.X`/`hotfix-X` branch
