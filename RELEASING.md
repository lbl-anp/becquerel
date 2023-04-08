# Create Tagged Release

We (loosely) follow the `git flow` [release process](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

- [ ] Pull the most recent version of `main`
- [ ] Branch off of `main` and name the branch `release-X.X.X` or `hotfix-X` or `vX.X.X`
- [ ] Update version number within the repository with `bump2version`
- [ ] Commit the changes, push to GitHub, and start a pull request into `main`
- [ ] Once PR approved, merge it into `main`, and delete release branch.
- [ ] Create tagged version (`vX.X.X`) on
      [GitHub](https://github.com/lbl-anp/becquerel/releases/new) pointing to
      the merge commit to `main`
- [ ] Add release notes to the tag on GitHub with a list of changes

Once the release is submitted and `main` is tagged, github actions will
automatically deploy to `pypi`.
