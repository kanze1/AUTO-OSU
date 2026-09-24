# Repository workflow

The main branch is `master`. Changes arrive through pull requests and are squash merged after validation. Repository settings enable squash merging, disable merge/rebase merging, and delete merged remote work branches.

## Main-branch protection

The protection configuration is recorded in [.github/branch-protection.json](../.github/branch-protection.json) and has been applied to the GitHub repository:

- Require a PR and an up-to-date branch, including for administrators.
- Require `repository-checks`, `pytest (ubuntu-latest, 3.12)`, and `pytest (windows-latest, 3.12)` from GitHub Actions.
- Resolve review conversations before merging.
- Keep linear history; block force pushes and branch deletion.
- Route review through CODEOWNERS. There is no mandatory second-person approval while the project has one maintainer, so the maintainer can merge their own PR after checks pass.

Maintainers can inspect the current remote configuration with:

```bash
gh api repos/kanze1/AUTO-OSU/branches/master/protection
gh pr checks <number>
```

When required check names change, update the saved configuration and remote protection together after verifying the new checks on a PR. Do not bypass failing checks.

## Evidence and task tracking

[CONTRIBUTING.md](../CONTRIBUTING.md) defines contribution terms and validation. [docs/TASKS.md](TASKS.md) records implementation state without delivery dates. Use the issue forms for reproducible failures or coordinated proposals; use the PR template to record final behavior and actual validation.

Application unit tests do not establish model quality or osu! client behavior. Record actual inference, independent parsing, client round-trip checks, and human playtesting separately. Model experiments must freeze source/weight versions, song-grouped splits, settings, and acceptance criteria before final evaluation.

