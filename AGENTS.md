# AUTO-OSU contributor instructions

- Keep explanations simple and direct. Report actual problems and verification limits plainly.
- Read `CONTRIBUTING.md` and `docs/TASKS.md` before making changes. Work on a focused branch, never directly on `master`. Maintainer work uses `kanzei/<topic>`.
- Preserve unrelated local work, generated artifacts, songs, environments, and model checkpoints. Stage exact paths for commits.
- Follow task priority and dependencies in `docs/TASKS.md`. Plans have no delivery dates or duration promises. Update task state with evidence when completing work.
- Keep the Chinese and English README and interface text consistent. Planned features must remain distinct from implemented behavior.
- Run `python scripts/check_repository.py` for documentation and repository changes. For code changes run relevant tests, then `python -m pytest -q` before the PR is ready.
- For model or generation changes, also check real outputs with frozen settings. Distinguish unit tests, actual inference, independent parsing, GUI checks, and osu! client playtesting.
- Source labels and manifests are declarations. A local-record match is scoped to that local record store. Statistical non-detection is inconclusive, and model hashes do not prove execution.
- Do not publish a statistical detector without an independent low-false-positive evaluation. Keep model versions, song-grouped splits, thresholds, false positives, recall, and abstentions in the evidence.
- Keep model weights, raw songs, training datasets, and generated maps out of Git. Track reproducible recipes and compact reports instead.
- Open PRs against `master`, wait for required checks, review the final diff, and use squash merging. Do not bypass branch protection to complete a task.

