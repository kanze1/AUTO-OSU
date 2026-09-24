# Implementation queue

Tasks follow priority and dependencies, without time estimates. This file records implementation state; [the analysis](roadmap-2026-09-24.md) explains the design. A task is complete only when its acceptance evidence is linked here. Research may conclude that a proposed capability does not meet its release gate.

| ID | Priority | Task | Dependencies | State | Acceptance evidence |
| --- | --- | --- | --- | --- | --- |
| GOV | P0 | Collaboration rules, templates, CI and branch protection | — | Merged | [Workflow and applied protection](governance.md); all 3 CI checks passed; [PR #1](https://github.com/kanze1/AUTO-OSU/pull/1) merged as `3f67706` |
| D1 | P0 | Feasibility of detecting unlabelled v0 outputs | GOV | Validated | [40-song / 160-output screen](detection-v0-screen.md); stop app integration of these baselines: fragile under edits and confuse rules with models |
| D2 | P0 | Generation provenance and source checker | GOV | Merged | [PR #2](https://github.com/kanze1/AUTO-OSU/pull/2), `ddb1728`, all 3 checks passed; [CLI/GUI and actual worker validation](source-check.md); client round-trip still unverified |
| W1 | P0 | Default content watermark for newly generated maps | D2 | Merged | [PR #4](https://github.com/kanze1/AUTO-OSU/pull/4), `2d2e0bd`, all 3 checks passed; [frozen evaluation](generation-watermark.md): 0/1504 eligible unmarked controls detected; actual client round-trip and playtesting pending |
| Q1 | P1 | Measured difficulty and fixed evaluation | D2 | Merged | [PR #3](https://github.com/kanze1/AUTO-OSU/pull/3); [initial measurement evidence](difficulty-evaluation.md); reserved-song control evaluation is now recorded under Q2; human labels/playtests remain separate |
| Q2 | P1 | Higher-difficulty density and spacing control | Q1 | In progress | Q2.1–Q2.4 implemented in 0.3.0.dev3; [frozen comparisons and local acceptance app](generation-controls.md); Q2.5 still requires human gameplay acceptance |
| H1 | P1 | Manual and automatic highlight control | Q1, Q2 | In progress | H1.1–H1.4 prototype and four-way comparisons implemented in 0.3.0.dev3; [controls and evidence](generation-controls.md); human location labels and musicality acceptance remain; H1.5 is gated on those results |
| D3 | — | Statistical attribution of old unmarked maps | — | Out of scope | Maintainer decision: old maps are not pursued. D1 remains an archived experiment; W1 marks new outputs explicitly |
| M1 | P2 | Training-data and representation improvements | Q1, Q2, H1 | Research gate | Full-data coverage and quantization audit; controlled improvement before replacing released weights |

States: Planned, In progress, Validated, Merged, Research gate, or Out of scope. Record missing external verification as a specific limitation, not a passing result.

## Release gates

- Shared pipeline changes preserve standard generation, folder batches, and managed-runtime behavior.
- Engine labels distinguish rules, one-model hybrids, and both-model generation; custom weights retain their actual identity.
- The source checker separates public content markers, file declarations and matching local records. It never authenticates model execution or labels non-detection as human authorship. Old-map statistical attribution is out of scope.
- Model quality is tested on held-out songs with reference and automatic timing separated. Developer-tuned samples cannot become the final test set.
- Actual osu! editor round-trip and gameplay checks are recorded separately from parser tests. Do not claim client testing when only parsing was checked.
