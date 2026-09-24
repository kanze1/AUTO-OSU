# Implementation queue

Tasks follow priority and dependencies, without time estimates. This file records implementation state; [the analysis](roadmap-2026-09-24.md) explains the design. A task is complete only when its acceptance evidence is linked here. Research may conclude that a proposed capability does not meet its release gate.

| ID | Priority | Task | Dependencies | State | Acceptance evidence |
| --- | --- | --- | --- | --- | --- |
| GOV | P0 | Collaboration rules, templates, CI and branch protection | — | Merged | [Workflow and applied protection](governance.md); all 3 CI checks passed; [PR #1](https://github.com/kanze1/AUTO-OSU/pull/1) merged as `3f67706` |
| D1 | P0 | Feasibility of detecting unlabelled v0 outputs | GOV | Validated | [40-song / 160-output screen](detection-v0-screen.md); stop app integration of these baselines: fragile under edits and confuse rules with models |
| D2 | P0 | Generation provenance and source checker | GOV | Validated | [Implementation and validation](source-check.md): CLI/GUI, records, 8 real worker outputs; client round-trip still unverified |
| Q1 | P1 | Measured difficulty and fixed evaluation | D2 | Planned | Pinned calculator, NM stable/lazer scope, actual SR and strain, frozen comparison settings |
| Q2 | P1 | Higher-difficulty density and spacing control | Q1 | Planned | Condition-response comparisons, valid outputs, documented target misses and playtest limits |
| H1 | P1 | Manual and automatic highlight control | Q1, Q2 | Planned | Manual/automatic ablations; density, spacing, kiai and transitions inspected |
| D3 | P1 | Statistical detector in the application | D1, D2 | Research gate | [First screen fails the gate](detection-v0-screen.md); remains unavailable, needs fresh verified human/edited/other-generator corpus |
| M1 | P2 | Training-data and representation improvements | Q1, Q2, H1 | Research gate | Full-data coverage and quantization audit; controlled improvement before replacing released weights |

States: Planned, In progress, Validated, Merged, or Research gate. Record missing external verification as a specific limitation, not a passing result.

## Release gates

- Shared pipeline changes preserve standard generation, folder batches, and managed-runtime behavior.
- Engine labels distinguish rules, one-model hybrids, and both-model generation; custom weights retain their actual identity.
- The source checker separates file declarations, matching local records, and calibrated statistical evidence. It never labels non-detection as human authorship.
- Model quality is tested on held-out songs with reference and automatic timing separated. Developer-tuned samples cannot become the final test set.
- Actual osu! editor round-trip and gameplay checks are recorded separately from parser tests. Do not claim client testing when only parsing was checked.
