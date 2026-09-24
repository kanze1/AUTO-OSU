# Difficulty measurement and evaluation

The source application measures the exported `.osu`, independently of the star condition supplied to the models. CLI and GUI generation display the measured rating. Folder batches retain it in each item's `diffs` array. Pure rules have no model condition, even when `--star` was supplied.

## Calculator and reports

- Pinned calculator: **rosu-pp-py 4.0.2**, osu!standard, **NM, `lazer=False`**. This is the stable scoring calculation in that pinned library, not a promise to match every past or future osu! client version. The binding depends on rosu-pp 4.0.1 ([upstream dependency](https://github.com/MaxOhn/rosu-pp-py/blob/v4.0.2/Cargo.toml)).
- Measurements use the serialized map, after timing shift, slider fitting and SV overrides. Model conditions remain separate input fields.
- Each `.osz` includes `autoosu-evaluation.json`. A matching `<pack>.evaluation.json` is saved beside it for easy inspection. The schema is `autoosu.evaluation/1`; per-map SHA-256 ties measurements to the corresponding exported bytes.
- Reports retain actual SR, aim, speed, slider factor, maximum combo, and ordered 400 ms aim / aim-without-sliders / speed strains. These strain values are not local star ratings. For v14 maps whose parsed object counts agree, the first section ends at the second object's time rounded up to a 400 ms boundary, following the [pinned strain implementation](https://github.com/MaxOhn/rosu-pp/blob/v4.0.1/src/util/macros.rs). Other formats leave the origin unknown.
- Structural diagnostics include serialized overlap with a 3 ms rounding tolerance, simultaneous starts, head/anchor bounds, invalid slider fields, shortened sliders, quarter-beat circle-run length, peak two-second NPS and spacing quantiles. Bounds do not inspect every point on the rendered slider curve. These are diagnostics, not proof of playability.
- Missing/incompatible calculators and measurement errors produce `stars: null` and a reason, never a fabricated zero or the model condition. Generation still saves its map. The [pinned package](https://pypi.org/project/rosu-pp-py/4.0.2/) requires Python 3.11+; Python 3.10 can generate but cannot use this measurement version.

The provenance manifest includes a compact measurement summary; full strain arrays stay in the separate evaluation report. Neither report authenticates authorship.

## Frozen comparison protocol

Use [scripts/evaluate_difficulty.py](../scripts/evaluate_difficulty.py). Audio, reference maps, outputs and detailed strain reports stay under ignored `out/`; compact evidence can be checked into `docs/`.

```powershell
.venv-gpu/Scripts/python.exe scripts/evaluate_difficulty.py prepare --exclude-corpus out/d1-v0-probe/corpus.json
.venv-gpu/Scripts/python.exe scripts/evaluate_difficulty.py run
.venv-gpu/Scripts/python.exe scripts/evaluate_difficulty.py report
```

The corpus is frozen before generation: eight development songs and 24 reserved songs, balanced between reference BPM below 180 and BPM at least 180. Selection uses deterministic hash order among local standard reference maps, with one 4/4 red line, BPM 100–240 and 60–150 seconds of audio. Exact audio hashes used in the earlier detector experiment are excluded. Different encodings/recordings and overlap with v0's training set remain unverified.

The development sweep holds Insane's settings and both v0 weights fixed. It varies only star condition (5.5, 6, 6.5, 7, 7.5, 8) and timing source (reference or automatic), with seed 240924, temperature 0.9, 12 rhythm steps, 100 coordinate steps and CFG 1. Reference timing adds 26 ms before generation so the exported red line retains the reference offset. Automatic timing uses the original estimator. The recipe records every application Python file's hash, base commit, model hashes and corpus hash; a changed recipe cannot silently resume previous results.

The `report` stage independently opens every exported map with the pinned parser/calculator and checks its measured rating again. Results are developer diagnostics. Human highlight labels, blind listening/playtesting and client import are separate work. Reserved songs must stay unused for control tuning; use `--partition heldout` only after the candidate and evaluation criteria are frozen.

The optional `curves` stage requires `slider` (this run used 0.8.2). It independently samples the exported paths of sliders flagged for out-of-field anchors, at 1,001 positions per curve. A Bezier control point outside the field does not by itself put the curve outside the field.

## First development screen

Evidence: [frozen 32-song corpus](difficulty-corpus.json), [96-output report with source/model identities and per-map results](difficulty-screen.json), and [independent curve checks](difficulty-curves.json). All 24 reserved songs remain unused. This screen does not verify training-set exclusion or generalize success rates to all music.

| Model condition | Reference timing: median measured stars | Automatic timing: median measured stars | Within ±0.5 stars, reference / automatic |
| --- | --- | --- | --- |
| 5.5 | 4.74 | 4.73 | 3/8 / 1/8 |
| 6.0 | 5.21 | 5.08 | 1/8 / 1/8 |
| 6.5 | 5.53 | 5.45 | 1/8 / 1/8 |
| 7.0 | 5.74 | 5.82 | 1/8 / 1/8 |
| 7.5 | 5.93 | 5.76 | 1/8 / 1/8 |
| 8.0 | 6.17 | 6.06 | 1/8 / 1/8 |

Six of eight songs with reference timing and five of eight with automatic timing have at least one decrease in measured SR as the condition increases. All eight automatic BPM estimates agree with the reference BPM to rounding, but offsets/bar phases differ; this does not establish accurate musical alignment. Higher conditions alone are insufficient to meet the proposed quality gate on these development songs. The next controlled experiments should separate density and spacing; [the implementation tasks](generation-control-tasks.md) define those comparisons and the highlight ablations.

All 96 exported maps matched an independent parse and recalculation. No simultaneous/reversed starts, overlaps beyond the specified tolerance, out-of-field heads or invalid slider fields were recorded. Fifteen maps contained 19 sliders with out-of-field control points; the second parser sampled all 19 paths and found no out-of-field samples. Sampling is not an exact-extrema proof or a client check. Source generation took 729.5 seconds in total on the local CUDA environment, with other validation work running; this is an execution record, not an isolated performance benchmark.

## Software verification

- 114 tests passed locally, including exact exported-byte/report binding, measurement failure, missing/wrong calculator, report-write failure, and strain-origin translation.
- A separate `python -I -m autoosu.worker` process on CUDA generated an Insane map with condition 6.5 and measured 4.9213 stars. A two-song worker batch measured 3.9690 and 3.6532 stars for condition 4.5. Worker JSON, batch reports, embedded reports and local provenance checks agreed.
- The actual Chinese and English GUI result handlers displayed those worker results; [Chinese screenshot](images/difficulty-zh.png), [English screenshot](images/difficulty-en.png). This checked display of real worker data, not a full click-through generation or GPU-install flow.
- A minimal PyInstaller executable successfully loaded the pinned calculator and its packaged version metadata and measured an exported map. The full application release package was not built or published.
- No osu! client import, editor round-trip, blind listening or gameplay validation was performed.

## Acceptance boundary

Measurement infrastructure and a development sweep do not complete high-difficulty quality validation. Q2 must establish response to density and spacing controls; H1 needs explicit segment ablations and human labels. The target of at least 80% within ±0.5 stars, without worse structure or playability, is still an unpassed gate. Do not add a guaranteed 7–8 star preset solely because a higher condition can be entered.
