# First v0 detection screen

**Decision: do not ship either statistical baseline.** They expose signals in untouched output, but cannot reliably identify execution of our models. A small edit defeats both baselines, and pure rules output is frequently attributed to the models. Continue with explicit provenance in the app; further statistical work needs a substantially stronger corpus and edited-output evaluation before integration.

## Frozen experiment

- Generator: `8f508921534bef39c4f7beb54201a09fb7db8aca`, copied with `git archive` before changing generation code; official rhythm and coordinate v0 SHA-256 values are in [the report](detection-v0-screen.json).
- 40 audio identities from the existing local `project-riz/osu-beatmaps` shards; 60–150 seconds, eligible historical standard maps between 3 and 7 stars. Generate four modes per song: both models, rhythm only, coordinates only and pure rules, for **160 outputs**. Fixed Insane/5.5 condition, seed 240924, automatic timing, 12 rhythm decoding rounds, 100 coordinate steps, temperature 0.9, CFG 1.0, RTX 4090.
- One historical reference per audio hash, 1,871 total, filtered to maps last updated by 2020. Ranked/approved/loved status and age do **not** individually verify human authorship. Do not present this corpus as audited human ground truth.
- Split by deterministic audio hash: reference counts 1,144 training / 347 calibration / 380 test; positive-song counts 20 / 10 / 10. Neither same-audio difficulties nor generated variants cross splits. 163 of 310 test reference mappers also occur in training; alternate recordings of the same composition are not deduplicated. Generator-training overlap is possible.
- 21 content features covering rhythm intervals, beat-grid alignment, sliders, combos, spacing, angles and density variation. Metadata, author, preset difficulty values, filenames and formatting are excluded from features. Fit logistic regression and a random forest using dual-model training outputs and historical references. Choose each threshold immediately above the highest calibration-reference score, then report test results without lowering it.

## Results

| Test input | Count | Logistic flagged | Forest flagged |
| --- | --- | --- | --- |
| Both models | 10 songs | 3 / 10 | 10 / 10 |
| Rhythm only | 10 songs | 6 / 10 | 9 / 10 |
| Coordinates only | 10 songs | 0 / 10 | 8 / 10 |
| Pure rules, no trained models | 10 songs | **3 / 10** | **7 / 10** |
| Historical reference maps | 380 audio identities | 0 / 380 | 0 / 380 |
| Mapperatorinator comparison artifacts | 3 maps of 1 song | 0 / 3 | 0 / 3 |
| Both models with every tenth object's start moved (+7, −5 px, +3 ms) | 10 songs | **0 / 10** | **0 / 10** |
| Middle half of dual-model objects | 10 songs | 5 / 10 | 9 / 10 |

No input was excluded for parsing/length in this run. Threshold non-detections remain inconclusive. The edit and crop probes are synthetic stresses, not actual editor resaves or mixed human/AI maps. The crop probe does not establish calibrated fragment scanning.

Zero flagged references among 380 audio groups has a one-sided 95% binomial upper bound of about **0.785%**, under an independence assumption; this is not enough to substantiate a 0.1% false-positive target. Three external maps of one song establish no reliable generator-specificity rate. The forest's 100% unedited recall is just ten songs and is contradicted by the edited-output and rules-attribution failures.

## Reproduction

The corpus and generated assets stay in ignored `out/d1-v0-probe`; they are not redistributed. Use the existing eight local shards and official weights. The script requires NumPy, scikit-learn, and the project's CUDA generation environment; scikit-learn is a research dependency, not an app dependency.

```powershell
New-Item -ItemType Directory -Force out/d1-v0-probe/frozen
git archive --format=tar --output=out/d1-v0-probe/frozen.tar 8f508921534bef39c4f7beb54201a09fb7db8aca autoosu
tar -xf out/d1-v0-probe/frozen.tar -C out/d1-v0-probe/frozen
.venv-gpu/Scripts/python.exe scripts/probe_v0_detection.py prepare
.venv-gpu/Scripts/python.exe scripts/probe_v0_detection.py run
.venv-gpu/Scripts/python.exe scripts/probe_v0_detection.py evaluate
```

The local `corpus.json`, `recipe.json`, `generated.json` and `report.json` retain per-file hashes, partitions and failed examples. Reuse this set for development only now that its results have been inspected. A future detector requires fresh verified references, disjoint low-FPR calibration/test songs, varied presets/seeds/model versions, real resaving, edited and mixed maps, and multiple independent songs from other generators. Public app integration remains gated.
