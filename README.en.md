# AUTO-OSU

[![release](https://img.shields.io/github/v/release/kanze1/AUTO-OSU?label=download&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/releases)
[![stars](https://img.shields.io/github/stars/kanze1/AUTO-OSU?style=flat&color=e6a93c)](https://github.com/kanze1/AUTO-OSU/stargazers)
[![tests](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml/badge.svg)](https://github.com/kanze1/AUTO-OSU/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/license-MIT%20%2B%20attribution-4fb8ff)](LICENSE)

**Drop in a song, get a playable osu!standard beatmap a minute later.**

[中文](README.md) · [Download](https://github.com/kanze1/AUTO-OSU/releases) · [How to use](#how-to-use) · [Quality and limits](#quality-and-limits) · [Development schedule](#development-schedule) · [Contribute](#branch-management) · [Community](#community-and-feedback)

**QQ community group: 1124526648** — search for the group number in QQ to discuss feedback, beatmaps, model experiments, and contributions.

![AUTO-OSU main window](docs/screenshot_zh.png)

<details>
<summary>Light theme / batch results</summary>

![light theme](docs/screenshot_en.png)

![batch results](docs/screenshot_busy.png)

</details>

## Why this exists

I am bad at osu! and I love playing it. The worst part: the songs I want to play have no maps, and I can't map.
So: AUTO-OSU. Drop in the song you like, and a minute later you can play it.

The published app is **0.2.0**; the source development version is **0.3.0.dev0**, still using **v0** rhythm and coordinate models. It already makes maps I am happy to play all the way through: the rhythm sits on the drums,
the jumps and streams are learned from over a hundred thousand ranked / approved / loved maps, and all four difficulties come out in one go.
It will keep getting better: mapper intent, deliberate highlights, longer sliders and multiple red lines for tempo changes
are all on the roadmap.

If you are also someone who "just wants to play that one song", take it, open issues, improve it with me.
If it helps you, a ⭐ **star** means a lot. — kanzei

This project is for local play, practice, and generation experiments. When sharing a map, disclose the use of AUTO-OSU, preserve its generation information, and do not present it as handmade.
The current osu! [Ranked AI policy](https://osu.ppy.sh/wiki/en/Ranking_criteria#ai-policy) requires hit objects, hitsounds, and timing to be created by direct human input.
Do not submit maps generated with this tool for Ranked; manual review or AI attribution does not change that requirement.

## How to use

### No install (Windows)

1. Download `AUTO-OSU-<version>-win64-cpu.zip` from [Releases](https://github.com/kanze1/AUTO-OSU/releases) (models included) and unzip it anywhere.
2. Run `AUTO-OSU.exe`.
3. Drag a song onto the window, tick difficulties, click **Generate**.
4. The `.osz` is written to the `output` folder next to the exe and, by default, opened in osu! (it imports itself). Open osu! and it is in the song list.

No GPU needed: a 3-minute song, one difficulty, takes about a minute on a modern CPU (70 s measured on 16 cores; 16 s on an RTX 4090).
Windows 10 / 11, 64-bit.

### Set up GPU acceleration

Click **Set up GPU acceleration** under Device. The app prepares uv, a separate Python 3.12, and a CUDA-enabled PyTorch selected for your driver. After an actual CUDA operation succeeds, the runtime is ready immediately, without restarting the window.

- No existing Python, uv, or CUDA Toolkit installation is needed. An NVIDIA GPU and its driver are required.
- The first download is several GB. Progress and installer logs appear in the window, and setup can be cancelled.
- The runtime lives in `%LOCALAPPDATA%\AUTO-OSU\runtime` and leaves your system Python alone. A failed repair keeps the previous working runtime active.
- `auto` prefers an available GPU; `cpu` always uses CPU; an explicit `cuda` selection reports a clear error when unavailable.

### Generate a whole folder

Select **Folder batch**, browse or drop a directory, optionally enable **Include subfolders**, and click **Generate batch**. Supported audio and video files appear in the processing queue.

Each song has its own output folder, so duplicate filenames cannot overwrite one another. A damaged file is recorded and the next file is processed. A `batch-report-*.json` records outputs, errors, and the device used. **Stop after current** finishes the active song and cancels the remaining queue. Import the generated `.osz` files into osu! when you are ready.

### Which audio works

Input is normalised before it enters the pipeline, so the format hardly matters:

- audio: mp3, ogg, wav, flac, m4a / aac, wma, opus, aiff, ape, alac …
- video: mp4, mkv, webm, mov, avi … — the audio track is extracted automatically
- the audio packed into the `.osz` is always something osu! can play: mp3 and ogg-vorbis are kept as they are; anything else becomes mp3 at a bit rate that follows the source (lossless -> 320 kbps, lossy -> the source rate rounded up, 192 kbps minimum)
- embedded cover art becomes the beatmap background; for a video without cover art a frame is grabbed. Title and artist come from the tags

ffmpeg ships with the program; nothing to install.

### The window

| Area | What it does |
| --- | --- |
| Song | Single song / Folder batch, drag and drop or Browse, with a processing queue. |
| Difficulties | Easy / Normal / Hard / Insane, any combination, all packed into one `.osz`. Default Hard + Insane. |
| Output | Target folder; "Import into osu! when done" opens the `.osz`, same as double-clicking it. |
| Models | Shows whether the models are present; one-click download if not (checksums verified). |
| Device | Actual CUDA availability, GPU and memory, plus Recheck and automatic GPU setup. |
| Cover | Chinese / English, light / dark. Settings, last song and folder are remembered. |

**Advanced options** (click "Advanced options"):

| Option | Meaning |
| --- | --- |
| Seed | Another number gives another layout for the same song; the same seed reproduces the same map. |
| BPM / offset | Blank = detected. Fill in by hand when detection is off (tempo changes, near-empty intros). Offset in ms. |
| Creator name | Written into the `.osu` as Creator, default AUTO-OSU. |
| Star rating | Difficulty hint for the models; blank = per-difficulty default: Easy 2.0 / Normal 3.2 / Hard 4.5 / Insane 5.5. This is not the measured rating of the output; measurement and calibration are on the development schedule. |
| Placement quality | Diffusion steps of the coordinate model: fast 50 / standard 100 / fine 200. Standard is plenty. |
| Engine | "AI models" is the normal mode; "rules only" needs no models and maps in seconds — for comparison or when models are missing. |
| Preview mp3 | Also saves an mp3 with the song turned down and a click on every object, to check the rhythm without opening osu!. |

### Command line

`python -m autoosu SONG [options]`; the exe accepts the same arguments (`AUTO-OSU.exe song.mp3 -d Hard`).

```powershell
python -m autoosu --setup-runtime
python -m autoosu --check-cuda
python -m autoosu "D:\Music" --recursive -d Hard Insane -o "D:\Beatmaps"
```

| Option | Meaning |
| --- | --- |
| `-d Easy Normal Hard Insane` | difficulties to generate |
| `-o DIR` | output folder, default `out` |
| `--seed N` | random seed |
| `--bpm` / `--offset` | manual BPM / red-line offset (ms) |
| `--title` / `--artist` / `--creator` | override metadata (default: audio tags, or an "Artist - Title" file name) |
| `--star X` | star-rating condition |
| `--coord-steps N` | diffusion steps, default 100 |
| `--cfg-scale X` | classifier-free guidance of the coordinate model, default 1.0 |
| `--temperature` / `--density` / `--density-bias` / `--decode-steps` | rhythm-model sampling: temperature, target objects per measure, "no note" bias (negative = denser), decoding rounds |
| `--device auto\|cuda\|cpu` | compute device |
| `--setup-runtime` | install and verify an app-managed GPU runtime with uv |
| `--check-cuda` | check the effective inference runtime, including the managed environment |
| `--check-source PATH` | check declarations and local records in an `.osu`, `.osz`, or folder (source development version) |
| `--source-report report.json` | save the source-check report as JSON |
| `--records-dir DIR` | select the local record store for both generation and checking |
| `--recursive` | include subfolders when the input is a directory |
| `--rules` | rule-based mode |
| `--rhythm-model` / `--coord-model` | explicit model files; otherwise looked up in `models/` |
| `--no-coord-model` | rhythm model only, rule-based placement |
| `--download` | fetch missing models from the GitHub release |
| `--osu-shift 26` | how many ms early objects are written into the `.osu` |
| `--preview` | also write the preview mp3 |
| `--debug-plot` | save an analysis image: loudness and kiai sections, onsets and beat grid, chosen notes per difficulty |
| `--dump-events` | print every object with time, type and beat |

### Check beatmap source (source development version)

The source development version has a **Check source** button at the bottom of the app. Select a beatmap, pack or folder and optionally save a JSON report. From the CLI:

```powershell
python -m autoosu --check-source "map.osz" --source-report "source-report.json"
python -m autoosu --check-source "D:\Beatmaps" --recursive
```

New packs include model identities, settings and content fingerprints, with a separate local generation record. Results distinguish declarations, local-record matches and inconclusive evidence. Declarations can be copied; no record match does not mean human authorship. Statistical detection is not available: the [first v0 screen](docs/detection-v0-screen.md) failed under small edits and confused some rules-only output with model output.

Records default to `~/.autoosu/provenance`; use `--records-dir` or `AUTOOSU_RECORDS` for another store. See [scope and validation](docs/source-check.md). The published 0.2.0 download does not include this feature; the development app requires a matching managed GPU runtime version.

### Python install

```bash
git clone https://github.com/kanze1/AUTO-OSU
cd AUTO-OSU
python -m venv .venv && .venv\Scripts\activate       # Windows; Linux/macOS: source .venv/bin/activate
pip install -e .[gui]
python -m autoosu --setup-runtime                     # optional: prepare the separate GPU runtime
python -m autoosu --download                          # fetch the models once (~320 MB)
python -m autoosu                                     # open the window
python -m autoosu "song.mp3" -d Hard Insane -o out    # command line
```

Python 3.10 or newer. This is also how to run it on macOS / Linux; the exe is Windows only.

## Quality and limits

- **Rhythm model.** The v0 training evaluation recorded a generated-onset F1 of 0.963. Evaluation uses the reference map's timing and true local density, with a default of six sequences of up to 512 ticks each. This does not measure end-to-end quality on arbitrary songs or establish superiority over human mapping.
- **Coordinates and sliders.** The coordinate model generates positions from pure noise, learning jumps, streams, and slider shapes. Sliders are fitted to the playfield and required length before export. Historical test samples had no paths outside the playfield; playability still needs checking on actual songs.
- **What is still missing.** One red line per song; slider length is not a model input yet, so long sliders on fast songs are occasionally shortened (a green line keeps the timing right);
  hitsounds are simple drum-based whistle / clap / finish; no storyboard. Check timing, readability, and playability in the editor before playing or sharing a map.
- **Source identification.** The source development version distinguishes both-model, hybrid and rules output and checks declarations and local records. The older 0.2.0 `ai-generated` tag also occurs on rules-only output, so it cannot establish model use. Statistical detection of unlabelled old output has not passed the integration gate.

## How it works

![architecture](docs/architecture.png)

**Analysis and timing (rules).** Harmonic / percussive split, onset envelopes per drum band (kick / snare / hat); tempogram BPM with octave correction;
1 ms offset refinement on the waveform; downbeats from kicks, chord changes and loudness; loudness sections → kiai.
Objects are written 26 ms before the audio transient, the convention of ranked maps that players calibrate their offset against.

**Rhythm model** `rhythm_v0.pt`, about 29 M parameters. A bidirectional transformer on a 1/4-beat grid: each tick sees ±80 ms of mel spectrogram,
its position in the bar, local loudness, plus the requested star rating / CS / AR / OD / HP, and predicts one of six classes
(none / circle / slider head / body / end / spinner), decoded MaskGIT-style in 12 parallel rounds.

**Coordinate model** `coord_v0.pt`, about 130 M parameters. The DiT-B architecture from [osu-diffusion](https://github.com/OliBomby/osu-diffusion),
trained from scratch here on the full 1000-step noise schedule. Input is the object token sequence (circles, slider heads, anchors, slider ends, spinners, each with its time);
it denoises x / y for every point from pure noise. The only conditions are star rating and CS; half of the training samples had the "distance to the previous point" zeroed, so it chooses its own spacing.

**Slider fitting.** The model draws the shape, the rhythm dictates the length: each slider is scaled about its head to the required length; if it would leave the field it is mirrored, then rotated,
and only as a last resort shortened with a local green line. Fast songs get a lower SliderMultiplier.

### Training record

![training curves](docs/training_curves.png)

| Model | Data | Hardware | Steps | Wall time | Result |
| --- | --- | --- | --- | --- | --- |
| Rhythm, masked v0 | 139,582 osu!standard beatmaps ([project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps)) | 2 × RTX 5880 Ada | 60k, batch 128 | 4.5 h | step 40k used: generated-onset F1 0.963, density error 0.10 |
| Coordinates, DiT-B v0 | 140,018 maps of the same corpus (ORS layout) | 2 × RTX 5880 Ada | 200k, batch 128 | 8.5 h | final loss 0.125; 0 % out of bounds; layouts at 50k / 100k / 200k nearly identical for one seed |

These are historical experiment records. See "Quality and limits" above for the F1 conditions and sampling scope; boundary and layout comparisons refer to tested samples.

An autoregressive rhythm model was trained too; causal attention could not hear the upcoming audio and it liked to hide behind spinners, so the masked version won.
The full log, failed routes included, is in [docs/rhythm_model_design.md](docs/rhythm_model_design.md) (Chinese);
wandb projects: [autoosu-rhythm](https://wandb.ai/kanzei/autoosu-rhythm), [autoosu-coords](https://wandb.ai/kanzei/autoosu-coords).

## Train it yourself / build the exe

Everything used for training is in the repo: `autoosu/ml/prepare_data.py` (HF shards → features and labels), `autoosu/ml/train.py` (rhythm model, `torchrun` multi-GPU),
`coord/` + `scripts/coord_make_ors.py` (coordinate model, accelerate), `scripts/server_*.sh` (server workflow), `scripts/coord_export.py` (export to release files).
Both model files load with `torch.load(weights_only=True)`, no pickled code.

Building the exe:

```powershell
pip install -e .[build]
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1            # dist/AUTO-OSU-<version>-win64-cpu.zip
```

A venv with a CUDA torch plus `-Venv .venv-gpu -Suffix cuda` gives the GPU build. `python scripts/make_icon.py avatar.png` creates the window avatar and exe icon.

## FAQ

**Model download fails?** Download `rhythm_v0.pt` and `coord_v0.pt` from [models-v0](https://github.com/kanze1/AUTO-OSU/releases/tag/models-v0) by hand
and put them into the `models/` folder next to the exe (or `~/.autoosu/models/`).

**Antivirus complains?** PyInstaller bundles are often flagged. Run it the Python way, or build it yourself with the steps above.

**BPM or offset wrong?** Set them in Advanced options. Songs with tempo changes currently get a single red line.

**osu! did not open?** `.osz` is not associated with osu! on your system; drag the generated `.osz` onto the osu! window.

**Too slow?** About a minute per difficulty on CPU is normal; "fast" placement quality halves it; with an NVIDIA GPU click **Set up GPU acceleration** in the window.

**A format will not decode?** Make sure the file plays at all; the program tries libsndfile, then the bundled ffmpeg, and reports the exact reason if both fail.

## Development schedule

Work follows priority and dependency order, without time commitments. Completion and release depend on validation, PRs, and Release notes.

| Priority / order | Work | Completion criteria | Status |
| --- | --- | --- | --- |
| P0 · Documentation and collaboration | Usage and attribution guidance, bilingual README, branch and contribution rules, community group | Consistent documentation that distinguishes planned and released features | Updated |
| P0 · Source identification | Generation records, model identities, content fingerprints, and an in-app source checker; a feasibility study for older v0 outputs | Read `.osu` / `.osz`; report tags, record matches, and experimental detection false positives / recall separately | Source checker implemented in source; statistical screen failed the integration gate |
| P1 · Difficulty control | Measured star ratings, a fixed evaluation song set, and density / spacing control for higher difficulties | Compare target and measured ratings; validate 6–7★ before extending to 7–8★, including local difficulty peaks and playability | Planned |
| P1 · Musical sections | Highlight control and automatic section selection | Validate manual regions first, then automatic selection; coordinate density, spacing, hitsounds, and kiai through buildup, peak, and release | Planned |
| P2 · Model iteration | High-star data, finer rhythm representation, slider-length and section conditioning, multiple timing sections | Update released models only after controlled comparisons and the fixed evaluation set show improvement | Depends on earlier experiments |

Statistical detection of older maps will enter the app only after independent evaluation at a low false-positive threshold. No detection means "unable to determine", not proof of human authorship.
See the [detailed task plan and analysis](docs/roadmap-2026-09-24.md) (Chinese) for scope, dependencies, and acceptance criteria.

## Branch management

| Branch | Purpose | Merge process |
| --- | --- | --- |
| `master` | Release-ready mainline with validated features | Merge through PRs reviewed by a maintainer; existing Windows / Ubuntu CI must pass |
| `feat/*`, `fix/*`, `docs/*` | Features, fixes, and documentation, one focused change per branch | Start from the latest `master` and open a PR against `master` |
| `exp/*` | Model training, new game modes, and detection experiments | Provide baselines, settings, and evaluation before extracting deliverable changes into PRs |
| `kanzei/*` | Maintainer and automation work | Follow the same PR and validation process |

Keep development changes off direct pushes to `master`; do not force-push or delete the main branch. Maintainers manage version tags and official Releases.
External contributors should fork the repository, work on a branch, and open a PR. Develop new game modes separately and verify the existing osu!standard flow.
GitHub main-branch protection requires a PR, documentation checks, and both platform tests; force pushes and deletion are blocked. Use squash merging.
See [CONTRIBUTING.md](CONTRIBUTING.md#english) for submission steps, [repository workflow](docs/governance.md) for configuration, and the [task queue](docs/TASKS.md) for implementation state.

## Open-source collaboration agreement

Forks, fixes, translations, model experiments, and new game modes are welcome. These are the collaboration terms; [LICENSE](LICENSE) governs use of the code and models:

- When contributing, confirm that you have the right to provide the code, models, or assets and agree to distribute your original contributions under the project's existing licence. Contributors retain their copyright and attribution.
- Identify the source of third-party code, models, and assets, and preserve their licences and copyright notices. Their original licences continue to apply.
- Open an issue before a substantial change, or record a group discussion in an issue, to coordinate work. Explain the change, validation, and known limitations in the PR.
- AI-assisted development is welcome; the submitter is responsible for understanding, reviewing, and validating the result. Model changes should include data sources, configuration, and reproducible evaluation.
- Clearly identify forks and derivative releases, their relationship to this project, and their changes. Do not present them as official releases. Follow the attribution requirements below for commercial or large-scale use.

The full submission workflow is in the [contribution guide](CONTRIBUTING.md).

## Community and feedback

**QQ community group: 1124526648** — search for this group number in QQ. Discuss generated maps, high-difficulty and highlight requests, usage feedback, and development.

Also record reproducible bugs, feature proposals, and collaboration tasks in [GitHub Issues](https://github.com/kanze1/AUTO-OSU/issues),
including the app version, generation settings, reproduction steps, and relevant logs so they can be tracked.

## Licence and attribution

Code and models are released under **MIT plus an attribution condition** (see [LICENSE](LICENSE)):

- Personal use, learning, playing around in the community: keep the copyright notice, exactly like plain MIT.
- **Commercial use or large-scale deployment** (a public web service, an app distributed to the general public, bulk generation for a platform or community):
  show **AUTO-OSU by kanzei** with a link to this repository, https://github.com/kanze1/AUTO-OSU, prominently in the product's interface, about page, store page or documentation.

Generated beatmaps are yours; the songs belong to their artists.

## Acknowledgements

- [osu-diffusion](https://github.com/OliBomby/osu-diffusion) (MIT) — DiT architecture and diffusion code, vendored in `autoosu/ml/coord`.
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) (MIT) — tokenisation ideas and the baseline we compared against.
- [project-riz/osu-beatmaps](https://huggingface.co/datasets/project-riz/osu-beatmaps) — the training corpus.
- [osu-dreamer](https://github.com/jaswon/osu-dreamer) — an earlier baseline.
- [Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC) (OFL) — the interface font, bundled as AUTO-OSU Sans.

Author: kanzei
