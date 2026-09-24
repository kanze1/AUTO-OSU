# Source evidence in the application

Source checking started in **0.3.0.dev0**; **0.3.0.dev2** adds [new-map content watermarks](generation-watermark.md). The published 0.2.0 download does not have these features. Rhythm and coordinate weights remain v0.

Use **Check source / 检查谱面来源** in the desktop app. Select an `.osu`, `.osz`, or folder, optionally include subfolders, and save a JSON report. Checking runs in a background thread and does not load models, contact a server, or add records.

```powershell
python -m autoosu --check-source "map.osz" --source-report "source-report.json"
python -m autoosu --check-source "D:\Beatmaps" --recursive --source-report "source-report.json"
```

The result distinguishes:

| Result | Meaning |
| --- | --- |
| Contains an AUTO-OSU declaration | Tags or a matching archive manifest claim this source; anyone can copy that claim |
| AUTO-OSU content watermark detected | Circle coordinates match the public marker; its engine claim is not authentication and can be copied or removed |
| Matches a local generation record | Exact file bytes or the defined content fields match a record in the selected local store |
| Inconclusive | Insufficient evidence; this never means human authorship |
| Could not read | Invalid archive, unreadable input, or a resource limit prevented reading |

Statistical attribution of older unmarked maps is out of scope by maintainer decision. [The v0 screening experiment](detection-v0-screen.md) is retained as historical evidence. New-map marker checking does not depend on that classifier experiment, tags, the archive manifest or access to local records. A local-record match takes display priority while the JSON and text reports still show the marker result separately.

## New generation records

Generation writes `autoosu-provenance.json` inside each `.osz` and records the same manifest locally. By default records live in `~/.autoosu/provenance`. Set `AUTOOSU_RECORDS` or pass `--records-dir` to use another store for both generation and checking. There is no automatic upload or import of records from checked archives. Losing this folder loses that machine's matching history.

Schema `autoosu.provenance/1` contains the application version, random generation ID, timestamp, input and packaged audio hashes, actual model file hashes, seeds, effective presets, timing and sampling settings, and raw/content hashes for each map. Absolute input/model paths are omitted. The model hash is computed from the loaded checkpoint file identity, cached within a batch, and compared with the official v0 hashes; renaming a checkpoint does not change its identity. The version bump prevents selecting an older 0.2.0 managed runtime for the new worker; configure the runtime for the new app version when needed.

Engine labels distinguish rules, a single-model hybrid, and both-model generation. Custom checkpoints are identified by hash as `custom-model`. Pure rules output uses `rule-generated`, while outputs using either model use `ai-generated`. The song's `Source` metadata field remains the song source.

If a local record cannot be saved, generation preserves the completed `.osz` and reports a warning. Batch reports and worker results include the generation ID, record-save outcome, and warnings. A manifest and editable local records are not signatures or independent proof that a model ran.

## Content fingerprint scope

`autoosu-map/1` supports osu!standard formats v3–v14. It includes format version, mode, stacking leniency, sample set, six difficulty values, ordered timing points, ordered hit objects including slider anchors and hit-sample fields, and breaks. It normalizes numerical spelling, line endings, and surrounding field whitespace.

It excludes title, artist, creator, tags, IDs, colours, editor state, audio filename and non-break events. Audio, storyboard and external hitsound asset bytes are not included in a map's content match. Packaged audio has a separate recorded hash, which the source checker does not verify. The report therefore describes matching checked fields, not identical audiovisual behavior or authorship. This is a conservative canonical format, not a full osu! equivalence engine: a format-version change or a different equivalent curve representation can prevent matching.

ZIP inputs are read without extracting files. Limits are 4,096 archive entries, 16 MiB per map, 128 MiB total map data per archive, and 2 MiB for a manifest. Ambiguous duplicate names are rejected. Malformed manifests cannot create local matches, and one damaged input does not stop a folder report.

## Validation

- Unit/integration coverage includes metadata stripping, actual content edits, copied manifests, custom weight identities, corrupted archives/records, batch propagation, CLI checking without model loading, and a record-write failure after a successful export.
- Real inference used the released v0 weights, one frozen song, Hard and Insane, seed 240924, 100 coordinate steps: 8 maps across both models, both hybrid directions, and rules. All 8 matched the isolated local record store through an independent `python -I -m autoosu.worker` process; model paths used CUDA on an RTX 4090 and rules used CPU.
- A second independent worker check on 0.3.0.dev0 processed two songs as a folder batch; both outputs persisted their generation IDs and matched local records.
- Chinese and English GUI windows were exercised with an actual generated archive, with report results checked and screenshots visually inspected: [Chinese](images/source-check-zh.png), [English](images/source-check-en.png).
- osu! stable/lazer import, actual editor resaving, and player testing have not been performed. No claim about client round-trip retention or AI-detection accuracy follows from these checks.
