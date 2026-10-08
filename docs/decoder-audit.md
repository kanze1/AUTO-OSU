# Decoder audit after the Haru comparison

The maintainer compared the generated maps with the local human Haru mapset and reported no increased satisfaction. This is negative qualitative feedback, not release acceptance. Training remains stopped. The production decoder and default parameters are unchanged by this audit.

## Verified implementation limits

- `sample_masked` commits ticks in confidence order. Committed labels cannot be revised within the decoding pass. More rounds change the commitment schedule; they do not reconsider previous decisions. This resembles the [official MaskGIT decoder](https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py), rather than being a departure from its design.
- Masked sampling does not enforce the slider label grammar while sampling. `repair_structure` subsequently removes orphan body/end labels, changes isolated heads to circles and closes incomplete sliders. It discards probabilities when choosing those repairs. Autoregressive sampling already uses a transition mask, but the current checkpoints use masked decoding; switching paradigms is not a compatible runtime option.
- Object attributes are sampled categorically after the rhythm pass, with the same temperature as rhythm by default. The five heads share context but do not condition on one another's sampled attributes. Attribute windows have no overlap, unlike rhythm windows. Whether these boundaries cause perceptible discontinuity remains untested.
- Normal target selection accepts the first structurally eligible candidate within 0.5 stars, or chooses the eligible candidate closest in stars. It is not a musical-quality or human-preference ranker.
- All three rhythm checkpoints still share coord v0. Its context consists of time, type, distance and class conditions, without direct audio input. Improving the rhythm encoder alone cannot be assumed to improve phrase-level spatial composition. Distance dropout exists in the coordinate training loader, so zero distance conditioning is not by itself evidence of a train/inference bug.

## Actual inference ablation

[Frozen settings, weights, source hashes and results](decoder-audit-20261008.json) cover Haru TV Hard and Mesheer, four variants each, one seed. Both use rhythm v1 best step 10,000 and coord v0. Rhythm conditions are fixed at 3.7373 and 6.5 respectively. The measured-star controller is disabled to isolate decoding; stars therefore differ. Coordinate sampling remains 100 steps with guidance 1.0.

| Variant | Haru stars | Haru repaired ticks / 1207 | Mesheer stars | Mesheer repaired ticks / 1968 |
|---|---:|---:|---:|---:|
| D0: 12 rounds, temperature 0.9 | 3.26 | 25 | 6.38 | 61 |
| D1: 24 rounds, temperature 0.9 | 3.39 | 22 | 6.24 | 19 |
| D2: 24 rounds, temperature 0.7 | 3.50 | 11 | 6.69 | 22 |
| D3: D0 rhythm, attribute argmax | 3.30 | 25 | 6.27 | 61 |

The repair counts refer to pre-export tick classes changed by structural repair, not invalid exported objects. The baseline repair rates are 2.07% and 3.10%. They expose a real decoding issue but do not explain the entire subjective gap. Increasing rounds helps Mesheer much more than Haru in these samples. Lowering temperature helps Haru repair counts but is slightly worse than D1 on Mesheer; there is no uniform winning setting.

D2 changes both rhythm and attribute temperatures; it does not isolate attribute calibration. D3 isolates attribute selection and was checked to preserve D0 event times, kinds and durations exactly. D3 reduces Mesheer new-combo starts from 150 to 116, but Haru changes only from 57 to 54. Taking argmax is not established as a better attribute policy and could suppress uncommon patterns. Star rating is not a quality metric, and neither fewer repairs nor fewer combo starts establishes better play feel.

All eight actual generation calls succeeded without warnings. Independent parsing checked object counts, slider repeats and edge sound fields; slider curves sampled at 1001 points remained finite and inside the playfield. Export diagnostics found no overlapping objects or invalid sliders. Map archives and the delivery ZIP passed CRC checks. There was no osu! client playtest, blind preference comparison or independent held-out evaluation. Recipes, the executed instrumentation script and eight maps remain outside Git under `out/decoder-audit-20261008`.

## Prioritized experiments

1. **Revisable rhythm decoding with legal structure.** Test confidence-based remasking at equal forward-pass budgets against the 24-round baseline, then integrate probability-aware valid-path decoding of slider head/body/end sequences. Grammar constraints express object validity; they must not hard-code preferred stream length, slider ratio or phrase density. Reusing an already revealed token to measure its own confidence would risk a copying shortcut; retain masked prediction confidence or evaluate masked spans instead.
2. **Separate attribute calibration and context continuity.** Calibrate each head on held-out data and test overlapping attribute windows. Keep stochastic options for rare patterns rather than setting every head permanently to argmax. Any future joint attribute model would require training; inference cannot invent missing dependency weights.
3. **Quality-aware candidate comparison.** Separate star tolerance from quality. Start with blinded human comparisons of similarly rated candidates and collect preference evidence before training or trusting a learned ranker. Ad hoc weighted counts are not validated proxies for satisfying maps.
4. **Coordinate/phrase conditioning if decoding reaches its limit.** Evaluate spatial generation separately from rhythm with fixed event sequences. Shared audio/phrase context is an architectural hypothesis requiring new supervision and training, not a decoder-only promise.

[ReMDM](https://remdm.github.io/) demonstrates remasking with pretrained masked diffusion models and increased inference budgets in other domains. It motivates the first experiment, but AUTO-OSU is a MaskGIT-style rhythm model; its training objective and task differ. This audit has not implemented ReMDM or verified a remasking quality gain on osu! maps.
