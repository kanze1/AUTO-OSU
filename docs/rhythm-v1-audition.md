# Rhythm v1 full-song A/B/C audition

The maintainer stopped continuation at approximately 23,900 cumulative steps and requested playable comparisons. Four development songs were generated with three checkpoints each, for 12 full-song maps. These are comparison material for human review, not independent test-set or client acceptance.

- A: released rhythm v0, step 40,000.
- B: new rhythm / attribute model, best validation onset F1, step 10,000.
- C: continuation best validation onset F1, cumulative step 21,000.

All variants use the same current generation source, coord v0, 6.5-star input condition, seed 240924, reference BPM / offset, 12 rhythm rounds, 100 coordinate steps, temperature 0.9 and highlights off. No reference rhythm or skill label is supplied. Output stars are measured; identical input conditions do not guarantee identical output difficulty. Frozen source / model / audio identities and numerical results are in the [compact report](rhythm-v1-audition-20261008.json).

| Song | Variant | Stars | Objects | Sliders | Longest quarter-beat circle run |
|---|---|---:|---:|---:|---:|
| Joushiki! Butler Koushinkyoku (Cut Ver.) | A-v0 | 5.64 | 468 | 27 | 3 |
| Joushiki! Butler Koushinkyoku (Cut Ver.) | B-v1-best | 5.15 | 399 | 48 | 3 |
| Joushiki! Butler Koushinkyoku (Cut Ver.) | C-v1-continued | 5.00 | 393 | 40 | 3 |
| Fruit Salad | A-v0 | 4.42 | 241 | 204 | 3 |
| Fruit Salad | B-v1-best | 4.64 | 227 | 200 | 3 |
| Fruit Salad | C-v1-continued | 3.92 | 203 | 49 | 2 |
| eyes to eyes | A-v0 | 5.71 | 490 | 278 | 5 |
| eyes to eyes | B-v1-best | 5.95 | 516 | 296 | 12 |
| eyes to eyes | C-v1-continued | 6.34 | 536 | 240 | 14 |
| Mesheer | A-v0 | 6.72 | 645 | 128 | 33 |
| Mesheer | B-v1-best | 6.38 | 558 | 113 | 4 |
| Mesheer | C-v1-continued | 6.13 | 511 | 56 | 4 |

Observed differences: eyes to eyes has longer circle runs (5 / 12 / 14), while Mesheer loses the old long run (33 / 4 / 4). Fruit Salad changes from 204 / 200 / 49 sliders. These examples demonstrate substantial per-song variation and a remaining fast-song long-run regression; they do not justify replacing the released weights. The new variants export nonzero slider-tail additions and a few repeat sliders.

Verification: 5187 objects and 1679 sliders independently parsed; slider repeat counts and head / tail / body sound fields match exports. All slider curves were sampled at 1,001 points, with no nonfinite or out-of-playfield samples. Two sliders were shortened by the existing placement correction, one in each new first-song version. Manifest and local-record raw matches were checked only within this local store. All 12 generation calls succeeded; the final archive passed its CRC check. Local full suite: 201 passed; no osu! client playtest is claimed.

The audition package contains 12 `.osz` files, 12 synthesized-click rhythm previews, instructions, recipe and results. Click previews expose timing and new-combo differences, not actual osu! hitsound samples. Import the maps in osu! to review real sound additions, slider geometry and play feel. The package stays outside Git under `out/rhythm-v1-20261008/abc-audition-v2`.
