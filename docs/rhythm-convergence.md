# Rhythm v1 convergence continuation

The maintainer requested training to a validation plateau instead of stopping at 20,000 updates on 2026-10-08. The original four-GPU run remains intact. After its successful completion, `scripts/continue_architecture_training.py` hands GPUs 0–3 to `autoosu.ml.train_convergence`, initialized from its final checkpoint. No failed job is automatically restarted.

## Continuation boundary

The original script exported FP16 inference weights without AdamW state. This continuation therefore starts a new AdamW optimizer from those weights; it is not uninterrupted optimizer-state resumption. The original best checkpoint remains available for final comparison. The new phase saves FP32 model parameters, AdamW state, plateau-controller state, per-rank Python / NumPy / Torch / CUDA RNG state, epoch, batch position and DataLoader-generator state to `state.pt`. Worker prefetch state is not captured; bitwise DataLoader resumption is not claimed. Inference exports separately retain best rhythm F1, best attribute NLL, and latest weights.

## Frozen stopping policy

- Keep the same song-grouped splits, 128 validation songs, center crops of 1,024 ticks, unknown reference density, 12 generation rounds, temperature 0.9 and evaluation seeds. Test songs do not determine training duration or checkpoint choice.
- Use four GPUs, batch 32 per GPU, unchanged model and training objectives. New optimizer: AdamW, learning rate `5e-5`, 500-step warmup, betas `(0.9, 0.95)`, weight decay `0.05`, gradient clipping `1.0`.
- Validate every 1,000 continuation steps. Meaningful progress is either an absolute onset F1 increase greater than `0.002`, or a relative reduction greater than `0.005` in the equally weighted mean of the five heads' reference NLL. These are declared experiment thresholds, not statistical significance claims. Improvements accumulate against the last meaningful best value.
- After three evaluations without either improvement, halve the learning rate. Its floor is `3.125e-6` (four reductions). At that floor, eight consecutive evaluations without either improvement end training, provided at least 10,000 continuation updates have run. Any meaningful improvement resets plateau counters.
- A 120,000-continuation-step review boundary prevents unbounded execution; reaching it is explicitly `step_limit_pending_review`, never labelled convergence. Nonfinite values and process failures stop with diagnostics.

Convergence here means this validation policy reached a plateau. It does not prove global optimality, superiority to v0, or playable full-song quality. Independent test evaluation, frozen full-song generation, parsing and human playtesting remain distinct release checks.

## Verification

The controller tests cover learning-rate reductions before stopping, progress in either task, tolerance for small fluctuations, minimum training duration, state roundtrip and nonfinite / missing-supervision failures. An AdamW serialization test compares the next update exactly after restoring full-precision model and optimizer state.

Verified on 2026-10-08:

- Local full suite: **201 passed in 31.87 s**; repository checks: **24 Markdown files / 148 local links**; `git diff --check` passed.
- Real four-rank CUDA smoke initialized from an isolated copy of the 15,000-step checkpoint, used the frozen 128-song validation, performed two updates (batch 2 per GPU), validated again, and exited successfully after 73.33 s. Its terminal state was correctly `step_limit_pending_review`, not convergence.
- Read back `convergence-smoke/state.pt`: 443,032,003 bytes, continuation step 2 / cumulative step 15,002, all floating model parameters FP32, AdamW state for 128 parameters and four rank RNG records. The smoke lives under remote `runs/rhythm_v1_20261008`; it does not replace any main-run checkpoint.
- At **2026-10-08 18:30:51 Asia/Shanghai**, dispatcher **3548798**, owner **p2522808** (UID 1010), was confirmed in `waiting_for_original_training`, waiting on original controller **3515080**. Tmux session: `autoosu-v1-convergence`; output: `runs/rhythm_v1_convergence_20261008`. The continuation training children had not started at this observation.

Training-stage startup, actual convergence, independent acceptance and release are not asserted by these smoke / queue checks.
