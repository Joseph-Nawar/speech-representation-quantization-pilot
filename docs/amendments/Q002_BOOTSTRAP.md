# Q-002 — Pre-outcome correction to crossed-cluster uncertainty calculation

**Status:** ADOPTED after the pre-outcome synthetic statistical validation below passed; adopted before any empirical P2 outcome metric.

**Date:** 2026-11-22  
**Study:** `speech-quant-diagnostic-20261122-v3`, protocol version 3, parent `speech-quant-diagnostic-20261122-v2`.  
**Applies after:** Q-001 protocol v2 and before the first full-grid FP task evaluation or quantized development outcome.  
**Scope:** Bootstrap resampling mechanism and implementation tests only.

## Inconsistency and replacement

Protocol v2 combined independent multinomial resampling of speaker and prompt IDs with fixed macro-F1 labels and zero scores for labels omitted by a replicate. On a perfect 63 × 18 crossed classifier, this assigns a zero class score whenever a factor level is omitted, making the bootstrap score distribution substantially less than 1 despite perfect predictions. This conflates absent bootstrap factor levels with classifier errors. The synthetic-only regression test reproduces this deletion effect.

Conditioned on the tests below, replace multinomial deletion resampling with positive-weight crossed-factor multiplier resampling. For each of 2,000 replicates, draw independent positive `Gamma(shape=1, scale=1)` speaker weights `a_s` and prompt weights `b_p`; set every observed cell weight to `w_sp = a_s * b_p`. Mean-one normalization within each factor is allowed and used. Reuse the same cell weights for every condition in a paired contrast. Calculate weighted confusion counts and class F1 over every predeclared label; all classes with original true-label support must remain positive in every replicate. Undefined precision/recall uses zero division only where genuinely undefined.

Use root `bootstrap_seed=20261124`, percentile 95% intervals, and deterministic named partition substreams. Each derived seed is SHA256-bound to the root seed, study ID, split hash, bootstrap specification hash, and partition name; `dev` and `test` are separate. Test data and outcomes remain sealed until P4. There is no optional stopping, seed shopping, iid clip bootstrap, or alternative weight distribution.

This method is an **approximate positive-weight multiway bootstrap**, inspired by [Owen & Eckles (2012), *Bootstrapping data arrays of arbitrary order*](https://doi.org/10.1214/12-AOAS547). Their result concerns variance estimation for means over crossed random-effects arrays; it does not establish exact 95% coverage for this pilot's nonlinear macro-F1. Positive weights keep the observed class labels in each replicate, but do not test unseen speakers or novel prompts. The P2/P3/P4 reports must retain these limitations.

## Required pre-outcome validation

All validation below uses synthetic data only and must finish before new audio is decoded or any P2 empirical score is calculated.

1. A perfect 63 × 18 crossed classifier scores exactly 1.0 in all 2,000 weighted replicates for both speaker and prompt tasks.
2. Identical paired predictions have exactly zero weighted delta (within `1e-12`) in every replicate.
3. Weighted macro-F1 matches a separately coded weighted-confusion reference on small grids with fixed labels, repeated weight values, and zero-division cases.
4. Every observed class retains positive weighted true-label support in every replicate; no NaNs or silent label deletion.
5. Named stream generation is reproducible and independent for dev/test; all conditions within one stream reuse the same weights.
6. Run the fixed, bounded synthetic Monte Carlo diagnostic in `tools/q002_synthetic_mc.py`: 40 × 16 crossed observations, four fixed classes, balanced prompt-class counts `[4,4,4,4]` and imbalanced counts `[8,4,2,2]`, speaker-effect SD `0.55`, prompt-effect SD `0.40`, interaction SD `0.0` and `0.8`, 250 outer datasets and 500 bootstrap replicates per dataset. Its target is the analytically integrated population macro-F1 for the stated four-class DGP. Report empirical coverage, its outer-simulation Wilson interval, interval widths and mean point error. These results are descriptive diagnostics for this DGP only; they do not prove pilot coverage or exact nominal coverage and cannot be used to tune the pilot interval.
7. Keep the legacy multinomial deletion-bias test and explain why the former fixed-label zero-fill rule is invalid for this target metric.

If a required invariant fails, the Monte Carlo results show material instability, or material doubts about validity remain, stop with `BLOCKED_STATISTICS` before empirical P2 outcomes and propose a follow-up amendment. Do not change the weight distribution or substitute an iid clip bootstrap.

## Invariants and non-changes

Q-002 does not change H1–H4; G0–G5 definitions or thresholds; class labels; VCTK source, grid-selection, split or minimum-support rules; WavLM/processor revisions; Q-001's 72 eligible matrices; probe architecture or C grid; groupwise symmetric quantizer; bit widths; primary `f=0.50` budget; 2,000 pilot bootstrap repetitions; seed; or confidence level. P2 computes only dev conditions FP, uniform4 and uniform8, then stops. G3/G4/G5 stay open/unassessed at P2.

## Versioning and audit

Protocol v3 links Q-001 and Q-002 and names v2 as parent. Exact affected v2 files are archived in `docs/archive/v2/` with `SHA256SUMS`; P0/P1 reports, statuses, local logs and earlier hashes remain historical evidence. The original P0/P1 entry point `sqpilot.cli` remains unchanged; Q-002 statistical preflight uses a separate module. Cache identity includes study ID, split SHA256, bootstrap-spec SHA256, and named partition. The final amendment diff and amended file checksums are recorded in `runs/P2_Q002/` and `P2_STATUS.json`.

## Validation result before P2 outcomes

- Command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`; **35 passed**, 1 pre-existing PyTorch attention-mask dtype deprecation warning; 21.78 s pytest wall time, 22.89 s process wall time, 517,554,176 B maximum resident set.
- The legacy multinomial deletion regression produced a mean perfect-classifier fixed-label macro score of about 0.64 (test bounds: 0.58–0.70); the positive-weight 63 × 18 invariant returned exactly 1.0 in all 2,000 replicates for both tasks.
- Identical paired predictions had zero delta within `1e-12`; independent weighted-confusion reference, strict positive support, finite-value checks, reproducible distinct named streams, cache-key binding, and corrupt-stage rejection passed.
- Fixed 250 outer × 500 inner Monte Carlo results: balanced/no interaction coverage **0.960** (Wilson 95% Monte Carlo interval 0.928–0.978); imbalanced/no interaction **0.940** (0.903–0.963); balanced/interaction SD 0.8 **0.976** (0.949–0.989); imbalanced/interaction SD 0.8 **0.928** (0.889–0.954). All interval widths were finite and positive; mean widths were 0.156–0.183. This is plausible but limited diagnostic evidence for the specified four-label DGP, not a coverage guarantee for the pilot's 18/63 classes.
- Synthetic result SHA256: `982a5c03a2d68fd015fdfbdf4d2532258905e1d95234b3f9dcecc2bed231ecaa`; bootstrap specification SHA256: `3e537fe0aebe21813c634d535e6e2760e1bcb8b3536569a513df94273ea52e05`.
- No corpus audio, model forward, task prediction, P2 empirical metric, or test outcome was accessed during this validation.

The result leaves a finite-sample/generalization limitation, but no invariant failed and the bounded diagnostics did not show clear systematic undercoverage. Q-002 is adopted as an **approximate** method with this residual risk explicitly retained for researcher review. Full statistical validity for this task/class structure is not established.
