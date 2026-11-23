# 03 — Quantization intervention, storage accounting, and baselines

## Fixed perturbation scheme (pilot)
- **Weight-only** post-training symmetric uniform quantization-dequantization. No activations quantized, no encoder training, no observer/calibration learning and no GPTQ optimization in the pilot.
- Quantize **exactly the six allowlisted projection weights in each of the 12 Transformer encoder blocks** under [Q-001](amendments/Q001_SCOPE.md). Verify 72 eligible matrices and 12 excluded `attention.gru_rel_pos_linear.weight` matrices against the pinned WavLM implementation. Fail closed on a path, type, count, or shape mismatch.
- Exclude the relative-position gate and every other non-allowlisted parameter, including conv frontend, feature projection, positional conv/bias, layer norms, embedding-related modules, all biases and both task probes; these remain baseline precision. This isolates block-level sensitivities but limits achievable whole-model compression.
- Per-output-row *contiguous* groups of 64 weights; groupwise symmetric affine scaling with **zero-point 0**. Levels for signed n-bit: `[-(2^(b-1)-1), ..., +(2^(b-1)-1)]`; scale = max(abs(group)) / qmax, with explicit zero-scale behavior. Deterministic nearest rounding and saturation. Report the unused most-negative signed code. Float16 scales by default.
- Test b = 4 and 8. Retain FP16/FP32 unquantized baseline. If group-size or reference precision changes, version the protocol and restart before outcome testing.
- For the primary diagnostic, reconstruct values into the original floating dtype for the standard PyTorch forward pass. It is **fake quantization**, not an actual fast int4/int8 kernel.

## Strict implementation controls
- Rebuild every variant from the same immutable master checkpoint; never quantize a previously quantized checkpoint.
- Before model inference, record per-module changed weights, reconstruction error and selected bit map. Assert only intended modules changed.
- Model API and source hash unchanged across configurations; downstream heads and label encoders unchanged.
- Test same input against FP reference twice, same quant config twice, quantized-then-restored vs fresh FP, padding invariance, and no NaNs.
- If b=8 performs anomalously worse than b=4, investigate first, but never discard the result without documented cause.

## Storage budget (do not pretend fake quant saves actual RAM)
For each eligible matrix/group, model packed data cost as `ceil(n_group_weights * b/8)` bytes + 2 bytes for FP16 scale, including padding overhead for last groups. Excluded parameters and probes are counted at their fixed baseline dtype. Save per-module bit cost and packed-layout spec. Either implement bit packing and assert serialized length agrees with model, or label it explicitly as **calculated packed bytes** rather than measured serialized bytes.

Build endpoint costs for uniform all-4 and all-8 in eligible transformer weights, with fixed parameters included. Define three *total packed-storage byte* budgets, including all fixed reference-precision parameters, as:
`B(f) = total_bytes_4 + f*(total_bytes_8 - total_bytes_4)` for `f∈{0.25,0.50,0.75}`.

Configuration uses `{4,8}` per Transformer block. Enumerate all `2^L` bit assignments (L expected 12; enumeration is trivial), reject ones above B(f). Score policies only on permissible configs. Report achieved bytes and allow a <=0.5% tolerance between compared configurations; if not matched, disclose bytes and use Pareto plots, not a winner claim.

## Predeclared allocation competitors
All *ranking* or *sensitivity* scores are computed using **development data only**, under 4-bit single-block interventions (all other blocks full precision). Rank with exact deterministic ties.

1. **Random-feasible:** 10 predeclared random bitmaps per budget selected from admissible configurations using fixed RNG seeds. Report median and spread; do not cherry-pick random winner.
2. **Position-prior:** maintain higher precision in early blocks, late blocks, or alternating blocks; these simple structural controls are necessary.
3. **Global representation drift:** prioritize blocks with greatest pooled-vector cosine drift relative to FP model, measured on dev audio; no labels used to derive drift.
4. **Content-only sensitivity:** prioritize blocks with greatest dev content macro-F1 drop under isolated 4-bit perturbation.
5. **Speaker-only sensitivity:** analogous speaker metric.
6. **Mean normalized task sensitivity:** equal-weight mean of the two tasks' dev-normalized sensitivity scores.
7. **Minimax sensitivity:** prioritize blocks according to worst normalized task-specific sensitivity (`max` over tasks), a strong simple multitask comparator.

For score-based policies, choose a bit assignment maximizing `sum_l score_l * 1[bit_l=8]` subject to exact byte budget using exhaustive enumeration of block assignments. This is an **additive heuristic**; real block interactions must be measured by evaluating the full chosen mixed-precision model. Reconstructed full-model performance, not summed scores, determines results.

## Headroom diagnostic (not a fair winner baseline)
Draw an additional **20** predetermined feasible random mixed-precision configurations on dev per budget, evaluate them, and pick the best by predeclared dev worst-task objective. This is a **search-based diagnostic upper-envelope** (can overfit dev), not an algorithm we claim to beat or a publishable new method. If it beats simple multitask policies by a meaningful margin on dev, seal that choice and confirm it once on test. Do not report it as a scientifically fair equal-search-budget method comparison.

## What the pilot cannot claim
- No MPS int4/int8 runtime speedup from float roundtrip.
- No end-to-end model binary compression unless a real serialized packed artifact exists.
- No superiority over full published quantization methods without implementing them or citing and explaining exact comparability limitations.
- No inference that rank disagreement necessarily justifies a novel objective.
