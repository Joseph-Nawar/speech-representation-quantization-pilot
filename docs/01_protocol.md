# 01 — Scientific protocol (frozen before P2)

## Question and scope

Protocol version 3 applies [Q-001](amendments/Q001_SCOPE.md) and [Q-002](amendments/Q002_BOOTSTRAP.md), and descends from study `speech-quant-diagnostic-20261122-v2`. Q-001 fixes the WavLM intervention at six specified projection matrices per block. Q-002 changes only the uncertainty resampling procedure; all hypotheses, tasks, data/split definitions, thresholds, metrics, quantization conditions, and primary budget are unchanged.

Under **one shared speech corpus, one frozen self-supervised encoder, one fixed audio preprocessing policy, and two frozen task-specific heads**, does 4/8-bit weight-only compression produce conflicting *layer-level* sensitivity for (A) spoken sentence/prompt classification and (B) closed-set speaker identification? If so, can simple existing cost-matched multi-task allocation policies already handle the conflict?

### Primary hypotheses
- **H1 (non-trivial quantization):** the 4-bit intervention measurably harms at least one clean task, but 8-bit does not catastrophically break either task.
- **H2 (cross-task conflict):** layer-level task performance losses under 4-bit interventions are not strongly aligned, with consequential task-specific differences; the difference is robust to resampling.
- **H3 (unhandled conflict):** when selecting 8-bit vs 4-bit layers under fixed byte budgets, strong simple multi-task allocation policies leave a practically relevant gap relative to a *feasible* alternative, confirmed on sealed test data.
- **H4 (distinct mechanism, not established by this pilot):** a new mechanism could address the observed failure better than known task-aware/multi-objective methods. This requires a **separate** novelty review and later research.

H1/H2/H3 must pass to authorize method design. H4 is not automatically established by passing H1–H3.

## Primary endpoints
1. Per-task **macro-F1** (higher is better), computed over all predefined speakers or prompt labels with explicit zero-division handling; accuracy as a secondary metric.
2. Layer-level macro-F1 drop in **percentage points**, quantized minus FP16 reference (report sign convention clearly, normally loss = FP16 - quantized).
3. **Spearman rank correlation** of the 12-layer sensitivity scores across two tasks, **top-4 sensitive layer overlap**, and absolute between-task loss differences.
4. For complete mixed-precision configurations, **worst-task normalized headroom loss** `R(config) = max_t max(0, (F1_FP_t - F1_config_t) / max(F1_FP_t - 1/K_t, 0.15))`, using each task's number of balanced labels K_t. Also report raw macro-F1 differences.
5. Exact *encoded* bytes under the specified quantizer, including metadata/scales and fixed unquantized weights. Do not present FP16 dequantized simulation RAM as compressed model RAM.

## Primary analysis unit
Unique VCTK clip ID `(speaker_id, prompt_id, microphone=mic1)`. The exact **same clip IDs** must be evaluated for both tasks, for every quantization condition. No pooled mic1/mic2 duplicates.

## Design stages
- P0: resource, license, technical, novelty preflight, no evaluation claim.
- P1: small smoke run; confirm actual same-audio dataset grid, MPS compatibility, quantizer round-trip tests, feasible clean linear probes.
- P2: make and **freeze** the complete manifests, cleaned audio sample lists, full-precision heads, quantizer and all analysis scripts on training/dev only.
- P3: development-only sensitivity and policy screening; pre-register configurations and test plan before P4.
- P4: sealed test, uncertainty analysis, predeclared decision engine, negative/positive report.

## Training and evaluation isolation
- Encoders **never fine-tuned**. Train each linear head on **full-precision training features only**. Freeze it for all 4/8-bit experiments. Do not retrain under quantization (would measure recoverable information, not fixed-task robustness).
- All scale fitting/normalization for probes uses train only. Probe regularization choices use dev only.
- Optimization, bit selection, calibration samples, prompt selection, or calibration hyperparameters never use test outcomes. Test IDs/known class labels may be used only for predeclared integrity/support audits; no test features, predictions, or outcome metrics are accessed until P4.
- Fix two class label maps for the entire study. No change in class count between conditions.
- No data augmentation or manually removing inconvenient failed samples. Predeclare deterministic audio acceptance rules.

## What is *not* claimed
- Prompt ID classification is a controlled linguistic-content proxy. It is **not** proof of phoneme preservation or natural ASR performance.
- A mismatch between task sensitivities does not imply a new mechanism exists.
- A positive pilot does not establish cross-corpus or cross-encoder generalization.
- Headroom against a simple policy does not beat the strongest published methods; a later full paper must compare with appropriate recent baselines.

## Negative controls
- FP16 repeated inference produces identical predictions within numerical tolerance for the chosen backend.
- Assign all transformer blocks 8-bit vs 4-bit: 8-bit should generally perturb less; report exceptions, never auto-correct results.
- Randomly permuted labels yield near-chance probe performance on dev (implement a dedicated control, never train using test labels).
- Accuracy on quantized features is evaluated with **the exact FP16 head checkpoint and scaler**.
- Results do not change when the same condition is resumed from cache vs executed from scratch (within declared numerical tolerance).
