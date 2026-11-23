# Q-001 — WavLM relative-position gate exclusion

**Status:** RESEARCHER-APPROVED FOR IMPLEMENTATION, subject to source/model verification.

**Date:** 2026-11-22

**Reason:** P0 found a seventh `nn.Linear` per encoder block (`attention.gru_rel_pos_linear.weight`) not explicitly named in the six-projection intervention. Scope ambiguity changes which weights are perturbed and the encoded-byte accounting.

**Prior result integrity:** P0 reported zero audio downloaded/processed, zero model forwards, zero quantizer interventions, and zero test outcome reads. This must be checked against actual P0 logs before applying this amendment. No empirical quantization result can be reused under the revised protocol without revalidation.

## Binding intervention rule

For each of the 12 WavLM Base Plus Transformer blocks in the pinned checkpoint, quantize **exactly six weight matrices**:

1. Attention query projection weight (`attention.q_proj.weight`)
2. Attention key projection weight (`attention.k_proj.weight`)
3. Attention value projection weight (`attention.v_proj.weight`)
4. Attention output projection weight (`attention.out_proj.weight`)
5. Feed-forward intermediate projection weight (`feed_forward.intermediate_dense.weight`)
6. Feed-forward output projection weight (`feed_forward.output_dense.weight`)

Resolve these names against the *actual pinned module paths* and fail closed if paths, module types, counts, or shapes differ. These are role descriptions, not permission to guess or alias unmatched weights.

**Explicit exclusions** (fixed at the predeclared reference precision in every intervention):

- `attention.gru_rel_pos_linear.weight` and its bias: this small linear gate modulates relative-position bias. Its exclusion is controlled isolation, not an assertion that the gate is unimportant.
- All other non-allowlisted parameters, including other biases, normalization parameters, positional modules, convolutional feature extractor, feature projection, embeddings, and frozen task-head parameters.

Each of 12 blocks must inventory 7 in-block `nn.Linear` **weight matrices**, of which exactly 6 are quantizable and 1 is excluded. Expected inventory totals: **84 in-block linear matrices; 72 eligible and 12 explicitly excluded gates**. Inventory the complete checkpoint as well and explain any other non-block modules, but do not add them to the intervention. All 72 eligible matrices receive 4- or 8-bit fake quantization according to the selected block bitmap. Unselected weights remain unchanged.

## Reproducible byte accounting

**Two separate measurements must never be conflated:**

A. **Calculated packed-storage estimate** for the specified per-output-row contiguous groups of 64, signed symmetric 4/8-bit codes and FP16 scale per group. The model continues to run with reconstructed floating weights; its *runtime RAM and speed* are not packed inference results.

B. **Actual device/host memory and timing** observed during the fake-quantization experiments, which are not compression or inference-acceleration claims.

For a 2-D eligible matrix with row width `C`, enumerate each row's consecutive groups of up to 64 weights. For each group containing `n` weights at bit-width `b`:

`group_cost_bytes = ceil(n * b / 8) + sizeof(FP16_scale)` (2 bytes per scale), plus any additional explicitly serialized format overhead if the implementation uses it. Sum per-group, per-module, per-block, and across all 12 blocks. Test partial groups and odd 4-bit counts. Do not round at the whole-matrix level if the representation pads each group separately.

For the *total*-model estimate add every excluded model parameter at its recorded fixed reference dtype (including all twelve relative-position gate matrices and biases). Report model-only totals and, separately, any fixed probe/scaler storage if the protocol counts probes in the evaluated artifact. Probe byte costs cannot be asserted as exact before the heads are fitted. Any fixed overhead must be treated consistently across all policies, including uniform endpoints.

Recalculate `B4`, `B8`, and `B(f) = B4 + f*(B8-B4)` for `f = 0.25, 0.50, 0.75` based on the revised scope. The **primary `f=0.50`**, tolerances, quantizer design, fixed heads, evaluation metrics, split rules, G0–G5 thresholds, and bootstrap design remain unchanged. Identify exactly how budgets change compared with the earlier interpretation; report either actual bytes or a derivation showing why some differences cancel. Do not silently reuse stale byte-budget files.

## Research versioning and audit trail

- Preserve the original P0 preflight report, P0 logs, initial protocol/config, and proposed amendment verbatim as historical evidence.
- Create `docs/amendments/Q001_SCOPE.md` in the research repo documenting decision, interpretation before/after, rationale, effects, and affected artifacts.
- Since `configs/pilot.yaml` explicitly requires a new `study_id` for research changes, introduce a distinct amended study identifier (e.g. `speech-quant-diagnostic-20261122-v2`) and increment `protocol_version`; link to the prior v1 ID and Q-001 in a machine-readable field or provenance record.
- Update `docs/01_protocol.md`, `docs/03_quantization_and_baselines.md`, `docs/04_analysis_and_gates.md` only where needed to eliminate conflicting scope or accounting wording; update `configs/pilot.yaml` accordingly. Don't rewrite gates or hypotheses.
- Update all code, tests, and reports to use amended identifiers; explicitly invalidate any cached costs or quantization artifacts from a different scope hash.
- Provide an old/new diff and hash the amended files before any P1 audio or forward pass.

## Scientific interpretation

The intervention is **selective projection-weight quantization**, not all-linear WavLM quantization. Holding the relative-position gate fixed helps isolate major projection sensitivity but limits generalization to fully quantized deployments. A later experiment quantizing all seven linears would require a separate study or preregistered secondary condition, not silent inclusion.

**Do not change any non-scope variable just to make P1 run.** If source, audio decoding, mask handling, or backend behavior forces additional scientific changes, stop and propose Q-002 rather than improvising.

## Repository record

Study `speech-quant-diagnostic-20261122-v2`, protocol version 2, supersedes v1 for Q-001. The v1 protocol/config are preserved verbatim in commit `4cf306f`, and the original P0 report and indexed logs remain in place. The prior status is preserved in [P0_STATUS_V1.json](../../reports/P0_STATUS_V1.json). Before amendment implementation, all prior indexed SHA256s matched, and the archived status and logs showed zero audio clips, checkpoint forwards, quantization variants, fitted heads and test reads.

The amended configuration pins the already audited model and processor revision. The eligibility list is exact and shared by the intervention and accountant. Compatibility with cached artifacts requires the amended study and scope hash.
