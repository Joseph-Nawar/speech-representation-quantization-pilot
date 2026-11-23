# 02 — Dataset, data splits, and model

## Primary corpus: VCTK 0.92, not two unrelated benchmarks
**Official source:** https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full. Edinburgh describes roughly 110 speakers, including shared passage material and CC BY 4.0 attribution. The full official archive is ~10.94 GB, so do **not** download it without authorizing resource use. A Hugging Face mirror (e.g. `thelarryproject/vctk` or `kth-tmh/vctk`) may permit streaming, but provenance, content consistency, actual streaming feasibility, and license must be checked, not assumed.

### Same-audio factorial construction
1. Read *metadata only* first: speaker ID, prompt/utterance index, transcript, microphone, audio path/hash, duration. Verify mirror IDs/metadata against official descriptions.
2. Keep **mic1** only; drop any duplicate `(speaker,prompt)` cells, missing audio/transcript, extreme durations, corrupted audio, and inconsistent transcript-to-prompt pairings using **fixed, logged rules**: duration in `[1.0, 10.0]` seconds after decoding, nonempty normalized transcript, one canonical prompt text per `prompt_id` (validate consistency over selected speakers; do not silently rename labels).
3. Derive **shared prompt IDs** present in many speakers. This is not a free-form text classification benchmark: every prompt class denotes a fixed shared spoken sentence/passage item.
4. Deterministic first choice: sort eligible prompt IDs ascending and take first 24 that occur in >=64 speaker candidates; sort eligible speakers by ID and choose first 64 that have all 24. If impossible, use deterministic maximum complete biclique search with priority `(number of unique clips, number of speakers, number of prompts, lexicographic IDs)` under bounds 40–64 speakers × 16–24 prompts; record the actual rule and selection. **Do not optimize selection using task or quantization scores.** If no >=40×16 complete grid, STOP data gate.
5. Keep exactly one audio example per speaker × prompt pair. Do not include mic2 recordings of the same underlying utterance.
6. Save `manifests/grid.csv` and SHA256 for IDs, source, checksums, label vocabulary, original source revisions. Store local audio in ignored `data/` only.

### Split protocol
- Closed-set classification requires **both speakers and prompt classes represented in all splits**. Splitting strictly by speaker would make closed-set speaker ID impossible; splitting strictly by prompt would make closed-set prompt ID impossible.
- Use a **speaker×prompt cell split** with train/dev/test = 60/20/20 by hash of `(speaker_id, prompt_id, 20261122)` plus deterministic balance repair **without inspecting labels beyond IDs and availability**. Verify each class occurs in each partition; require at least 8 train, 3 dev and 3 test clips per speaker and at least 12 train, 5 dev and 5 test per prompt. If requirements impossible at 40×16, STOP.
- Each cell occurs in exactly one split. Same audio must be used for both labels. No recording duplicates or near-duplicate microphone channels crossing partitions.
- This is **within-known-speaker and within-known-prompt** generalization, not generalization to new speakers or unseen text. Explicitly state this limitation.
- Freeze split manifest SHA before selecting heads or quant policies. Test *speaker/prompt metadata* may be used to audit split support and duplication because labels define the grid; test audio features, predictions, task scores, and any test-derived tuning signal remain sealed until P4.

## Input preprocessing
- Decode to mono, resample to **16,000 Hz** with a single pinned library/version, validate waveform finite and non-silent, retain full accepted waveform up to 10.0 seconds. No volume-based filtering or VAD after seeing results.
- Use model-supplied processor; record model and processor git revisions and all feature-extraction settings.
- Pad only within a batch and perform **attention-mask-aware** mean pooling of last four transformer hidden states (same weighting for both tasks); if an encoder requires special masking behavior, log the adaptation and verify with padding invariance tests. Never average zero-padded time steps.
- Disable dropout/SpecAugment and set `model.eval()`; inference uses `torch.inference_mode()`.
- Feature shape: one shared vector per clip, derived from unmodified model outputs; class heads get the **same vector**, ensuring fair task comparison.

## Primary model
- `microsoft/wavlm-base-plus` from Hugging Face (pin exact commit hash at preflight, not latest).
- Encoder and processor are frozen. Run MPS if correct; otherwise CPU with a report. Never assume CUDA.
- Full-precision reference: use **FP16 on a validated backend**; if MPS/CPU precision creates functional incompatibility, predeclare and document a fallback **FP32 reference** and compare all interventions consistently. Do not mix reference precisions across experiments.

## Probe training
- Shared frozen pooled features, then separate `StandardScaler` + multinomial `LogisticRegression` (class-balanced, fixed solver, `max_iter>=2000`, documented numerical tolerance) for:
  - **Content task:** classification among shared VCTK prompt IDs.
  - **Speaker task:** classification among VCTK speaker IDs.
- Tune regularization C on dev from `[0.01, 0.1, 1, 10]`, using macro-F1, deterministic tie-break smaller C; train features only for scaler and classifier. Freeze chosen parameters/weights before first quantized evaluation on dev.
- Keep no additional task-specific encoder layers, pooling methods, or auxiliary features. If either frozen probe is weak, STOP rather than hand-select a convenient representation.
- Primary probe qualification: dev macro-F1 >= **0.55** for both tasks and dev macro-F1 >= (majority-class macro-F1 + **0.20**) for both. Test predictions are forbidden until P4.

## P1 micro-preflight
- Use at most 128 selected audio clips, four evenly distributed transformer blocks (1, 4, 8, 12 if 12 total; validate actual indexing), and at most 4- and 8-bit fake-quantized variants.
- Ensure no mixed mic duplicates, valid audio resampling, head pipeline working, memory and timing within user-approved constraints.
- These P1 samples are **for engineering diagnostics**, not for any outcome estimate or final bit allocation.

## Secondary generalization (not required to pass initial pilot)
- If pilot passes, test on a second encoder (e.g. HuBERT Base), and more established downstream tasks (SUPERB keyword spotting / speaker ID). Do not mix these into the primary same-audio causal interpretation.
