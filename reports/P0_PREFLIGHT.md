# P0 preflight

**P0 BLOCKED · P1 BLOCKED (unstarted) · 2026-11-22**

Study `speech-quant-diagnostic-20261122-v1` · audit code `7f12c4b99c62dd57d5af2cae358edde378e55519`

## Required amendment

The pinned WavLM architecture has **seven** in-block `nn.Linear` weights per block. The eligibility wording in [protocol §03](../docs/03_quantization_and_baselines.md) and the config can be read as covering all in-block linears, while §03 names exactly six Q/K/V/output and feed-forward projections and excludes positional-bias parameters. The seventh matrix, `encoder.layers.{i}.attention.gru_rel_pos_linear.weight` (8 × 64), takes the attention head state and gates the relative position bias in every block. Both interpretations change the actual model perturbation and total byte budgets. Runtime module enumeration and checkpoint tensor metadata independently agree on its presence in all 12 blocks.

**Proposed amendment A1.** Explicitly allowlist the six named weights per block and list the relative-position gate weight, its bias, `gru_rel_pos_const` and the relative-attention embedding as excluded. Use the same eligibility inventory for intervention, restoration assertions and byte accounting. If the researcher intends to quantize every in-block linear weight, expressly include the gate weight and document the changed intervention and endpoints. Record the decision with an amended study ID and protocol version before running P1. The original study config and protocol files retain their handoff hashes.

**Proposed clarification A2 before evaluation.** §01 describes a signed quantized-minus-reference delta and then names `FP16 - quantized` as loss. §04 defines loss and every gate using reference minus quantized. Name the two quantities separately if retaining both. Preserve §04's loss direction and gate values. No metric ran in this preflight.

## Dataset and source

- Edinburgh DataShare, VCTK 0.92 (September 2019 release, DOI `10.7488/ds/2645`). The official README and license match the published MD5s and specify CC BY 4.0. The archive bitstream ID is `535f4286-e54c-4038-838c-a02285e32cb2`, `ETag "8a6ba2946b36fcbef0212cad601f4bfa"`, 11,747,302,977 bytes. The archive contains 48 kHz audio with separate mic1 and mic2 paths. Mic1 is the DPA 4035 channel. The audit used only the ZIP directory and the pre-audio transcript region. The directory covers 44,455 mic1 and 43,873 mic2 recordings. The full ZIP SHA256 was not measured. [Official release](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)
- `thelarryproject/vctk` at `0492fa00cf72afb16e753e95d0ec3439eebb07a6` declares CC BY 4.0 and names the official archive in its provenance. A bounded Parquet read of all 22 pinned shards retrieved **2,790,546 bytes**, excluding the audio column. All 88,156 mirror rows agree on original path, speaker, utterance/prompt suffix, mic and official transcript after deterministic Unicode NFKC, case folding, apostrophe harmonization and whitespace normalization. There were zero duplicate (utterance, mic) keys. All 172 official recordings omitted by the mirror have no official transcript. The mirror adds duration, normalized text and a mismatch score. Selection used mirror duration and the official transcript only. Audio byte parity and decoded duration remain unverified. The official directory supplies per-member CRC32 for a future bounded official clip read or mirror parity check. [Pinned mirror](https://huggingface.co/datasets/thelarryproject/vctk/tree/0492fa00cf72afb16e753e95d0ec3439eebb07a6)
- `kth-tmh/vctk` at `bf27f6ca7ad62afe2212b7094ef1f198a81d8b66` exposes `speaker_id`, `text_id`, `text` and `file` with mic suffix. The card pairs a Public Domain label with CC BY 4.0 and provides no duration column. Its viewer is a partial export. This audit used its card and pinned repository listing. The mirror with durations supplied the grid evidence. [Pinned alternative](https://huggingface.co/datasets/kth-tmh/vctk/tree/bf27f6ca7ad62afe2212b7094ef1f198a81d8b66)
- **Metadata candidate PASS.** Twenty-four prompt IDs met the 40-speaker canonical-text criterion before the duration screen, and 23 afterward. The 64 × 24 first choice was unavailable. The fallback returned a complete **63 × 18** mic1 candidate with **1,134 cells** after the inclusive 1–10 s metadata filter. The search tested dimensions in descending clips, speakers and prompts order. Only one prompt set attained the winning dimensions under the audited normalization. The candidate remains local under `cache/`. **G0 UNVERIFIED** pending audio integrity, decoded eligibility and split-support checks. No split was generated and there were no held-out label or audio reads.

## Model, runtime and engineering

Model and processor both pin to `4c66d4806a428f2e922ccfa1a962776e232d487b`. The 377,617,425-byte checkpoint's published LFS SHA256 is `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`. The audit fetched the config, processor and card, plus a bounded 65,536-byte checkpoint prefix. The tensor metadata was decoded with pickle opcode inspection without loading or executing the checkpoint. The checkpoint card links UniSpeech's CC BY-SA 3.0 terms. Microsoft's upstream UniLM source has MIT terms. The report records the checkpoint-specific card reference. [Pinned checkpoint](https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b) [Linked terms](https://github.com/microsoft/UniSpeech/blob/main/LICENSE) [Upstream license](https://github.com/microsoft/unilm/blob/master/LICENSE)

Twelve runtime encoder blocks use indices 0–11. Each has Q/K/V/output weights of 768 × 768, feed-forward intermediate of 3,072 × 768, feed-forward output of 768 × 3,072, and the 8 × 64 gate. The serialized inventory has 248 FP32 tensors and 94,381,936 elements, matching the runtime parameter count. This arithmetic includes all encoder tensors and FP16 group scales. Probe bytes will depend on the head/scaler artifacts.

| Eligibility reading | Matrices | Eligible FP16 bytes | Excluded FP16 bytes | Calculated all-4 bytes | Calculated all-8 bytes |
|---|---:|---:|---:|---:|---:|
| Six named per block | 72 | 169,869,312 | 18,894,560 | 64,016,096 | 106,483,424 |
| Every in-block linear | 84 | 169,881,600 | 18,882,272 | 64,007,072 | 106,477,472 |

Apple M5 (10 CPU cores, arm64), macOS 26.4 build 25E246, 16 GiB unified RAM and about 100 GiB free on the project volume after setup. The default Python was 3.9.6. Project-local Python 3.12.15 with Torch 2.15.0, Transformers 5.24.0, torchaudio 2.11.1 and SoundFile 0.14.0 imported successfully. CPU and MPS repeated tiny FP16 and FP32 matrix multiplies exactly. CUDA was unavailable. The runtime check took 7.24 s and peaked at 495,779,840 bytes RSS. Peak unified memory usage was unavailable. WavLM forward compatibility, mask/padding invariance and repeatability across MPS/CPU remain untested. There are no measurements of deployment bytes or inference acceleration.

Five P0 tests passed for coverage of both eligibility readings, missing matrices and packed-group byte arithmetic. The first unrestricted pytest run traversed dependency tests in the local package cache. The first project-only run caught incorrect expected byte values in two test fixtures. Both logs were retained. Split, leakage, feature cache prediction parity, classifier fitting, signed-level round trip and master-weight restoration await the amended implementation. **Zero audio clips, zero model forwards, zero probe fits, zero simulated conditions.**

## Evidence and exact audit commands

The [status JSON](../P0_P1_STATUS.json) records full hashes, revisions, local log paths, results and amendment text. The [closest-work matrix](P0_NOVELTY_MATRIX.md) records direct prior art for cross-task rank disagreement, top-K overlap and additive, budgeted allocation in TASA. The audit found no verified replacement for this specific speech falsification test.

```sh
python3.12 tools/check_package.py
python3.12 tools/p0_source_ranges.py official
python3.12 tools/p0_source_ranges.py texts
.venv/bin/python tools/p0_source_ranges.py mirror
python3.12 tools/p0_grid_audit.py
.venv/bin/python tools/p0_env.py --out runs/P0/env.json
python3.12 tools/p0_model.py cache/p0-model/checkpoint-prefix.bin --out runs/P0/model.json
.venv/bin/python -m pytest -q
```

The range helpers originally ran in the ignored audit workspace and were retained under `tools/`. Package versions are pinned in [requirements-p0-observed.txt](../requirements-p0-observed.txt). The model evidence command exits successfully when it writes the inventory, whose explicit stage status is `BLOCKED`.

**STOP — blocked pending the WavLM eligibility amendment.**
