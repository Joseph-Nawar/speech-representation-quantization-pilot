# Amended P0 preflight

**PASS · Q-001 · 2026-11-22**

Study `speech-quant-diagnostic-20261122-v2`, protocol version 2. The [original P0 report](P0_PREFLIGHT.md), [original status](P0_STATUS_V1.json) and original logs remain intact. All indexed v1 hashes matched before amendment. Those records show zero weighted model forwards, test reads and quantization outcomes.

## Scope and provenance

[Q-001](../docs/amendments/Q001_SCOPE.md) makes the six named attention/feed-forward projection matrices per block eligible. The exact Transformers implementation and pinned checkpoint metadata agree on 12 blocks, 84 in-block linear weights, 72 eligible projections and 12 excluded relative-position gates. The convolutional frontend, feature projection, biases, normalization, relative-position embedding and constants, and any fitted heads remain at reference precision. The architecture inventory contains 248 parameter tensors with 94,381,936 elements. The originally serialized positional-convolution weight norm tensors map to the pinned runtime's two parametrized weights.

| Provenance | Identifier |
|---|---|
| Code | `be4fdc264fe961315dd0bf3a119e0d38cbd4a2cd` |
| Model and processor revision | `4c66d4806a428f2e922ccfa1a962776e232d487b` |
| Official checkpoint SHA256 | `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51` |
| Amended config SHA256 | `e7caee286965dc758ee0fca22b3bdd683910e4107fc060b7beedccd5aa301c9e` |
| Amendment SHA256 | `1fa4ee8e2d9ff40d86429d2a85708422cdf3ca91c579a7d86f74f58f3ef04a2d` |
| Scope SHA256 | `06f26cb2c13e07d9c31dfe99e9ae888760000e8f19c573e63de445ebecdd6dd1` |
| Candidate metadata grid SHA256 | `40e06426c15179de44848609c68abfa83567a651097ff8a656c6f1c6c0c686ac` |

The exact [protocol diff](../runs/P0_Q001/protocol-final.diff) and [source hashes](../runs/P0_Q001/amendment-final-sha256.txt) were recorded before any P1 audio transfer. Scope and code identity are part of cache verification. The checkpoint card links CC BY-SA 3.0 terms, and VCTK 0.92 is CC BY 4.0. The model remains local. [Pinned checkpoint card](https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b) [VCTK official release](https://doi.org/10.7488/ds/2645)

## Calculated packed-storage estimates

Model-only bytes include the fixed relative-position gates and all other excluded model weights, plus per-group FP16 scales. Probe storage is separate. These figures are arithmetic over the verified tensor shapes. Runtime still uses reconstructed floating weights.

| Reference | All 4 | B(.25) | B(.50) | B(.75) | All 8 | Fixed bytes |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 64,016,096 | 74,632,928 | 85,249,760 | 95,866,592 | 106,483,424 | 18,894,560 |
| FP32 | 82,910,656 | 93,527,488 | 104,144,320 | 114,761,152 | 125,377,984 | 37,789,120 |

Under FP16, Q-001 adds 9,024 bytes to all-4 and 5,952 bytes to all-8 versus treating all seven linears as eligible. The three budget differences are +8,256, +7,488 and +6,720 bytes respectively. Under FP32 the endpoint differences are +21,312 and +18,240 bytes, and budget differences +20,544, +19,776 and +19,008 bytes. The ordered ceilings admit 299, 2,510 and 4,017 of 4,096 bit assignments respectively. At each ceiling an assignment reaches it exactly. The policy comparison tolerance is 0.5%. No policy comparison ran.

## Verification and authorization

`.venv/bin/python -m pytest -q` passed 17 tests, including the original P0 audit, exact group costs for odd and partial groups, signed levels, rounding ties-to-even, saturation, zero-scale handling, restoration on exceptions, split and seal enforcement, CRC, preprocessing and padding controls. Nonzero scales that underflow to zero in FP16 fail closed. The signed minimum codes −8 and −128 are unused. [Test log](../runs/P0_Q001/pytest-frozen.log)

The full 377,617,425-byte checkpoint matched the official SHA256. The pinned model loaded with strict keys. Actual block 1 weights were quantized twice each at 4 and 8 bits without a forward pass. All six selected matrices changed. Each of the 12 gate weights and every other parameter stayed bit-for-bit fixed, and the entire model matched the immutable master after every restoration. Identical reapplication produced identical parameter fingerprints. The CPU preflight took **1.828 s** with **1147.5 MiB** peak process RSS. [Machine-readable P0 evidence and complete inventory](../runs/P0_Q001/p0.json) [Execution log](../runs/P0_Q001/p0-first.log)

The metadata audit still supports a **63 × 18** candidate grid. G0 remains unverified pending full audio validation and the study split lock. G5 remains open. The [frozen micro plan](../runs/P0_Q001/micro-method.md) authorizes 12 train/dev mic1 recordings and eight isolated variants only. **P1 may proceed under the existing approval.**
