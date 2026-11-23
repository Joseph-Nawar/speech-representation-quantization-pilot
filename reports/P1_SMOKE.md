# P1 bounded engineering smoke

**PASS · Q-001 · 2026-11-22**

Study `speech-quant-diagnostic-20261122-v2`. The predeclared [micro method](../runs/P0_Q001/micro-method.md) selected 12 official VCTK `mic1` clips using a metadata-only, hash-based split preview. Both tasks use the same 8 train and 4 dev waveforms, with four classes and two train and one dev clip per class. The planned transfer uses 3,453,180 bytes of official ZIP member ranges, under the 16 MiB engineering cap. The user-approved ceiling was 128 clips. [Frozen selection](../runs/P0_Q001/micro-plan.json)

## Observations

| Check or resource | Observed |
|---|---|
| Official FLAC members materialized | 12 |
| HTTP range bytes transferred in this invocation | 3,453,180 |
| Local FLAC bytes | 3,452,459 |
| Passes through the transformer stack, including controls | 128 |
| Wall time for P1 | 41.997 s |
| Peak process RSS | 827.6 MiB |
| Maximum sampled MPS driver allocation | 1,260.3 MiB |
| Minimum free disk | 99.22 GiB |
| Reference | mps/float16 |
| Audio integrity | PASS |
| Smoke probes | PASS |
| Test audio reads | 0 |

The allocation figure samples the Torch MPS driver and process peak RSS is a separate host observation. Whole-host unified-memory peak was unavailable. Time and memory describe this engineering run, which includes verification and floating fake quantization. The filesystem reported 3,392 KiB allocated in the audio directory and 480 KiB in the smoke artifact directory before the final test log.

All 12 decoded clips passed official ZIP path and CRC checks, source channel/rate validation, finite and non-silent checks, and the inclusive 1–10 s duration gate. Observed duration range was 5.231–7.802 s. Official transcript hashes agree by prompt. No duplicate payloads or decoded waveforms were detected. Independently repeating the mono 48→16 kHz resampling produced identical values for each clip. This is local support for the candidate grid's feasibility. G0 still requires the remaining recordings and the full split lock.

## Backend and numerical controls

- **cpu/float16 PASS**. Maximum absolute differences where observed were cross backend 0.00040182471, repeat 0, sample padding 0, transformer padding 0 0, transformer padding 5 0.00022482872.
- **mps/float16 PASS**. Maximum absolute differences where observed were repeat 0, sample padding 0, transformer padding 0 0, transformer padding 5 0.

The run used mps/float16 for every reported condition. FP32 fallback was unneeded. Group normalization aggregates over waveform time, so each waveform was trimmed with the processor mask before the frontend. A separate control injected masked frames after the frontend and exercised all transformer layers. Pooling used the equal-weight mean of the last four hidden states and the valid-frame mask.

The pinned Torch runtime emitted a deprecation warning about the attention mask types. Both FP16 backends passed their padding checks in this run. Repeatability was assessed on this machine under the recorded tolerances. Cross-machine determinism remains unverified.

## Isolated interventions

| Condition | Maximum replay difference | Exact weight restoration |
|---|---:|---|
| block01-4bit | 0 | Pass |
| block01-8bit | 0 | Pass |
| block04-4bit | 0 | Pass |
| block04-8bit | 0 | Pass |
| block08-4bit | 0 | Pass |
| block08-8bit | 0 | Pass |
| block12-4bit | 0 | Pass |
| block12-8bit | 0 | Pass |
| reference | 0 | Reference |

The two probes used train-only standardization and class-balanced multinomial logistic regression, L-BFGS, fixed C=1.0 and tolerance 1e-6. They converged in 19 and 30 iterations. Both scalers and heads were frozen during the perturbations. Stored numeric probe arrays account for 86,080 bytes separately from the model estimates. No probe quality or hypothesis score is reported.

Exactly six expected matrices were changed per condition. All 12 relative-position gates and all other parameters retained their reference bytes. Byte-identical reapplication and final restored reference prediction checks passed. The restored reference feature differed by at most 0. Cached and recomputed prediction checks passed. Logs contain hashes for the manifest, media and preprocessed samples, pinned checkpoint, feature arrays, trained head and scaler arrays, predictions and audited model conditions.

## Commands and remaining scientific risks

```sh
.venv/bin/python -m sqpilot.cli plan --out runs/P0_Q001/micro-plan.json
.venv/bin/python -m sqpilot.cli p0 --out runs/P0_Q001/p0.json
.venv/bin/python -m sqpilot.cli p1 --out runs/P1_Q001/p1.json
.venv/bin/python -m pytest -q
```

The final full suite passed 17 tests. [Post-smoke test log](../runs/P1_Q001/pytest-after.log)

See the [amended P0 preflight](P0_PREFLIGHT_Q001.md) and [combined status](../P0_P1_STATUS.json) for actual command outcomes and hashes. The [P1 machine record](../runs/P1_Q001/p1.json) and [execution log](../runs/P1_Q001/p1-first.log) are local evidence when P1 has run.

G0 and G5 remain open. The full candidate grid was not validated, probe qualification and task trade-offs were not measured, and the literature has close precedents for multi-objective allocation. The result concerns known speakers and prompts and the six-projection intervention. A later held-out evaluation requires the study lock and explicit authorization. P2 requires separate researcher approval.

**STOP — ready for researcher review**
