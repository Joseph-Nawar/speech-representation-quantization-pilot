# 06 — Codex build plan, execution order, and artifacts

## New-repository structure (to be implemented)
```
AGENTS.md, README.md, docs/, configs/, prompts/
src/sqpilot/{data.py, features.py, probes.py, quant.py, costs.py, policies.py, metrics.py, stats.py, reports.py, cli.py}
tests/{test_grid.py,test_splits.py,test_quant.py,test_costs.py,test_metrics.py,test_policies.py,test_bootstrap.py,test_no_test_leakage.py}
manifests/        # hashed ID/split files, no audio
runs/<run_id>/    # gitignored local artifacts; derived reports may be exported separately
cache/ data/      # gitignored audio, models, embedding caches
```

## P0 — Scientific & environment preflight (Codex FIRST deliverable)
1. Inspect local OS/chip, RAM, free disk, installed Python, PyTorch MPS support, torchaudio/librosa availability; **read-only** detection first.
2. Confirm actual dataset mirror fields, licensing, sample access, mic1, common prompts; metadata-only. Do not download the official 10.94GB zip.
3. Pin WavLM and processor revisions; verify tensor/module naming and eligible projection matrices, 12-layer assumption.
4. Expand literature novelty table; check for any new directly matching 2026 work. Document source links.
5. Implement skeleton CLI and minimal unit tests; generate `reports/P0_PREFLIGHT.md` with PASS/FAIL/UNVERIFIED and next actions.
6. Stop and return results. No bulk inference.

Suggested interfaces (Codex may refine but must preserve observable contract):
```
python -m sqpilot.cli inspect-env --out runs/P0/env.json
python -m sqpilot.cli inspect-source --config configs/pilot.yaml --out runs/P0/source.json
python -m sqpilot.cli inspect-model --config configs/pilot.yaml --out runs/P0/model.json
pytest -q
```

## P1 — Micro smoke test (only after P0 pass)
1. Materialize <=128 mic1 clips from the *predefined* candidate grid, verifying clips, labels and hashes; cap bytes/disk. Generate deterministic selection ID list.
2. Decode/resample, obtain pooled hidden states, verify padding invariance and deterministic FP outputs.
3. Implement symmetric groupwise fake quant + reference tests + model weight isolation checks; perturb blocks 1, 4, 8, 12 only for smoke, b=4/8.
4. Validate end-to-end feature extraction / classifier training mechanics with train/dev-only microdata; **do not report micro probe scores as scientific evidence**.
5. Record processed sample count, actual disk used, worst-case memory, runtime and device. Stop with `reports/P1_SMOKE.md`.

## P2 — Full protocol lock (owner authorization required)
1. Build exact balanced grid and splits; validate cell-only disjointness and per-class minimums; finalize data/model revisions and SHA256s.
2. Fit and freeze full-precision task probes; compute G1 with dev only. If fail, stop; no algorithm changes just to pass.
3. Prove quantizer/cost correctness with unit/integration tests. Measure uniform4/8 dev intervention, evaluate G2.
4. Freeze `protocol_lock.json` and declare immutable test policy. No test outcomes or model predictions read (test class counts for split-integrity checks are allowed).

## P3 — Dev-only diagnostic (after G0/G1/G2 pass)
1. Quantize each of 12 blocks **one at a time** to 4/8-bit, leaving others at FP. Score both heads on exact matched dev audio.
2. Compute per-layer losses, paired bootstraps and G3. If fail, stop with NO-GO.
3. Enumerate admissible `{4,8}` block maps per budget and implement the 7 allocation policies. Ensure cost matching.
4. Run dev performance for each policy and predeclared random configurations. Form test candidate bitmaps **once** and freeze them.
5. Evaluate preliminary G4 on dev. If inadequate, stop; do not read test.
6. Build `test_lock.json` and `reports/P3_DEV_FINDINGS.md`; request owner review for test unlock.

## P4 — Sealed-test confirmation (explicit owner authorization)
1. Verify hashes and lock before reading held-out test labels; no test-driven code changes.
2. Evaluate full model bitmaps + individual-layer 4bit effects for all fixed test examples; cache predictions with hashes.
3. Compute task metrics, two-way bootstrap CIs, rank stability, G3 replication, G4 effect and missingness.
4. Generate standardized Markdown, JSON, CSV tables, plots, risk log, complete evidence index and decision JSON.
5. If G0–G4 pass, independently conduct G5 novelty assessment. Else document NO-GO / BLOCKED with precise reason.

## Definition of done (for entire diagnostic)
- Every command is reproducible from a clean environment with documented dependency pinning and manifests.
- Unit tests, leakage tests, caching tests, groupwise quant exactness tests, and budget selection tests pass.
- Report both positive and negative results, including failed trials.
- Report distinguishes observations vs statistical uncertainty vs claims of novelty.
- Clear single `verdict`: `GO_TO_METHOD_DESIGN`, `NO_GO_CURRENT_DESIGN`, or `BLOCKED_INVALID`.
- No manuscript, no paper novelty declaration, no remote push unless separately requested.
