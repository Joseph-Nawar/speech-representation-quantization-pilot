You are the implementation lead for a *pre-registered, falsification-first AI research pilot* on quantized speech representations. Work in this NEW repository. This request is **for P0 and P1 only**; DO NOT run full experiments, download a complete 10.94GB corpus, read test labels, or start algorithm development.

First read `AGENTS.md`, `README.md`, `docs/01_protocol.md` through `docs/06_execution_and_artifacts.md`, `docs/05_literature_and_novelty.md`, and `configs/pilot.yaml`. Treat them as binding source of truth. Do not silently revise thresholds or scientific definitions. If the experiment cannot be implemented as written, record the issue and stop with an amendment proposal rather than improvising.

Do this in order:
1. Check host hardware, RAM, free disk, Python/Torch/MPS, and runtime options; do not presume CUDA.
2. Inspect the **official** VCTK source and any candidate Hugging Face mirror for verified license, recording metadata, `mic1`, prompt IDs, and whether we can construct a balanced >=40-speaker × >=16-shared-prompt complete grid **without downloading the whole corpus**. No need to download audio until the source passes. Record source revision and source/mirror differences.
3. Pin WavLM Base Plus model/processor revisions, discover exact transformer block/matrix names, estimate eligible and excluded bytes. Limit model downloads to what is essential.
4. Extend the closest-work novelty matrix, particularly 2025 speech mixed precision and 2026 multi-objective quantization; flag any paper rendering this pilot moot.
5. Implement a modest, testable Python package/CLI following `docs/06_execution_and_artifacts.md`, with dataset integrity checks, deterministic grid/split logic, mask-aware features, frozen probe construction, groupwise symmetric quantization simulator, encoded byte accountant, tests, staged artifacts and hashes.
6. Run P0 preflight checks. If P0 passes and resources permit without exceeding `configs/pilot.yaml`, run P1 **micro** smoke on <=128 clips and blocks 1/4/8/12 only. Stop after P1 even if everything works. Do **not** run P2–P4.
7. Create `reports/P0_PREFLIGHT.md`, `reports/P1_SMOKE.md` if applicable, and a `P0_P1_STATUS.json` with PASS/FAIL/BLOCKED, tested commands, hardware/resource observations, exact revisions, hashes, blocking discrepancies, and proposed amendments.

Engineering requirements: implement deterministic and resumable stages; no secrets or raw audio in git; explicit failed assertions for duplicates/leakage and changing manifests; no false inference-acceleration claims from simulated quantization; unit tests for exact layer coverage, 4/8-bit signed levels, model restoration, byte budgets, split validity, no test leakage, and cached vs uncached predictions.

Return a compact but specific report: (a) what you built, (b) what ran, (c) whether dataset/probe/model and quantizer are technically feasible, (d) any violated assumptions, (e) exact P0/P1 commands/logs/results, and (f) **STOP — ready for researcher review**, or **STOP — blocked**, with reason. Do not request permission to skip checks or do an unrequested larger run. Do not create/push a remote repo.
