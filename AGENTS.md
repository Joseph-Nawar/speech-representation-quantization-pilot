# Instructions for coding agents

## Research integrity is the highest priority
- Treat the files in `docs/` and `configs/pilot.yaml` as the protocol contract. Do not quietly change task definitions, dataset splits, model, bit widths, evaluation metrics, gate values, or direction of improvement.
- **Stop and document an amendment request** if implementation requires a scientific change. Never silently substitute a different dataset or head architecture.
- No test-set inspection or tuning before the full evaluation plan, allocator choices and code commit are frozen. A single sealed test read is allowed in P4.
- Never write public names/identities of speakers or infer identities. Use corpus-provided anonymized IDs only.
- Preserve all prior results and failure logs; never delete negative results. No cherry-picked plots.
- Use local relative paths, no credentials, no raw audio in git or shared artifact bundles.
- Record exact SHA256 manifests: model revision, downloaded artifacts, dataset sample IDs, checkpoints, head weights, quantizer configuration, code commit, run seeds, environment.
- Code for Mac Apple Silicon should have an MPS or CPU path with automatic device reporting; no false assumption of CUDA availability. Flag MPS/CPU nondeterminism.
- No whole-dataset downloads without an explicit user-approved resource override. Prefer bounded extraction from an appropriate published mirror; audit that it matches the official source.
- Do not launch long model sweeps before finishing P0/P1 and returning the required report.
- Do not claim quantitative runtime or true compressed deployment size from simulated (dequantized) weights.
- Fail closed on invalid data, missing samples, changed manifest, leakage, stale caches, metric errors, or ambiguous status.

## Engineering
- Python >=3.11; keep dependency versions in a committed lock or exact environment manifest after preflight.
- Make scripts `python -m ...` runnable, modular, with resumable checkpoints, stable IDs, clear exception messages, unit/integration tests.
- File I/O must be atomic for manifests/results. Store large data, models, and private metadata under gitignored `data/`, `cache/`, `runs/`.
- Each stage produces a machine-readable JSON and a short Markdown report. Log wall time, peak RSS/unified memory when available, actual processed clip count and device.
- Use non-interactive deterministic CLI; no mandatory web apps or manual labeling interfaces.
- No force push or publication. Do not create GitHub repositories or remote services without explicit instruction.
