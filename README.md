# Speech Representation Quantization Pilot

**Final decision: `NO_GO_CURRENT_DESIGN` (P2 G2 failed).** This repository documents a preregistered, falsification-first research pilot on quantized speech representations. It is an engineering and research record, **not a peer-reviewed publication**. The result rejects further work under this registered design; it does not disprove the broader research hypothesis.

## Motivation and research question

Speech encoders carry information useful for multiple downstream tasks. This pilot asked whether weight-only 4/8-bit quantization creates conflicting layer-level sensitivity for two tasks evaluated on the same VCTK recordings: closed-set prompt classification and closed-set speaker identification. A later stage would have tested whether simple precision-allocation policies leave a cost-matched worst-task gap.

The experiment was designed to stop early if the basic quantization effect was too small to justify layer-conflict analysis. That falsification gate failed at P2, so no layer-sensitivity sweep or allocation search was run. The exact hypotheses, endpoints, thresholds, and stop rules remain in the [protocol](docs/01_protocol.md) and [gate specification](docs/04_analysis_and_gates.md).

## Results at a glance

| Stage or gate | Outcome | Summary |
|---|---|---|
| Original P0 v1 | BLOCKED | WavLM has a seventh in-block linear gate matrix; eligible scope needed explicit amendment. |
| Amended P0, Q-001 | PASS | Verified 12 blocks, 72 eligible projection matrices, 12 excluded gates, model revision and byte-accounting implementation. |
| P1 smoke | PASS | 12 official-source mic1 clips; 128 model forwards; 4/8-bit smoke on blocks 1/4/8/12; no test reads. |
| Q-002 statistical preflight | PASS / ADOPTED | 18 focused tests; perfect 63×18 predictions remained 1.0 in all 2,000 crossed-weight replicates. The bounded four-class coverage simulation is approximate evidence only. |
| P2 G0: data integrity | PASS | 1,134 accepted cells; complete 63-speaker × 18-prompt grid; 680/227/227 train/dev/sealed-test split. |
| P2 G1: frozen probes | PASS | Dev macro-F1: speaker 0.902954, prompt 1.000000. Both cleared the registered baseline and majority margins. |
| P2 G2: uniform perturbation | **FAIL** | All-4-bit absolute dev drops: speaker 0.015432, prompt 0.000000; the gate required at least 0.03 on one task. All-8-bit drops were 0.000000 on both tasks and met its subcondition. |

The G2 failure was a scientific stop under the preregistered threshold, not a quantizer implementation failure. The [P2 report](reports/P2_G0_G1_G2.md) contains the exact metrics, checksums, gate conditions, resource measurements, and machine-readable status in [`P2_STATUS.json`](P2_STATUS.json).

## Method and implementation

- **Data:** official VCTK 0.92, mic1 only, one clip per speaker–prompt cell; deterministic complete grid and cell split. Prompt classification is a controlled closed-set content proxy, not ASR or phoneme recognition.
- **Encoder:** frozen Microsoft WavLM Base Plus, model and processor revision `4c66d4806a428f2e922ccfa1a962776e232d487b`. The checkpoint SHA256 is `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.
- **Probes:** separate train-fit standardization and multinomial logistic-regression heads; regularization selected on dev. The same mask-aware pooled representation and audio examples serve both tasks. Heads and encoder remain frozen for perturbation evaluation.
- **Quantizer:** symmetric groupwise, weight-only 4/8-bit fake quantization on six registered projection weights per each of 12 blocks. Q-001 explicitly excludes relative-position gates and other parameters. Reconstructed floating weights are used for model forwards.
- **Storage:** byte figures are calculated packed-storage estimates including group scales and fixed parameters. No packed inference kernel, deployment binary, or acceleration was measured.
- **Statistics:** Q-002 specifies positive-weight crossed-factor multiplier bootstrap resampling. Synthetic invariants passed, but the four-class Monte Carlo diagnostic does not establish exact coverage for this pilot's 18 prompt and 63 speaker labels.

The package is organized around data planning and integrity (`sqpilot/data.py`, `sqpilot/p2_data.py`), model and mask-aware features (`sqpilot/model.py`), frozen probes and gates (`sqpilot/p2_engine.py`), quantization and accounting (`sqpilot/quant.py`), crossed-factor statistics (`sqpilot/stats.py`), and stage CLIs (`sqpilot/cli.py`, `sqpilot/p2_cli.py`, `sqpilot/q002_cli.py`). Unit and integration tests are in [`tests/`](tests/).

## Scientific interpretation and limits

G2's all-4-bit drop did not reach the required threshold, and the prompt head was already at a dev ceiling of 1.0. H1 is not supported under the registered operational gate; H2/H3/H4 were not tested. G3 and G4 remain unassessed and G5 remains open. No P3/P4 stage was conducted and no sealed-test predictions were read.

Other limits include the six-projection-only quantization scope; approximate bootstrap coverage evidence; unverified audio parity for the metadata mirror (all P2 audio came from the official source); and duplicate checks limited to exact FLAC/decoded-PCM hashes plus mic1-only exclusion of the second microphone. The split tests within the known speaker and prompt sets, not unseen speakers or prompts. These limits and the final decision are detailed in the [research closeout](reports/RESEARCH_CLOSEOUT.md).

## Reproducibility

The recorded environment was Python 3.12.15 on Apple Silicon macOS, PyTorch 2.15.0, MPS available, CUDA unavailable. The exact observed package freeze is [`requirements-p0-observed.txt`](requirements-p0-observed.txt). Other platforms and full-run CPU behavior are not validated.

From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-p0-observed.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
python3.12 tools/check_package.py
shasum -a 256 -c docs/archive/v2/SHA256SUMS
```

The test suite uses synthetic fixtures for integrity, leakage, quantization, byte accounting, bootstrap invariants and cache replay. It does not require reading sealed predictions. To reproduce empirical P2, obtain VCTK through its [official source](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full), ensure the caps in [`configs/pilot.yaml`](configs/pilot.yaml) are respected, and use the pinned model reference in the config. **Do not download the complete 10.94 GB corpus.** The recorded P2 run used bounded official-source ranges and is already stopped at G2. Exact stage commands and resource records are listed in the [P2 report](reports/P2_G0_G1_G2.md); no P3/P4 command is part of this project’s permitted reproduction path.

### Data, model, and redistribution

VCTK 0.92 is released under CC BY 4.0; the repository includes attribution and scope details in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). WavLM weights are referenced by official model ID, revision and hash but are not included. Selected small JSON results, manifests and amendment diffs are published at their original `runs/` paths. Audio, model weights, feature caches, prediction/head arrays, full metadata caches and detailed execution logs remain local. The [P2 hash index](reports/P2_ARTIFACT_SHA256.json) records the original run inventory; omitted local artifacts are not in this public snapshot. Some immutable historical reports link to execution logs that contain machine-specific paths; those log files and their indexed hashes remain local. The exact P2-time README is preserved at [`README_P2.md`](README_P2.md) for its historical checksum. The project has no standalone reuse license.

The public GitHub repository is a single safe snapshot commit because the preserved local history contains machine-local email metadata; the original local history remains unchanged. Earlier commit identifiers in the stage status files are historical provenance and are not public commit ancestors.

## Protocol, reports, and amendments

- [Research closeout](reports/RESEARCH_CLOSEOUT.md)
- [P0 original blocker](reports/P0_PREFLIGHT.md) · [Q-001 amended P0](reports/P0_PREFLIGHT_Q001.md) · [P1 smoke](reports/P1_SMOKE.md)
- [P2 G0/G1/G2 report](reports/P2_G0_G1_G2.md) · [P2 machine status](P2_STATUS.json)
- [Q-001 quantization scope](docs/amendments/Q001_SCOPE.md) · [Q-002 bootstrap amendment](docs/amendments/Q002_BOOTSTRAP.md)
- [Protocol](docs/01_protocol.md) · [Data/model](docs/02_data_and_model.md) · [Quantization](docs/03_quantization_and_baselines.md) · [Analysis and gates](docs/04_analysis_and_gates.md) · [Novelty matrix](reports/P0_NOVELTY_MATRIX.md) · [Execution/artifacts](docs/06_execution_and_artifacts.md)

**Final decision: `NO_GO_CURRENT_DESIGN`.** This is a bounded pilot result and a project closeout, not a peer-reviewed publication and not a broad claim that task-conflict hypotheses are false.
