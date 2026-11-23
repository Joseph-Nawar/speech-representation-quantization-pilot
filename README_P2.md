# Speech quantization research: diagnostic pilot handoff

**Status:** Pre-registered diagnostic protocol, **not** an approved full research paper.
**Prepared:** 2026-11-22.
**Owner:** Researcher + Codex implementation assistant.

The current study is protocol v3 (`speech-quant-diagnostic-20261122-v3`), descending from v2 after Q-002's pre-outcome correction to the crossed-cluster uncertainty calculation. Q-002 was adopted after the required synthetic statistical validation, before any empirical P2 metrics. The exact v2 source snapshot is preserved under `docs/archive/v2/`. The approved P2 scope ends after G0/G1/G2; P3 remains unapproved.

## The question
Can a shared self-supervised speech encoder preserve spoken-content and speaker information equally under weight-only quantization? If not, is the conflict **real, statistically supported, and inadequately handled** by existing simple mixed-precision policies?

This is not a new-algorithm project yet. A pilot that finds no conflict or no unaddressed headroom is a successful early rejection.

## Files: read in this order

1. `docs/01_protocol.md` — research questions, design, exclusion rules, primary endpoints.
2. `docs/02_data_and_model.md` — same-audio corpus, speaker×prompt factorial split, training of frozen probes.
3. `docs/03_quantization_and_baselines.md` — exactly what is quantized, cost accounting, decision policies.
4. `docs/04_analysis_and_gates.md` — inference, thresholds, honest GO/NO-GO policy.
5. `docs/05_literature_and_novelty.md` — strongest overlaps, novelty requirements and links.
6. `docs/06_execution_and_artifacts.md` — commands, implementation phases, tests, outputs, exit criteria.
7. `configs/pilot.yaml` — locked defaults (changes require a protocol amendment).
8. `AGENTS.md` — repo-level rules for coding agents.

## How to begin with Codex

Place this entire folder in a **new, separate repository**. Open Codex in that folder and paste `prompts/CODEX_START.md`. Codex is asked to implement **P0/P1 only** and return the preflight report before expensive runs. After reviewing that report, use `prompts/CODEX_AFTER_PREFLIGHT.md` for P2–P4 if warranted.

## Why two tasks on the same recordings?

Earlier ideas compared keyword spotting on one corpus to speaker identification on another. That confounds *task* with *dataset/recording conditions*. The primary pilot instead samples **VCTK shared utterance prompts** across speakers; both task labels belong to the same clip. The content task is **closed-set prompt classification**, not phoneme recognition, ASR, or keyword spotting. Generalization to those tasks would need additional evidence.

## Scope and honesty

- **No new human annotation or review** is required. Automated scores use existing VCTK speaker IDs and prompt IDs.
- **No large model training.** Encoder frozen; small linear probes trained on full-precision features.
- **No speedup claims from fake quantization.** We measure task sensitivity and model *encoded* size; hardware gains require deployment proof.
- **GO means GO to develop a distinct method**, never an assertion of manuscript novelty or future publication.
- **NO-GO is acceptable**; never optimize the experiment until it yields a positive result.

## Primary sources

- VCTK official Edinburgh corpus: https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full
- WavLM Base Plus model: https://huggingface.co/microsoft/wavlm-base-plus
- SUPERB task background: https://huggingface.co/datasets/s3prl/superb
- Speech mixed-precision 2025: https://arxiv.org/abs/2501.03643
- LLM multi-objective task-aware quantization 2026: https://arxiv.org/abs/2607.00908
