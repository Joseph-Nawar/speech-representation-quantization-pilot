# 05 — Known prior art and novelty checkpoint

**Critical warning:** Existing research already establishes task-specific losses under compression and multiple mixed-precision optimization mechanisms. The pilot is a *falsification test*, not evidence that the topic is novel.

## Core papers (read originals, not just abstracts)

1. **Efficient Speech Representation Learning with Low-Bit Quantization** (2023), https://arxiv.org/abs/2301.00652. Existing low-bit SSL speech encoder evaluation across SUPERB tasks; disproves any claim that differential task sensitivity is itself new.
2. **Effective and Efficient Mixed Precision Quantization of Speech Foundation Models** (ICASSP 2025), https://arxiv.org/abs/2501.03643. Mixed-precision search and training for speech; we cannot claim mixed-precision allocation is new.
3. **You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations** (2025), https://arxiv.org/abs/2511.06516. Task-specific representation guidance; cannot claim task sensitivity itself is novel.
4. **Beyond Activation Alignment: The Alignment-Diversity Tradeoff in Task-Aware LLM Quantization** (2026), https://arxiv.org/abs/2607.00908. Multi-objective task-aware precision selection, despite application to LLMs. A simple worst-task or aggregate score likely has close precedent.
5. **Toward Extremely Low-Bit and Multi-Precision Conformer and Speech Foundation Model Quantization** (2026), https://ieeexplore.ieee.org/abstract/document/11656827. Extremely low-bit and multi-precision speech compression.
6. **SUPERB** benchmark: https://huggingface.co/datasets/s3prl/superb ; official implementation https://github.com/s3prl/s3prl. VCTK paired-task experiment is a custom controlled diagnostic, **not** a SUPERB score.

## Before P2, collect a novelty matrix
For each paper: backbone; level of quantization; objective/sensitivity signal; one vs multiple tasks; label supervision; trained vs PTQ; hardware/size; code availability/license; empirical failures; strongest published comparator; whether their mechanism already handles H2/H3.

**Required additional targeted searches:** 2024–2026 multi-task post-training quantization, layer conflict between phonetic/speaker information, speech SSL quantization under multiple downstream probes, sample-efficient paired-task evaluations, and predictive power of isolated-layer sensitivity for joint mixed precision.

## Distinct contribution standard
At the end of the pilot, write a proposed 2–3 sentence method novelty claim **without** saying merely "use multitask loss," "use drift," "preserve representations," "prioritize sensitive layers," or "use a minimax objective". For each claimed technical primitive, identify the nearest existing method and one test where the methods differ. If this cannot be done, mark `G5=FAIL` and reject paper development regardless of positive pilot findings.

## Source/license audit
- Original VCTK data/license: https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full
- Possible non-official mirror metadata: https://huggingface.co/datasets/thelarryproject/vctk ; treat correctness and reuse as *unverified until checked*.
- WavLM checkpoint: https://huggingface.co/microsoft/wavlm-base-plus ; confirm model weights license + exact revision before download/use.
- Never commit redistribution-restricted raw audio to GitHub. Do not publish speaker-identifying predictions.
