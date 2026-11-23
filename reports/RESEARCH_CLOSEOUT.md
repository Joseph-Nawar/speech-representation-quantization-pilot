# Research closeout

**Study:** `speech-quant-diagnostic-20261122-v3` · protocol v3 · amendments Q-001 and Q-002 adopted

**Closeout date:** 2026-11-23

**Final decision:** **`NO_GO_CURRENT_DESIGN`** after P2 G2 failed. This is a closed research pilot, not a peer-reviewed publication.

## Research question and hypotheses

The pilot asked whether 4/8-bit weight-only perturbations to one shared self-supervised speech encoder produce conflicting layer-level sensitivities for (A) closed-set VCTK prompt classification and (B) closed-set speaker identification on the same recordings, and whether simple cost-matched multi-task allocation policies leave practically relevant headroom.

The preregistered hypotheses were: **H1**, 4-bit perturbation measurably harms at least one qualified task while 8-bit does not catastrophically harm either; **H2**, isolated-layer losses across the two tasks are not strongly aligned and show consequential, resampling-robust differences; **H3**, simple allocation policies leave a useful fixed-budget gap against a feasible alternative on sealed data; and **H4**, a distinct mechanism could address any observed failure better than existing task-aware methods. H4 requires a separate novelty case and is not established by H1–H3.

The proposed contribution was a falsification-first, same-audio diagnostic of cross-task sensitivity and shared precision allocation, with a possible later method-design question. It was not a new quantization algorithm. Existing work already covers speech low-bit quantization, 2025 speech mixed precision, task-derived sensitivity and mixed precision, and 2026 multi-objective task-aware quantization. The P0 novelty matrix identifies TASA and other recent work as close comparisons. A generic claim about task-aware allocation, layer sensitivity, or a minimax objective would not be novel. No paper novelty claim was made; G5 remains open.

## Method and implementation

The locked design used the official VCTK 0.92 source, one `mic1` recording per speaker–prompt cell, a deterministic complete grid and cell split, a frozen `microsoft/wavlm-base-plus` encoder, and separate frozen train-fit `StandardScaler` plus multinomial logistic probes for prompt and speaker labels. The encoder revision was `4c66d4806a428f2e922ccfa1a962776e232d487b`; checkpoint SHA256 was `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.

Q-001 fixed the quantized scope at six named projection matrices in each of 12 blocks (72 matrices total), excluding relative-position gates and all other non-allowlisted parameters. The simulator applies deterministic symmetric groupwise 4/8-bit fake quantization and reconstructs floating weights for ordinary PyTorch inference. It does not provide packed-kernel runtime or acceleration evidence. Encoded-byte figures are calculated storage estimates using the registered group and scale format.

Q-002 replaced a multinomial deletion bootstrap with positive Gamma-weighted crossed-factor resampling before P2 outcomes. Its perfect-classifier invariant and other synthetic checks passed; its bounded coverage simulation is approximate evidence for a four-class data-generating process, not a coverage guarantee for this pilot.

## Gate outcomes

| Stage/gate | Result | Evidence |
|---|---|---|
| Original P0 v1 | BLOCKED | The WavLM block contained a seventh linear gate matrix, making eligible scope ambiguous. The report and status are retained. |
| Amended P0 / Q-001 | PASS | Exact 72/12 eligible/excluded inventory, model and runtime checks, source metadata audit, byte-accounting tests. |
| P1 | PASS | 12 clips, 128 WavLM forward passes, blocks 1/4/8/12 at 4/8 bit, audio integrity, restoration, padding and smoke-probe mechanics. No test reads. |
| P2 Q-002 validation | PASS / ADOPTED | 18 focused statistical tests; a perfect 63×18 classifier scored 1.0 in all 2,000 replicates. Four-class bounded coverage was 0.928–0.976 across scenarios, with stated limits. |
| G0 | PASS | 1,134/1,134 audio cells accepted; 63×18 complete mic1 grid; split 680/227/227; all registered support minima passed. No rejected files. |
| G1 | PASS | Speaker dev macro-F1 0.902954; prompt dev macro-F1 1.000000; both exceeded preregistered baseline and majority margins. |
| G2 | **FAIL** | Uniform 4-bit speaker drop was 0.015432, below the required 0.03 drop on at least one task. Uniform 8-bit drops were 0 on both tasks and passed that subcondition. |
| G3 / G4 / G5 | NOT ASSESSED / NOT ASSESSED / OPEN | Execution stopped at the failed G2 gate. No layer sensitivity sweep, allocation search, test lock, or novelty approval followed. |

For G2, FP dev macro-F1 was speaker 0.902954 and prompt 1.000000; all-4 was 0.887521 and 1.000000; all-8 was 0.902954 and 1.000000. The observed absolute all-4 drops were 0.015432 speaker and 0 prompt. The protocol required an all-4 drop of at least 0.03 on one task. The implementation, model restoration, fixed heads, replay, and cost-accounting checks passed; this was a preregistered scientific gate failure, not a quantizer implementation failure. The decision is therefore `NO_GO_CURRENT_DESIGN`.

The calculated model-only packed estimates were B4 = 64,016,096 bytes, B(.50) = 85,249,760 bytes, and B8 = 106,483,424 bytes. They reconcile with the registered accountant; they are neither serialized deployment binaries nor evidence of inference acceleration.

## What was demonstrated

- The approved official-source audio workflow produced a complete, integrity-checked 63×18 mic1 grid and valid deterministic split within the declared resource caps.
- On this split, the pinned WavLM model supported strong frozen development probes for both labels.
- The registered groupwise quantizer altered only the 72 allowlisted matrices, restored the master model, preserved probe artifacts, replayed cached predictions, and reconciled calculated byte costs.
- Under the registered uniform 4-bit perturbation, the speaker-probe drop was below the preregistered minimum, while the prompt probe remained at its development ceiling. That result did not satisfy H1's operational gate, so the current design stopped before testing the layer-conflict question.

The result does **not** show that broader task conflict is absent, that the hypotheses are disproven across models or datasets, or that another precision allocation could not help. H1 is not supported under this operational gate; H2, H3, and H4 were untested. G3 and G4 were not assessed, and G5 remains open. No P3 or P4 work was conducted; no sealed-test predictions were read.

## Limitations and unresolved risks

1. **Prompt-classification ceiling:** FP prompt dev macro-F1 was 1.0 and remained 1.0 under both quantized conditions. The task is closed-set prompt-ID classification on known prompts and speakers, not ASR, phoneme recognition, or general content preservation. The ceiling leaves no observed prompt-task headroom on this split.
2. **Limited quantization scope:** only the six Q-001 projection weights per block were quantized. Relative-position gates, frontend, other modules, biases, normalization parameters, and probes stayed at reference precision. Results do not characterize fully quantized WavLM or deployment behavior.
3. **Approximate bootstrap evidence:** Q-002's multiplier bootstrap passed its preregistered synthetic tests, but coverage diagnostics used four classes and a specified synthetic process. They do not establish exact nominal coverage for 18 prompt and 63 speaker classes or generalization to unseen factors.
4. **Mirror-audio parity:** mirror metadata was cross-audited against official-source evidence, while all P2 audio was fetched from the official source. Audio-level parity of the Hugging Face mirror was not tested.
5. **Duplicate detection:** exact FLAC and decoded-PCM hashes found no duplicate cells or cross-split duplicate payloads; the design also excluded `mic2`. No general fuzzy acoustic duplicate threshold was preregistered, so near-duplicate detection beyond these controls remains unresolved.
6. **Generalization and backend:** the design concerns within-known-speaker and within-known-prompt splits, one encoder, and this MPS/FP16 environment. Cross-corpus, unseen-speaker/prompt, cross-machine determinism, and CPU behavior for the full run remain unestablished.
7. **Resource and artifact limits:** P2 stayed within its transfer, disk, and free-space caps. Model weights, audio, sealed test audio, feature caches, fitted-head binaries, per-example prediction arrays, full metadata caches, and execution logs remain local. Selected small result JSON, manifests, and amendment diffs are included. Some immutable historical reports link to excluded execution logs; those logs contain machine-specific paths, so they were not published. Their original paths and hashes remain recorded in the reports/status manifests, and the local copies were preserved.

## Preserved evidence and interpretation

The original P0/P1 reports, Q-001 and Q-002 amendment documents, protocol/configuration, P2 report/status, P0/P1 status files, and historical checksum indexes are retained unchanged. The P2-time README is preserved at [`README_P2.md`](../README_P2.md), matching the README SHA256 recorded in the P2 artifact index (`b215df4060be3019ad4adae93c187abffb58e6c1f2e530a8fbbaf70a1175401c`). The P2 index records both published safe summary artifacts and intentionally omitted local files such as feature caches and per-example prediction arrays; see the artifact-scope note in the root README.

The pilot is an engineering and scientific screening record, not a peer-reviewed paper or a publication. Its negative result is specific to the registered design and decision threshold. It neither proves nor disproves the broader research hypothesis.

## Closeout integrity checks

- The final source snapshot contains **86 tracked files (1,109,247 bytes)**. A scan found no credentials, API keys, email addresses, machine-specific absolute paths, oversized files, audio, model weights, feature arrays, or restricted paper content in the release tree. VCTK is attributed under the license shown on the official dataset record; the model is referenced at its official pinned revision and its weights are not included.
- The original local Git history has 10 commits; 3 commit-identity rows contain machine-local email metadata. The public repository uses a single safe snapshot commit to avoid disclosing that metadata. The local `main` history and experimental evidence remain intact. Historical commit identifiers in the status files remain provenance records but are not ancestors of the public snapshot.
- All 57 automated tests passed with one pre-existing Torch attention-mask deprecation warning. Package handoff check, archived v2 checksum verification, frozen P0/P1/P2 report/status hashes, the archived P2 README hash, and `git diff --check` for authored closeout files passed.
- The public snapshot resolves 53 of 56 Markdown local targets; the other three are immutable links to local P0/P1 execution logs that contain machine-specific paths and remain intentionally unpublished. The status/hash manifests record their checksums. There are no hash mismatches among the published entries: 7 of 42 P0/P1-indexed local artifacts and 27 of 47 P2-indexed artifacts are safely published; omitted entries are preserved locally and include path-bearing logs, model/data-derived arrays, and full metadata artifacts.
- The indexed P0/P1 and P2 local artifacts were left unchanged. Eleven trailing-whitespace warnings occur within the two frozen generated protocol-diff files; those byte-exact amendment artifacts were preserved rather than edited.
