# 04 — Statistics, sealed-test policy, and hard decision gates

## Metric calculation
Per task and condition:
- `macro_f1`: sklearn macro-F1 over fixed complete class label set, zero_division=0; separately report accuracy and per-class support.
- `loss_pp = 100*(f1_fp - f1_quantized)`; may be negative (improvement), never clip it in descriptive tables.
- `normalized_loss = (f1_fp - f1_q) / max(f1_fp - 1/K, 0.15)`; `R = max(0,normalized_loss_content,normalized_loss_speaker)` for bit policy selection.
- For the 12 isolated 4-bit block perturbations, Spearman correlation between content and speaker `loss_pp`; if too many exact ties/constant vectors, report undefined and fail H2 (do not coerce to 0).
- Top-4 most sensitive layer overlap; source of deterministic ties declared.
- Compare actual full-model mixed-precision policies at matched achieved encoded byte budgets (or use Pareto plot if mismatched).

## Uncertainty and sampling (Q-002; protocol v3)
- **Paired evaluation** uses exactly the same sample IDs across FP and quantization variants, and every paired condition uses the same cell weights in a given replicate.
- Use positive-weight crossed-factor multiplier resampling on the observed speaker×prompt grid. In each replicate independently draw `a_s ~ Gamma(shape=1, scale=1)` for every speaker and `b_p ~ Gamma(shape=1, scale=1)` for every prompt, then weight each observed cell by `w_sp = a_s * b_p`. Normalize each factor's weights to mean 1 within the replicate if used; this common positive rescaling does not change F1.
- Calculate weighted confusion counts and weighted per-class F1 over the fixed, complete label set. Every class with original true-label support must retain strictly positive weighted support in every replicate. Use `zero_division=0` only for a genuinely undefined precision/recall contribution; never omit a predefined class. Apply the same paired weights to score differences, rank correlations, and policy differences.
- Generate deterministic named RNG substreams for each data partition from root seed 20261124 and SHA256-bound study ID, split hash, bootstrap specification hash, and partition name. Dev and test streams are distinct; test outcomes remain sealed until P4. No sequential optional stopping or seed shopping.
- Use **2,000** replicates and percentile **95%** intervals. Report effect estimates and intervals; descriptive one-sided proportions for gates; no unsupported p-value claims.
- This is an **approximate positive-weight multiway bootstrap**, inspired by Owen & Eckles (2012), not a proven exact 95% frequentist coverage procedure. It preserves the observed label classes but does not test generalization to held-out speakers or novel prompts. Report the bounded synthetic stability/coverage diagnostic and its limits; do not tune the interval to pilot outcomes.
- Preserve a synthetic regression test of the prior multinomial deletion rule (perfect predictions scored below 1 when resampled classes disappear), alongside perfect-classifier, paired-zero, independent confusion-reference, support, and reproducibility checks. If a required invariant fails or the synthetic diagnostic leaves material validity doubts, stop before empirical P2 metrics with `BLOCKED_STATISTICS`.
- For multiple candidate budgets/contrasts, use a **single predeclared primary budget** `f=0.50`. Others are sensitivity checks, not independent significance opportunities. No choosing the most favorable budget post hoc.
- Primary test comparisons focus on `f=0.50`, `minimax` vs fixed dev-selected search diagnostic, and the isolated 4-bit sensitivities.

## Hard gates
These are **decision-policy engineering thresholds** selected before results, not universal scientific significance criteria. Distinguish P1 engineering checks (no formal outcome estimates) from P2 dev gating and P4 test confirmation.

| Gate | Requirement | If not met |
|---|---|---|
| G0 Data integrity | >=40 speakers ×16 shared prompts; complete mic1 grid; no duplicate (speaker,prompt); valid split supports; verified provenance/license | NO-GO for this design |
| G1 Feasible baseline | Both FP encoder dev macro-F1 >=0.55 and >= majority-class macro-F1 +0.20 | NO-GO for this design |
| G2 Perturbation meaningful | On dev, uniform all-4 decreases macro-F1 by >=0.03 on >=1 task; uniform all-8 decreases macro-F1 by <=0.05 on *each* task; all quantizer tests pass | NO-GO / implementation failure diagnosis |
| G3 Robust conflict | On dev, 4-bit per-layer Spearman <=0.35; top-4 overlap <=2; at least 3 layers have absolute cross-task loss difference >=3 F1 percentage points; 95% two-way-bootstrap upper CI for Spearman <0.70 | NO-GO for task-conflict hypothesis |
| G4 Unhandled policy failure | At primary f=0.50, fixed dev-selected search envelope improves worst-task `R` by >=0.02 **absolute normalized-headroom units** over minimax on dev, and the signed gain has 95% two-way-bootstrap test CI with lower bound >0 | If not: NO-GO for method-development investment |
| G5 Novelty | Manually review closest published methods and specify an *algorithmic* differentiator; document a limitation those methods don't already solve | No authorization for paper method even if G0–G4 pass |

### Important cautions about G4
The search-based envelope has **more validation search budget** than the deterministic minimax policy, so G4 only diagnoses headroom, not a fair algorithmic improvement. The sealed test comparison is deliberately hard. Even if positive, later work must make a new method and compare with equal search budgets.

### G3/G4 on dev vs test
- G3 is computed on dev; report its estimates and CIs on sealed test **without modifying thresholds or methods**. If test contradicts G3 (Spearman >0.70, top-4 overlap >2, or 95% bootstrap CI includes strongly aligned rankings), mark `NOT_REPLICATED` and use NO-GO.
- G4 requires held-out positive confirmation on primary budget; if CI includes zero, the *decision* is NO-GO for further investment under the current protocol, with scientific inference stated as inconclusive rather than evidence of no effect.
- G0/G1/G2 are primarily technical/dev gates; final test quality and intervention sizes reported transparently.

## Verdict definitions
- **GO_TO_METHOD_DESIGN**: G0–G4 pass and G3 not contradicted on sealed test, *plus* a separately documented G5 novelty case. This DOES NOT authorize a full paper.
- **NO_GO_CURRENT_DESIGN**: Any mandatory gate fails, evidence is weak/inconclusive at the precommitted resource limit, or a published existing method already resolves the problem.
- **BLOCKED_INVALID**: Missing/corrupt data, unverified legality, metric leakage, incorrect quantizer, unsupported hardware behavior. Fix and restart before judging science. Never portray as a scientific NO-GO.

Do not replace a borderline result with new thresholds after seeing data. A fresh scientifically motivated amendment means a new study ID, not editing the original verdict.

## Test lock before P4
Create immutable `runs/<run_id>/test_lock.json` recording: git commit, dataset/grid/split/model/probe SHA256, quantizer SHA/version, predeclared baselines and bitmaps at each budget, metric and CI scripts, source manifests, resource usage, analysis thresholds, dev outputs, exact hypotheses and stop rules. No test labels or predictions in policy selection files.

## Required final plots (with confidence intervals where appropriate)
1. Paired FP, uniform4, uniform8 performance on both tasks.
2. 12-layer paired sensitivity chart for each task (raw percentage points, bootstrap CIs).
3. Scatter content vs speaker sensitivity, Spearman and top-4 overlap.
4. Mixed-policy Pareto: worst-task loss vs **encoded bytes**, all baselines and random distributions.
5. Selected primary budget 0.50 held-out performance comparison, per-task F1, normalized worst-task loss, CI.
6. Negative controls and resource/runtime/memory logs.
