# 07 — Stage report templates

## P0/P1 stage report
```
# P0/P1 status
Date | stage | Git commit | config SHA | hardware | Python/torch versions
Data source/revision/license | metadata row count | mic1 availability | grid size candidate
Model and processor revision | actual Transformer blocks/target matrices
Disk available/used | memory measurement | MPS or CPU | actual sample count
Tests: pass/fail + commands
Changed/ambiguous protocol assumptions: NONE or table with proposed amendment
Gate G0: PASS / FAIL / UNVERIFIED
Artifacts (relative paths + SHA)
Next permissible action: STOP / AUTHORIZE P2 / FIX BLOCKER
```

## Dev stage report
```
Frozen dataset and split hashes; probe/head hashes; config/quantizer hashes
Class counts and train/dev/test support table (do not open test labels)
Clean dev macro-F1, majority baseline, G1 verdict
Quantizer validation results, encoded bytes, G2 verdict
Layer sensitivity table content/speaker 1..12, Spearman/top4 overlap/G3 with CIs
Policy bits, costs, objective raw/normalized, random baseline distribution
Primary budget f=0.50 headroom and exact pre-registered test bitmap(s)
Literature novelty matrix complete? yes/no
Test lock SHA, status: NO-GO or REQUEST TEST UNLOCK
```

## P4 decision report
```
# Final diagnostic report
## Key question and predeclared hypotheses
## Provenance, task limitations, exact match audio corpus
## Gates G0 G1 G2 G3 G4 and G5 (pass/fail/not assessed + evidence)
## Main plots and uncertainty with sample/support counts
## Search budget bias and small-data limitations
## Reproduction commands / file hashes / CPU-MPS behavior
## Novelty challenge: closest paper, unresolved mechanism, or why no novelty
## Verdict: GO_TO_METHOD_DESIGN / NO_GO_CURRENT_DESIGN / BLOCKED_INVALID
## What result would have reversed the verdict (without revising thresholds)
```
