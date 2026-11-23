CONTINUE THE SAME REPOSITORY ONLY AFTER THE RESEARCH OWNER HAS REVIEWED `reports/P0_PREFLIGHT.md` and `reports/P1_SMOKE.md` AND EXPLICITLY APPROVED P2.

Read all docs and `AGENTS.md` again, then:
- Freeze dataset/audio/grid/split/head/model/quantizer hashes as P2 describes, train baseline frozen heads using **train/dev only**, and run quantizer technical controls. Calculate G0/G1/G2. Stop immediately on any failure.
- If approved to continue P3 and G0–G2 pass, run **dev-only** isolated block sensitivity and predeclared allocation policy comparisons as in `docs/03_quantization_and_baselines.md` and `docs/04_analysis_and_gates.md`. Evaluate G3 and provisional G4 without test access. Save all failures, scores, mixed-precision bitmaps, provenance, and CIs.
- Freeze the exact P4 test plan and `test_lock.json` SHA. Return `reports/P3_DEV_FINDINGS.md` with complete GO/NO-GO rationale. **STOP BEFORE TEST ACCESS.** Do not run P4 without separate explicit authorization.
- Once P4 is separately approved: read sealed test once, confirm all predeclared policies, generate uncertainty and final verdict, and produce auditable evidence files. Do not change threshold values on results.

Keep scientific modifications separate from code bug fixes. If any protocol amendment is needed after outcome inspection, create a new protocol version and a separate new study. Do not write a paper or promise publication.
