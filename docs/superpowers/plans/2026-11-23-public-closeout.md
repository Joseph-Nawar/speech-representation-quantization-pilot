# Public research closeout plan

1. Preserve the P2-time README and hash; leave all P0/P1/P2 results and protocol files unchanged.
2. Write a closeout report and portfolio README from the frozen protocol and P2 evidence; add third-party attribution and strengthen local-artifact ignores.
3. Audit licenses, tracked content, reachable Git objects, commit metadata, and release contents. Run only the existing test suite and non-outcome integrity checks.
4. If audits pass, publish a clean snapshot commit to the requested public GitHub repository, then verify repository settings, commit/tree equality, README and required files. Do not run P3/P4 or access test predictions.

## Stop gates

- Stop publication if a secret, sensitive sample, restricted content, or unresolved license issue is found.
- Stop scientific execution at the existing P2 `NO_GO_CURRENT_DESIGN` result.
- Keep the full local history and ignored data/model/run artifacts intact.
