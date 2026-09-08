---
type: Milestone
title: Every claim yowo makes has a runnable check behind it
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: milestone-planner
---
## CARD
goal: Every claim yowo makes about inference has a runnable, committed check behind it.
why: 2175 tests and 80% line coverage prove things about mocks. No test executes a real backend forward pass; the default PyTorch backend has no test file at all and OpenVINO is at 0%; the integration tier is gated on one developer's home directory and has been failing since before v2.4.0; and the architecture-equivalence harness everyone cites was never committed. The product's entire value rests on evidence that does not exist in the repository.
next: add new task <slug>

## SCOPE
In:  real-backend execution tests · a shared cross-backend conformance suite · a committed CI-invoked ultralytics equivalence harness · export round-trip numeric parity · mAP regression gate · revival of the integration tier · the CI matrix across the declared support surface · branch coverage and a floor
Out: fixing the export bugs the parity suite will find (→ m4-honest-deployment) · runtime resilience behaviour (→ m2, already done by then) · API surface changes (→ m5-declare-ga)

## GROUND
touches: tests/ · .github/workflows/ · pyproject.toml ([tool.coverage], [tool.pytest]) · a new tracked equivalence harness · src/yowo/backends/ (test hooks only)
risks:
  - This milestone will surface failures rather than fix them. Expect the parity and equivalence suites to go red on first run; that is success, and the fixes belong to m4. Do not weaken a check to make the suite green.
  - Real-backend tests need real weights, which means CI needs a cached, integrity-verified fixture — dependent on m1's weight-integrity work.
  - Settling `requires-python >=3.8` is a scope decision with user impact: the claim is currently untested by any CI or lockfile, and dropping it is a breaking change for anyone on 3.8.

## EXIT
- [ ] Every backend `create_backend()` can return is constructed and executed by at least one test   (← real-backend-smoke)
- [ ] A shared parametrized conformance suite asserts identical behaviour across backends — output shape/dtype, empty detections, class mapping, error type on bad input — with a stated tolerance and the actual deviation reported   (← backend-conformance-suite)
- [ ] The ultralytics equivalence harness is tracked, runs in CI, and its result is a gate   (← arch-equivalence-in-ci)
- [ ] PyTorch↔ONNX numeric agreement is asserted for every exported variant   (← export-roundtrip-parity)
- [ ] An mAP regression gate runs against a real dataset with a recorded baseline   (← map-regression-gate)
- [ ] `pytest tests/integration` passes from a clean checkout with no hardcoded personal paths, and CI invokes it   (← integration-tier-revival)
- [ ] CI runs the matrix the package claims to support, or the claim is narrowed to what CI runs   (← ci-matrix)
- [ ] Branch coverage is measured with a `fail_under` floor   (← coverage-floor)

## CLOSE
evidence: <one row per task>
