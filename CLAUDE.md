# YOWO - bring inference YOLO models to production include edge device with adaptive self-optimizing

## Read README.md of each module to clarify Commands and architect

### Parallel agent pattern

When implementing a new backend end-to-end, run all relevant agents in parallel after the implementation is written:

```
Simultaneously:
  - backend-compliance-reviewer   → protocol + error mapping
  - inference-perf-auditor        → hot-path performance
  - test-coverage-guardian        → coverage gaps
```

Only proceed to commit after all three return APPROVE / no P0 issues / no uncovered error paths.
