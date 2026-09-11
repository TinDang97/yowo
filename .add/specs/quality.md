---
type: Spec
title: Quality
lens: quality
project: yowo
description: what counts as proof
tags: [evidence, tdd, ci, parity]
sources: [docs/reviews/2026-09-08-production-readiness/, .github/workflows/release.yml, pyproject.toml]
generated: { by: add/3.5.0, at: 2026-09-08 }
delta_seq: 7
---
## Now

**What holds today** (measured 2026-09-08, not claimed):
- `uv run pyright src/yowo/` → `0 errors, 0 warnings, 0 informations` at `typeCheckingMode = "strict"`.
  `py.typed` ships. This gate is real and it passes.
- `uv run pytest tests/ --collect-only -q` → **2175 tests** (2142 unit + 33 integration).
- 80% line coverage over `src/yowo` (8883 statements, 1807 missed), 2132 passed / 11 skipped in 75s.
- `ruff check` + `ruff format --check` clean.

**What does not hold, and is the reason this spec exists:**
- **No test in the repository executes a real backend forward pass.** 541 `MagicMock` and 565
  `patch(` across 47 of 94 unit files. `onnxruntime.InferenceSession` is never constructed.
  `_pytorch.py` — the default backend — is 49% covered with `infer()`, `warmup()`, `unload()` and
  `_resolve_device()` entirely uncovered, and has no test file at all. `_openvino.py` is at **0%**.
- **The integration tier is dead.** `uv run pytest tests/integration -q` → 1 failed, 13 passed,
  20 skipped. `tests/integration/conftest.py:16` hardcodes a path under one developer's `~/Downloads`.
  CI never invokes it. `test_cli_e2e.py:76` has asserted version `"0.1.0"` against `2.5.0` since
  before v2.4.0.
- **The architecture-equivalence evidence does not exist.** `tmp/compare_arch.py`, cited as the source
  of the "10/10 variants PASS" claim, is gitignored and was never tracked.
- **No branch coverage, no `fail_under`, no `[tool.coverage]` section anywhere.**
- **The gate certifies one environment** — ubuntu-latest × Python 3.11 × `tests/unit/` — while the
  package claims 3.8–3.12, five backends, and seven extras. Nothing else is verified by anything.

## Decisions that bind

- **A green unit suite is evidence about mocks, not about inference.** Any claim about a backend
  requires a check that constructs that backend and runs it. Coverage percentage is not a proxy.
- **A claim without a runnable check is not a claim.** Prose in a README, a docstring, or `CLAUDE.md`
  asserting a property carries no weight at a gate. Three such claims were found to be actively false
  (result-JSON safety, arch equivalence, ultralytics-free packaging) — see
  `docs/reviews/2026-09-08-production-readiness/README.md`.
- **Evidence must survive the session that produced it.** A harness in a gitignored directory has
  produced nothing. Anything that proves a property is tracked and CI-invoked, or it does not count.
- **Numbers are produced by commands, and the command is recorded.** No estimated coverage, no
  remembered benchmark, no invented statistic. `add run` receipts carry the command.
- **The receipt is the narrowest command that reports every bound check** — the full suite rides CI.
- **The three residue lenses are mandatory at Verify**: security (HARD-STOP — weight
  deserialization, credential handling), concurrency (threaded readers, batch scheduler, shared
  caches), architecture (the backend Protocol, the engine hierarchy).

## The evidence this project still owes

Tracked as milestones in this bundle; each is a gate that does not exist yet:
1. A real-backend smoke check for every backend `create_backend()` can return.
2. A cross-backend conformance suite — same model, same input, asserted agreement within a stated tolerance.
3. A committed, CI-invoked ultralytics equivalence harness.
4. Export round-trip numeric parity (PyTorch ↔ ONNX ↔ downstream).
5. An mAP regression gate on a real dataset.
6. A measured accuracy delta for every quantized artifact.
7. A CI matrix that matches the declared support surface, or a declared surface that matches CI.

## Deltas
- 2026-09-08 · authored from the six-lane production-readiness review; replaces the scaffold's
- [TDD · Q7 · open · 2026-09-12] A memory check must allocate pages the OS actually backs. cli-bounded-memory built probe frames with np.zeros; calloc returns untouched zero pages that stay non-resident until written, so on Linux retaining 200 1080p frames cost 0.00 MB and the control measured no retention. macOS faulted them in, so it passed locally and failed in CI. np.full writes real bytes. The control is what caught it — a measurement check without one reports green about nothing. (evidence: /tasks/cli-bounded-memory.md)
- [TDD · Q6 · open · 2026-09-11] A check whose subject is TERMINATION must itself terminate, or the defect hangs the suite instead of reddening it. The first run of test_rtsp_reconnect did not fail -- it hung for ten minutes, because with every read failing the generator never yields and next() never returns. That is exactly the bug under test, but a hung suite is strictly worse than a red one: a red check names its cause in the failure message, while a timeout names nothing and trains everyone to deselect the file. The fix is to give the fake clock a BUDGET and raise a distinctive exception when it is exhausted, so 'this never terminates' becomes an assertion about a counter rather than a wall-clock wait. The general rule: whenever the property under test is that something STOPS -- a retry loop, a backoff, a poll, a recursive walk, a queue drain -- the harness must impose its own independent stop, and exceeding it must be a named failure. Corollary, from the same node: a check asserting a value RESETS cannot use a threshold. test_a_successful_read_resets_the_outage_clock asserted the second outage lasted '> 10s' and passed with the reset deleted, because it still inherited 23s of a 30s budget. Compare against a CONTROL -- a fresh run of the same scenario -- because 'reset' means 'indistinguishable from fresh', and only a control can say that. (evidence: /tasks/rtsp-reconnect-correctness.d/runs/1.md)
- [TDD · Q5 · open · 2026-09-11] A check that scans PROSE for a keyword must match on word boundaries, because English embeds the words you are looking for. Three instances in one node, all false positives on correct content: my own R:SILENTWIDEN check reported that m1 box 6 claimed the 'host' component, matching inside 'hostage' in the box's own justification; licensing-provenance's 'no LTS promise' check matched 'lts' inside 'results'; and its email-intake regex matched 'hunter2@10.0.0.5', because an RTSP userinfo on a dotted host is indistinguishable from an email address to a scanner. This is the same shape as the '1' that matched inside the host '10.0.0.5' one node ago, and it is worse than a check that proves nothing: a check that fires on correct content teaches the author to edit the CONTENT until the check is happy. I changed two SECURITY.md examples to dodge checks that were simply wrong, which is that failure happening. The rule: when a check searches text for a term, use a word boundary, and prove it by feeding it a document containing the term as a substring of an innocent word. The corollary for regexes over documents that legitimately contain URLs: scope the scan to prose, or exclude fenced code blocks, or the check cannot tell a credential example from the thing it forbids. (evidence: /tasks/box-6-scope-amendment.d/runs/4.md)
- [TDD · Q4 · open · 2026-09-11] A parametrised check proves nothing about ORDER if the parameters are already in the order the bug would impose. I wrote test_query_keys_and_their_order_survive over the keys 'alpha' and 'beta' and asserted alpha comes before beta. A sorted() injected into the implementation is indistinguishable from preserving order on that input, so the mutation SURVIVED — a check whose entire subject is ordering, passing under the exact change it exists to catch. Found by refuting, not by reading: the check reads correct, and rereading it a third time would not have found it. Fixed with 'zulu' and 'alpha'. The general rule: when a check asserts that some property is PRESERVED, the fixture must be one where the most likely wrong implementation would visibly differ — sorted input cannot test sorting, an already-deduplicated list cannot test deduplication, an idempotent value cannot test idempotence. Choose the fixture adversarially, then prove it by injecting the wrong implementation and watching the check go red. (evidence: tasks/query-string-credentials.d/runs/1.md — mutation 'A5: reorder query keys' -> 36 passed, the check survived; after changing the fixture to zulu/alpha the same mutation -> 1 failed. runs/3.md: PASS.)
- [TDD · Q3 · open · 2026-09-09] A skipped test is green. Two fixtures pointed at one machine's filesystem, so an entire integration tier skipped for everyone else and reported success for months — hiding a version assertion frozen at 0.1.0 while the project shipped 2.5.0. Fixtures that cannot obtain their input must FAIL in CI. (evidence: /tasks/ci-weight-fixture.md)
- [TDD · Q2 · open · 2026-09-09] A check that only asserts a symbol EXISTS passes while the symbol is dead code. load_verified_state_dict was never called by any real load path, every load still unpickled, and the check was green. Assert the call site, not the definition. (evidence: /tasks/weight-integrity.md)
- [TDD · Q1 · open · 2026-09-08] The pyright pre-commit hook fails on EVERY commit regardless of content: it runs via 'uv run', and uv 0.10.11 rewrites uv.lock from revision 2 to 3, so pre-commit sees 'files were modified by this hook' and rolls back. pyright itself reports 0 errors. A gate that fails identically on every input is not a gate — it trains the author to pass --no-verify. Same mechanism threatens CI, where 'uv sync' can re-resolve the lock the build claims to pin. (evidence: 25e907e)
  placeholder. Evidence: `docs/reviews/2026-09-08-production-readiness/` (33 P0, 64 P1, all cited).
