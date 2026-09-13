---
type: Task
title: Honour the requested device, and report the execution provider that ran
status: direction
depth: standard
sensitivity: architecture
milestone: m4-honest-deployment
scope:
  - src/yowo/backends/
  - tests/unit/
  - tests/integration/
gives:
  - S1 OnnxBackend._select_providers honours an explicitly requested device
  - S2 OnnxBackend.active_providers reports the EP that actually executed
generated: { by: add/3.5.0, at: 2026-09-13 }
verified: []
---
## CARD
goal: `device="cpu"` runs on the CPU execution provider; `device="auto"` stays fast and says what it chose.
why: RE-SCOPED 2026-09-13 A SECOND TIME, after the first re-scope was also wrong. The original filing
  claimed a general PyTorch-ONNX divergence of 0.33050537 px. CI refuted it — x86_64 measured
  0.00015450 px, inside the bound — so it was re-scoped to "explain the arm64 arithmetic", with a
  recorded unknown: "whether the arm64 path differs in kernel selection". It does, and that is the
  whole story. `_select_providers` returns the CoreML EP when the caller asks for `device="cpu"`
  (its own docstring documents this), and CoreML computes in FP16. Measured on ONE host, same
  weight, same exported ONNX:
      providers ['CoreMLExecutionProvider', 'CPUExecutionProvider']   0.82116699 px
      providers ['CPUExecutionProvider'] forced                       0.00012207 px
  All five CoreML-path confidences land exactly on the float16 grid; none of the CPU-path ones do.
  So this was never a platform property — it is yowo substituting a different device and a different
  precision than the caller asked for, then reporting `cpu/fp32`. With the EP pinned, all ten
  variants land inside the declared 1e-3 bound (9.155e-05 … 3.0899e-04 px).
  It is user-facing, not test-facing: every macOS user of the ONNX backend who asks for CPU/FP32
  silently gets FP16, with confidences shifted by up to 0.0073 against a default
  `confidence_threshold` of 0.25.
beat: direction · next: add freeze pytorch-onnx-numeric-divergence

## RULES
<must>
- M1 An explicitly requested `device="cpu"` yields exactly `["CPUExecutionProvider"]` — no CoreML, no CUDA — on a host where CoreML is available.
- M2 `device="auto"` keeps the CoreML EP where it is available. The 4-5x Apple Silicon speedup is not paid for this fix.
- M3 The backend reports the providers ORT actually bound, read back from the live session — not the list that was requested.
</must>
<reject>
- R:SUBSTITUTED An explicitly requested device is silently served by a different execution provider -> "SUBSTITUTED"
- R:REQUESTEDASRAN The reported provider list is the one passed in rather than the one read back from the session -> "REQUESTEDASRAN"
- R:AUTOSLOWED `device="auto"` stops selecting CoreML on a host that has it, trading the speedup for the fix -> "AUTOSLOWED"
</reject>

## ASSUMPTIONS
- A1 [who] n/a · no actor distinction; the EP is chosen per-session from the caller's own device argument.
- A2 [which] covers: S1 · the request does not say which device strings count as explicit; taking anything that is not `"auto"` — `"cpu"` and any `"cuda*"` — since `"auto"` is the only value that delegates the choice · probe: `"cpu"` yields only the CPU EP while `"auto"` still yields CoreML -> if `"auto"` were treated as explicit the speedup would be lost for every default caller.
- A3 [when] n/a · selection happens once at load; there is no re-selection boundary.
- A4 [absent] covers: S2 · the request does not say what `active_providers` reads before a session exists; taking an empty tuple rather than None, so a caller can iterate unconditionally · probe: the property is safe to read on an unloaded backend -> a None would make the honest-reporting path raise on the error path that most needs it.
- A5 [order] covers: S1 · the request does not say what orders the provider list; taking ORT's own convention that earlier entries win, so the CPU EP alone means the CPU EP runs -> a trailing fallback that silently wins would reproduce the defect this node fixes.
- A6 [experience] covers: S1, S2 · the request does not say who meets this; taking the macOS user who passes `device="cpu"` expecting CPU numerics and today gets the Neural Engine in FP16, and the developer comparing backends who cannot tell which EP produced a number.

## PLAN
contract: `_select_providers` branches on an explicit device BEFORE the CoreML fallback. `"cpu"` returns `["CPUExecutionProvider"]`; `"auto"` keeps today's CoreML-then-CPU order. The session's bound providers, already computed at `_onnx.py:160` and currently discarded after the OrtValue decision, are retained and published as `active_providers`, read back from `session.get_providers()` so the reported value is what ORT bound rather than what yowo asked for.

## EDGES
- E1 A host WITHOUT CoreML: `"cpu"` and `"auto"` both yield the CPU EP, and the fix is a no-op rather than a behaviour change.
- E2 `device="cuda"` is unaffected — it is explicit and already honoured.
- E3 `active_providers` read before `load()` — the error path, where knowing the EP matters most.
- E4 A host WITH CoreML where `"auto"` is requested: CoreML must still be selected, or the fix cost the speedup.

## CHECKS
- test_an_explicit_cpu_device_yields_only_the_cpu_provider · covers: M1, A2, A5, R:SUBSTITUTED · with CoreML reported available, `_select_providers("cpu")` returns exactly the CPU EP.
- test_auto_still_selects_coreml_where_available · covers: M2, E4, R:AUTOSLOWED · the speedup path is untouched.
- test_a_host_without_coreml_is_unaffected · covers: E1 · both device strings yield the CPU EP.
- test_cuda_is_still_honoured · covers: E2 · an explicit cuda request is unchanged.
- test_active_providers_is_read_back_from_the_session · covers: M3, R:REQUESTEDASRAN · the property reflects the live session, not the requested list.
- test_active_providers_is_empty_before_load · covers: A4, E3 · safe to read on an unloaded backend.
- test_a_cpu_request_actually_runs_on_the_cpu_provider · covers: M1, M3 · a REAL ONNX session built through the real load path reports only the CPU EP for `device="cpu"`.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- A measured deviation is a property of the configuration that produced it, not of the platform it ran on. This figure was filed twice as a platform fact — first as a general PyTorch-ONNX divergence, then as arm64 arithmetic — before anyone checked which execution provider had actually run. Pin the provider before attributing the number. -> add learn tdd
