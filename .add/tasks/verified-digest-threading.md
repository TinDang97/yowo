---
type: Task
title: The verified digest reaches the loader, closing the resolve-to-load window
status: direction
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/engine.py
  - src/yowo/arch/_weights.py
  - src/yowo/backends/_pytorch.py
  - tests/unit/
gives:
  - S1 `load_verified_state_dict(path, raw_digest)` — the pin comparison that now precedes conversion
  - S2 the pin's route from the registry to that loader, via `PyTorchBackend.load`
generated: { by: add/3.5.0, at: 2026-09-09 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:d95d0e07955acb10", receipt: /tasks/verified-digest-threading.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|R:SELFKEYED=confirm|R:CONVERTFIRST=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:e136f32b8ab54b7e", binding: "sha256:e9a79d98e3503d91" }
advised_by: artifact-integrity-steward
---
## CARD
goal: The digest `resolve_weights` verified is passed to the loader, so nothing between them can substitute the file.
why: found by the security residue lens while verifying `weight-integrity`, as a residual limit of that task rather than a defect it introduced.
  `resolve_weights` verifies the cached file against its pin and returns a path. `load_verified_state_dict`
  then re-hashes that path — but only to KEY the converted sidecar, not to re-check it against the pin.
  A file substituted in the window between those two calls yields a different key, so the sidecar misses,
  and the substituted bytes are converted and loaded without ever being compared to the pin.

  `load_verified_state_dict` already TAKES a `raw_digest` argument; nothing passes one. Threading the
  verified value from the engine and re-checking before conversion closes it — but the engine is outside
  `weight-integrity`'s frozen scope, which is why this is a separate node rather than a silent widening.

bounded, and the bound is why this was not a HARD-STOP on weight-integrity's gate:
  - it requires local write access to the weight cache;
  - code execution is already blocked regardless, by `checkpoint-loader`'s restricted unpickler — the
    worst case is substituted model weights, not RCE;
  - before `weight-integrity` there was no verification at any point, so the window is a narrowing of an
    open door, not a new one. Blocking that gate would have kept the door fully open to punish a fix.
beat: scaffold · next: author verified-digest-threading's RULES, ASSUMPTIONS and CHECKS, then add freeze verified-digest-threading

## RULES
<must>
- M1 The raw `.pt` is compared against its pinned digest immediately before the unpickling conversion,
  not only at resolution time.
- M2 The digest compared against is the registry pin for the loaded spec. A value computed from the
  file being checked is never treated as verification.
- M3 An unpinned model — an explicit `weights_path`, or a registry entry with `sha256=None` — keeps
  today's behaviour: it converts, and its sidecar is keyed on the computed digest.
- M4 The `Backend.load` protocol is unchanged. No backend that never unpickles gains a parameter it
  cannot use.
- M5 A sidecar hit whose stored `raw_sha256` equals the pin is served without re-hashing the raw file —
  the steady-state path `weight-integrity` made fast stays fast.
</must>
<reject>
- R:SELFKEYED A digest computed from the file under test may never satisfy the verification of that same
  file -> "SELFKEYED"
- R:CONVERTFIRST No unpickling conversion may run before the pin comparison -> "CONVERTFIRST"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S2 · the request does not say who supplies the digest; taking: the registry, via the
  loaded spec — not the caller, and never the file itself
  -> a caller-supplied digest is only as trustworthy as the caller, and a file-derived one is circular
  (R:SELFKEYED).
- A2 [which] covers: S1, S2 · the request does not say which load paths are in; taking: all three —
  `load_weights`, `load_classify_weights`, `load_obb_weights` — because all three call
  `load_verified_state_dict` and all three unpickle · probe: no call site of
  `load_verified_state_dict` omits the digest for a pinned spec
  -> covering detection alone leaves two unpickling paths open while claiming the window is closed.
- A3 [when] covers: S1 · the request does not say where the boundary falls; taking: re-verify on every
  call that will actually convert, and NOT when a sidecar already matched the pinned key — the sidecar
  is loaded `weights_only=True` and was produced from bytes that hashed to the pin
  -> hashing 5-110 MB on every steady-state load would tax the exact path the parent task made fast
  (M5).
- A4 [absent] covers: S1 · the request does not say what `raw_digest=None` means now; taking: unpinned,
  keeping the compute-and-key behaviour — but the computed value keys the sidecar and never satisfies a
  comparison -> conflating "no pin" with "verified" is R:SELFKEYED restated.
- A5 [order] covers: S1 · the request does not say the order; taking: the pin comparison runs strictly
  before `_extract_state_dict` -> converting first unpickles the bytes the comparison exists to reject
  (R:CONVERTFIRST).
- A6 [experience] covers: S1 · the request does not say what a load-time mismatch says; taking:
  `verify_digest`'s existing message, unchanged — it already names path, expected and actual
  -> a second, differently worded integrity error would leave a reader unable to tell whether they hit
  resolution or load.
- A7 [who] covers: S2 · the request does not say who reads the pin; taking: `PyTorchBackend.load`,
  which already holds `self._spec` and already imports `get_cls`/`get_obb`, rather than `engine.load`
  passing it down through `Backend.load` · probe: `Backend.load`'s signature is byte-identical across
  all five backends after the change -> threading it through `Backend.load` would widen a protocol
  shared by four backends that never unpickle, to carry a value only one can use (M4). This is why
  `src/yowo/backends/_pytorch.py` was added to this node's scope; `engine.py` stays in scope but is
  expected to be untouched.
- A8 [absent] covers: S2 · the request does not say what a swapped SIDECAR does; taking: it is out of
  scope and stays so — it loads `weights_only=True`, so the worst case is substituted tensors and not
  code execution, and closing it means signing the sidecar
  -> silently widening to sidecar signing would make this node unreviewable as a quick change.

## PLAN
contract:
  - S1 `load_verified_state_dict(path, raw_digest=None)` calls `verify_digest(path, raw_digest)`
    before `_extract_state_dict` whenever `raw_digest` is not None and no sidecar matched. Signature
    unchanged — the argument already exists and nothing passed one.
  - S2 `load_weights` / `load_classify_weights` / `load_obb_weights` each take
    `raw_digest: str | None = None` and forward it.
  - S3 `PyTorchBackend.load` resolves the pin for its spec from the registry (`None` when
    `spec.weights_path` is set, or when the entry is unpinned) and passes it to whichever of the three
    it calls.
strategy: red the tautology first — assert that a computed digest does NOT satisfy verification — then
  the comparison, then the threading, then the three call sites.
regression floor: `test_weight_integrity.py`, `test_checkpoint_loader.py`, `test_engine.py`,
  `test_classification_engine.py`, `test_obb_engine.py`, `test_classify_weights.py` stay green.

## EDGES
- E1 A pinned model whose cached `.pt` is swapped after resolution, with no sidecar present, must FAIL
  rather than convert.
- E2 A pinned model with a sidecar keyed on the pin must not re-hash the raw file (M5, A3).
- E3 An unpinned model — `weights_path` set, or `sha256=None` — must still load (M3).
- E4 The cls and obb paths must pass through the same gate as detection (A2).

## CHECKS
- test_file_swapped_after_resolution_is_refused · covers: M1, R:CONVERTFIRST, E1.
- test_computed_digest_never_satisfies_verification · covers: M2, R:SELFKEYED, A4 · the tautology this
  node exists to remove.
- test_verification_precedes_conversion · covers: M1, A5, R:CONVERTFIRST · `_extract_state_dict` is
  never reached on a mismatch.
- test_sidecar_hit_on_the_pin_skips_rehashing · covers: M5, A3, E2.
- test_unpinned_model_still_loads · covers: M3, A4, E3.
- test_all_three_load_paths_forward_the_digest · covers: M2, A2, E4.
- test_backend_load_protocol_is_unchanged · covers: M4, A7 · all five `load` signatures identical.
- test_mismatch_message_is_the_resolution_message · covers: A6.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
