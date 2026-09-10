---
type: Task
title: The verified digest reaches the loader, closing the resolve-to-load window
status: done
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
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-09, act: refreeze, authority: human, direction: "sha256:ada354e1b0ee85f3", binding: "sha256:e9a79d98e3503d91" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-09, act: refreeze, authority: human, direction: "sha256:73d5bdd5024446dc", binding: "sha256:e9a79d98e3503d91" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-09, act: refreeze, authority: human, direction: "sha256:0a9800393106e820", binding: "sha256:e9a79d98e3503d91" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-09, act: gate, authority: human, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/4.md, brief: "sha256:f43440757ce221d0", reason: "16 checks green on a bound receipt. R:SELFKEYED made structural: pin (registry-only, authenticates) and sidecar_key (may be computed, addresses the cache) never merge. M5 confirmed on the real cached yolo11n - sidecar hit 0.025s no rehash, substitution after resolution refused. Backend.load byte-identical across five backends; resolve_weights unchanged so the ~60 mock sites stand. Residue carried not swallowed: export/_exporter.py resolves then loads with no digest, opened as export-digest-threading." }
  - { by: loop, at: 2026-09-10, act: reopen, to: direction, reason: "M5 as frozen mandates the defect that sidecar-not-self-attesting exists to close: 'A sidecar hit whose stored raw_sha256 equals the pin is served without re-hashing the raw file.' The pin is public, printed in _registry.py, so a sidecar keyed on it authenticates itself with a value anyone can read - a planted .state_dict.pt naming the published digest served arbitrary tensors with no read of the checkpoint at all, reproduced 2026-09-10. The check bound to M5 asserted exactly that property with verify.assert_not_called() and rehash.assert_not_called(), so it encoded the hole rather than an invariant. What M5 was really defending is the cost bound - one read of the raw file per load, not two - and that survives intact. Amending M5 to say so, and the check to assert it." }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:e3be03091c443e4f", binding: "sha256:e9a79d98e3503d91" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:a78d3801a2969363" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/5.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/verified-digest-threading.d/runs/5.md, brief: "sha256:a78d3801a2969363", reason: "Re-gated after M5 was amended from 'a sidecar hit whose stored raw_sha256 equals the pin is served without re-hashing the raw file' to 'a load reads the raw checkpoint's bytes exactly once'. The old wording mandated the defect rather than an invariant: the pin is public, printed in _registry.py, so keying the sidecar on it let anything able to write the weight cache plant a .state_dict.pt naming the published digest and have arbitrary tensors served with the checkpoint never opened. Reproduced 2026-09-10. The check bound to M5 asserted precisely that property - verify.assert_not_called() plus rehash.assert_not_called() - so it encoded the hole; a green gate on it proved the wrong thing. The cost bound was the half worth keeping and it holds: I counted hashlib.sha256 constructions myself through load_verified_state_dict and got exactly one on the pinned fast path and one on the unpinned fast path. The check keeps its name so this node's binding survives, and its assertion is now a hash count plus an _extract_state_dict tripwire, which is strictly stronger than the assert_not_called pair it replaces. 19/19 reported, every cited id present, no rule unbound." }
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
beat: done · next: add status

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
- M5 A load reads the raw checkpoint's bytes exactly ONCE — the steady-state path `weight-integrity`
  made fast stays fast, and one measurement answers both the pin comparison and the sidecar's
  authentication. AMENDED 2026-09-10 by `sidecar-not-self-attesting`. This rule previously read "a
  sidecar hit whose stored `raw_sha256` equals the pin is served without re-hashing the raw file",
  which mandated the defect rather than an invariant: the pin is PUBLIC, printed in `_registry.py`,
  so keying the sidecar on it let anything able to write the weight cache plant a `.state_dict.pt`
  naming the published digest and have arbitrary tensors served with the checkpoint never opened.
  The cost bound was the part worth keeping; the not-touching-the-file part was the hole.
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
names reconciled to the tests actually built, 2026-09-09. The builder worked from a worktree cut
before this node's direction was committed, so it never saw the names authored here and chose its
own — and split several declared checks into finer ones. The rules each check covers are unchanged;
only the labels moved, and they moved toward what is on disk.
- test_swapped_file_after_resolution_is_refused · covers: M1, R:CONVERTFIRST, E1 · a file substituted after resolution, with no sidecar, fails instead of converting.
- test_computed_digest_never_satisfies_verification · covers: M2, R:SELFKEYED, A4 · the tautology
  this node exists to remove.
- test_conversion_is_never_reached_on_a_mismatch · covers: M1, A5, R:CONVERTFIRST ·
  `_extract_state_dict` is never reached on a mismatch.
- test_stale_sidecar_falls_through_to_the_pin_comparison · covers: R:CONVERTFIRST, A3 · a sidecar
  miss must land on the comparison, not on the conversion.
- test_sidecar_hit_on_the_pin_does_not_rehash_the_raw_file · covers: M5, A3, E2 ·
  the steady-state fast path costs exactly one hash of the raw file and never reaches the executing
  reader. Name kept so this node's gate still binds; the assertion was corrected from
  `verify.assert_not_called()` + `rehash.assert_not_called()` — which asserted the defect — to a
  hash count plus an `_extract_state_dict` tripwire.
- test_unpinned_load_converts_without_claiming_verification · covers: M3, A4, E3 · an unpinned model still loads.
- test_unpinned_sidecar_is_still_keyed_on_the_computed_digest · covers: M3, A4 · the cache key survives when there is no pin.
- test_every_load_path_forwards_the_pin · covers: M2, A2 · parametrised over all three loaders.
- test_every_load_path_forwards_the_pin[load_classify_weights] · covers: E4 · the classification
  loader forwards the pin, so cls passes the same gate as detection.
- test_every_load_path_forwards_the_pin[load_obb_weights] · covers: E4 · the OBB loader likewise.
- test_backend_threads_the_registry_pin_for_detection · covers: M2, A1, A7 · the pin reaches the loader for a pinned spec.
- test_backend_threads_the_registry_pin_for_cls_and_obb · covers: A2, E4 · cls and obb pass the same gate as detection.
- test_backend_sends_no_pin_for_an_explicit_weights_path · covers: M3, A1 · a user's own checkpoint
  is not compared to the official digest.
- test_backend_load_signatures_stay_identical · covers: M4, A7 · all five backends.
- test_resolve_weights_signature_is_unchanged · covers: M4, A7 · the ~60 mock sites stay valid.
- test_load_verified_state_dict_signature_is_unchanged · covers: M4 · the argument already existed; nothing widened.
- test_no_second_integrity_message_was_authored · covers: A6 · one event, one message, whatever caught it.
- test_integrity_failure_keeps_its_type_through_the_backend · covers: A6, A8 · a substitution
  surfaces as one event with one type, not two depending on timing.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
