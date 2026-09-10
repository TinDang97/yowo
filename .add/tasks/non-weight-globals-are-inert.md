---
type: Task
title: A global the state_dict does not need is stubbed, never resolved
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/
  - tests/unit/
  - scripts/
gives:
  - S1 the rule deciding whether a named global is resolved, stubbed, or refused
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:5e2e0a2d22c65b7c", receipt: /tasks/non-weight-globals-are-inert.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:RESOLVEGADGET=confirm|R:SILENTWIDEN=confirm|R:CORRUPT=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:78ec7422b239ca74", binding: "sha256:33ee11d29cc8a55d" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:b310775178405f0b" }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:78ec7422b239ca74", binding: "sha256:33ee11d29cc8a55d" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:a882f515bbd9092c" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: FAIL, receipt: /tasks/non-weight-globals-are-inert.d/runs/1.md }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/non-weight-globals-are-inert.d/runs/2.md }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/non-weight-globals-are-inert.d/runs/3.md }
  - { by: "cli", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:ce28dae36ce12a43", binding: "sha256:33ee11d29cc8a55d" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/non-weight-globals-are-inert.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/non-weight-globals-are-inert.d/runs/4.md, brief: "sha256:65a006118b143d2a", reason: "20 checks green, zero skipped. Independently re-verified, not taken on the builder's report: (a) find_class probe — __builtin__.getattr and torchvision.transforms.transforms.Compose return the inert stand-in; builtins.getattr, __builtin__.eval, builtins.exec, os.system, posix.system, subprocess.Popen, torch.FloatStorage and torchvisionEVIL.Anything are all refused; (b) the inert getattr cannot read an attribute — inert(obj,'secret') returns a stand-in, not the value; (c) R:CORRUPT measured on both branches — yolo11n.pt 499 entries, digest 85fc8064..b6819b44, identical on main and here, while -cls and -obb refuse on main and load here at 236 and 541 entries, matching the corrected E2 exactly; (d) M6 re-run — the repaired measure_checkpoint_globals.py regenerates the manifest identically apart from measured_at, which it could NOT do before this node because the script imported the deleted _ALLOWED_STORAGE_SUFFIX. TWO THINGS RECORDED AGAINST THIS GATE RATHER THAN HIDDEN. First: E3 and E6 were unbound on the first receipt because both cite parametrised checks, and a bare name cannot bind against a reported id ending in [param]. The gate refused the PASS and was right; the node was refrozen citing each parametrisation on its own line and re-run, and the seven items were passing throughout — the gap was in the citation, never in the coverage. Second: the build edited the parent check test_no_rule_admits_a_torch_namespace. Reviewed by hand — startswith('torch') was a substring false positive on 'torchvision.'; it now reads s=='torch' or s.startswith('torch.'), and a strictly stronger AST assertion was added beside it proving exactly one branch reaches super().find_class and that its test is the _ALLOWED membership check. Suite 2332 passed / 11 skipped, ruff clean, pyright 0 errors." }
advised_by: security-reviewer
---
## CARD
goal: `-cls` and `-obb` checkpoints load, by making the globals they name that the state_dict does not need INERT — never by resolving them.
why: `task-aware-weight-resolution` made a classify spec resolve and digest-verify its own `yolo11n-cls.pt`. The loader then refuses it. Measured 2026-09-10 against the real restricted loader, not a scan: `-obb` refuses on `__builtin__.getattr`, `-cls` on `torchvision.transforms.transforms.Compose`. The allowlist is correct to refuse — it was measured from the ten DETECTION checkpoints and these are different files — and this is the change-request path its own refusal message names. What it must NOT do is admit them. `getattr` in a restricted unpickler is a general attribute-access primitive: a crafted checkpoint that can call it can reach any attribute of anything it can name and chain from there, which is the classic pickle gadget and would undo the narrowing `narrow-loader-allowlist` and `storage-suffix-enumeration` performed. `torchvision.transforms.*` is the training-time preprocessing pipeline stored as objects, and `Compose` holds a list of arbitrary callables. Neither is weights. Both are metadata riding along in the same pickle. DECIDED BY THE HUMAN, 2026-09-10: `torchvision.` joins the stubbed prefixes beside `ultralytics.` and `models.`, and `__builtin__.getattr` is handed an inert callable, never the real one. Measured with an inert `getattr` and `torchvision.` stubbed, which is what this node specifies: `-obb` yields 541 entries and `-cls` 236, with nothing further refused. An earlier note here claimed a further `-cls` refusal; it was wrong, and the correction is recorded at E2 rather than quietly dropped.
beat: done · next: add status

## RULES
<must>
- M1 A global the state_dict does not need is never RESOLVED. It is refused, or handed an inert stand-in that constructs nothing and executes nothing. `super().find_class` is reached only for names on the exact allowlist.
- M2 `torchvision.` is a stubbed prefix, exactly like `ultralytics.` and `models.`. No torchvision class is ever constructed from checkpoint bytes.
- M3 `__builtin__.getattr` resolves to an inert callable, never to the real `getattr`. The same holds for any other general-purpose primitive a checkpoint may name.
- M4 The set of RESOLVED names is unchanged by this task. Nothing moves from refused to resolved. This node widens what may be STUBBED, which is a different and weaker permission, and the two must stay separately named in the code so a later reader cannot confuse them.
- M5 The nano variant of every registered task loads through the real loader and yields tensors, and the ten-variant detection regression is unchanged.
- M6 Every decision here is justified by a recorded measurement taken through the REAL loader, and the measurement is regenerable by a committed script.
</must>
<reject>
- R:RESOLVEGADGET Resolving `getattr`, `setattr`, `eval`, `exec`, `__import__`, `compile`, `open`, or any other general-purpose primitive, on any branch. -> "RESOLVEGADGET"
- R:SILENTWIDEN A name added to the RESOLVED allowlist without a recorded observation naming the checkpoint it came from. -> "SILENTWIDEN"
- R:CORRUPT A stand-in that changes which tensors the loader returns, or their values. Inert must mean invisible to the result. -> "CORRUPT"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose checkpoint is in scope; taking any file reaching `_extract_state_dict`, official or user-supplied, since the boundary cannot tell them apart and a pin only says a file is the one that was published, never that it is safe -> if wrong and only official weights matter, the boundary is stricter than it needs to be, which is the correct direction to be wrong in.
- A2 [which] covers: S1 · the request does not say which globals are in scope; taking exactly those the measurement shows `-cls` and `-obb` name and today's boundary refuses, and nothing else — no speculative additions for variants nobody has measured -> if wrong, a variant fails later and takes its own change-request, which is the process working.
- A3 [when] covers: S1 · the request does not say where inert differs from refused; taking a stand-in as the answer only where the name is metadata the state_dict provably does not need, and a refusal everywhere else, so the default stays "no" -> if wrong and stubbing is applied broadly, a genuinely needed class silently becomes an empty object and the tensors go missing quietly.
- A4 [absent] covers: S1 · the request does not say what a stand-in should do when the stream calls, indexes or sets an attribute on it; taking absorb-and-continue, because a stand-in that raises aborts the load partway and reports a truncated result as a complete one -> if wrong, a genuinely malformed checkpoint reads as loadable. · probe: a checkpoint whose tensors are absent must still fail, not return an empty mapping.
- A5 [order] covers: S1 · the request does not say the order of the three decisions in `find_class`; taking exact-allowlist first, then inert names, then stubbed prefixes, then refuse — most specific to least, with refusal last, so a name can never be admitted by a broad rule that a narrow rule already answered -> if wrong and prefixes are consulted first, a prefix silently outranks the exact set and the narrowing is undone.
- A6 [experience] covers: S1 · the request does not say who reads a refusal; taking a maintainer adding a variant, who needs the refusal to name the exact global, say whether the fix is a stub or an allowlist entry, and point at the script that produces the observation — the existing message already does this and must keep doing it -> if wrong, the next person disables the check instead of measuring.

## PLAN
contract: `_extract_state_dict(path) -> object` and `load_verified_state_dict` unchanged. Inside `_restricted_unpickler_module`, `find_class` gains a distinctly-named set of INERT names beside the existing `_ALLOWED` and `_STUBBED_PREFIXES` — resolved names and stubbed names stay separate constants with separate comments, because collapsing them is how a stub becomes an admission. `torchvision.` joins `_STUBBED_PREFIXES`. The measurement is recorded the way `narrow-loader-allowlist` records its own, extended to `-cls` and `-obb` — which is now possible precisely because those entries carry pins as of `task-aware-weight-resolution`, satisfying the script's own stated precondition that a trust boundary is never measured from unverified bytes.

## EDGES
- E1 `yolo11n-obb.pt` — refuses on `__builtin__.getattr` today; must load and yield 541 tensors.
- E2 `yolo11n-cls.pt` — refuses on `torchvision.transforms.transforms.Compose` today; must load and yield its
  236 entries. CORRECTED 2026-09-10: this edge previously asserted "there is at least one further refusal
  behind it". There is not. That claim came from a pre-freeze probe that admitted the REAL `getattr` and
  stubbed torchvision; the real `getattr` then failed against a stubbed object and the failure was
  misrecorded as a further refusal. Re-measured with an INERT `getattr`, which is what this node actually
  specifies: `-obb` loads 541 entries and `-cls` loads 236, with nothing else refused. An edge asserting a
  refusal that does not exist cannot be satisfied honestly, and the first build stalled on it.
- E3 A checkpoint naming `builtins.eval`, `builtins.exec` or `os.system` — still refused, by name, with the existing message. The gadget test.
- E4 The ten detection variants — byte-identical tensors before and after. R:CORRUPT.
- E5 The legacy non-zip RCE probe — still refused before anything runs. This node must not reopen the path `checkpoint-loader` closed.
- E6 A `torchvision` name in a position where the state_dict WOULD need it — must not silently yield an empty object; the tensors must still be complete or the load must fail.

## CHECKS
all in `tests/unit/test_inert_globals.py`, and every one goes through the real
`_extract_state_dict` rather than inspecting a constant.
- test_the_obb_checkpoint_loads · covers: M5, E1 ·
  the measured refusal on `__builtin__.getattr` is gone and the load yields its tensors.
- test_the_classify_checkpoint_loads · covers: M5, E2 ·
  the same for the `torchvision` refusal and whatever measurement shows behind it.
- test_super_find_class_is_reached_only_for_allowlisted_names · covers: M1 ·
  a spy on the real resolution records every name it is asked for across a load of each
  task's checkpoint, and every one is on the exact allowlist. Stubbed and inert names must
  never reach it — that is the difference between standing in for a class and importing it.
- test_getattr_resolves_to_something_inert_not_the_builtin · covers: M3, R:RESOLVEGADGET ·
  the object handed back is not `builtins.getattr` and cannot read an attribute off a real object.
- test_a_general_purpose_primitive_is_still_refused[eval] · covers: E3, R:RESOLVEGADGET · refused by name, with the message that names it.
- test_a_general_purpose_primitive_is_still_refused[exec] · covers: E3, R:RESOLVEGADGET · the same for `exec`.
- test_a_general_purpose_primitive_is_still_refused[import] · covers: E3, R:RESOLVEGADGET · the same for `__import__`.
- test_a_general_purpose_primitive_is_still_refused[os-system] · covers: E3, R:RESOLVEGADGET · the same for `os.system`.
  Cited one parametrisation per line: a bare name cannot bind against a reported id ending in
  `[param]`, and the gate correctly refused a PASS while these read as unbound (method M12).
- test_no_torchvision_class_is_ever_constructed · covers: M2 ·
  a checkpoint naming a torchvision transform gets a stand-in; the real class is never reached.
- test_the_resolved_allowlist_is_unchanged · covers: M4, R:SILENTWIDEN ·
  `_ALLOWED` matches the recorded measurement exactly, so a stub cannot be smuggled in as an
  admission — this node widens what may be STUBBED, which is a different permission.
- test_resolved_and_stubbed_are_separate_constants · covers: M4 ·
  the two sets are named separately in the source; collapsing them is how a stub becomes an
  admission, and a later reader must not be able to make that mistake by accident.
- test_a_stand_in_absorbs_calls_without_aborting_the_load · covers: A4 ·
  indexing, calling and attribute-setting on a stand-in do not raise, so a load cannot end
  early and report a truncated result as a complete one.
- test_a_checkpoint_with_no_tensors_still_fails[empty-checkpoint] · covers: A4, E6 · the probe A4 names: absorb-and-continue must not turn a malformed checkpoint into an empty success.
- test_a_checkpoint_with_no_tensors_still_fails[no-model-key] · covers: A4, E6 · the same for a checkpoint with no 'ema'/'model' key.
- test_a_checkpoint_with_no_tensors_still_fails[stand-in-without-tensors] · covers: A4, E6 · the same for a stand-in that walks the whole stream and holds nothing.
- test_the_ten_detection_variants_are_byte_identical · covers: E4, R:CORRUPT ·
  every tensor of every pinned detection variant is unchanged by this node.
- test_find_class_consults_the_exact_set_before_any_prefix · covers: A5 ·
  most specific to least, refusal last, so a broad rule can never answer a question a narrow
  rule already answered.
- test_the_legacy_format_probe_still_refuses_before_anything_runs · covers: E5 ·
  the path `checkpoint-loader` closed stays closed; this node must not reopen it.
- test_every_decision_here_names_its_measurement · covers: M6, A2, A6 ·
  each stubbed or inert name is traceable to an observation taken through the real loader, by a
  committed script, and the refusal message still tells a maintainer which to reach for.
- test_the_threat_model_covers_a_user_supplied_checkpoint · covers: A1 ·
  the boundary treats an unpinned local file exactly as it treats an official one.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
