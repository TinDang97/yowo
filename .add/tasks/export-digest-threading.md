---
type: Task
title: The export path verifies its weights the way the inference path does
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/export/
  - tests/unit/
gives:
  - S1 the pin's route from the registry to the loader on the export path
generated: { by: add/3.5.0, at: 2026-09-09 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:904200940e8be217", receipt: /tasks/export-digest-threading.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:CONVERTFIRST=confirm|R:PARTIALCLAIM=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:a471590593bace96", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:7e391f9397751efe" }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:f9126ad828ac39d1", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:0b35298571e01cb8" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/export-digest-threading.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:09e74b1cdcf1d88d", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:ff6cf7e91e1a654d" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/export-digest-threading.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/export-digest-threading.d/runs/2.md, brief: "sha256:ff6cf7e91e1a654d", reason: "13 checks green on a bound receipt. All three export branches thread the registry pin; the detection branch gained the lookup it never made, which is exactly the branch a partial fix skips. No new comparison logic - the exporter supplies the value and load_verified_state_dict still owns the single check. R:PARTIALCLAIM discharged: the resolve-to-load window is now shut on the export path as well as inference, so the parent's claim is true of the system rather than of one path. Three rules the build left unbound (M2, A5, R:CONVERTFIRST) were bound at review, with red evidence obtained by reverting the exporter to df02f7f." }
advised_by: artifact-integrity-steward
---
## CARD
goal: The export path re-compares a weight against its pin before conversion, exactly as the
  inference path now does.
why: found by the security residue lens while VERIFYING `verified-digest-threading`, in code that task
  does not own. That node closed the resolve-to-load window for the inference path only.
  - `export/_exporter.py:69` — `weights_path = resolve_weights(spec)`, then all three loaders are
    called with no digest: `load_classify_weights(model, weights_path)` (:80),
    `load_obb_weights(...)` (:91), `load_weights(...)` (:99).
  - Same window, same conversion, same restricted-unpickler read of the raw `.pt`. A file substituted
    between the resolve and the load is converted without ever being compared to the pin.

  `verified-digest-threading` wired the gate: all three loaders now TAKE `raw_digest` and
  `load_verified_state_dict` re-verifies when one is supplied. This node passes it. Two of the three
  branches (`classify`, `obb`) already hold the `meta` the pin lives on; the detection branch does not
  call `get()` yet and will need to.

not a regression, and not a HARD-STOP on `verified-digest-threading`'s gate: the default is `None`, so
  export behaves exactly as it did before that task. But that node's claim — that the window is shut —
  holds only for the inference path until this one lands, and a claim that is true of one path and
  asserted of the system is the kind of thing that gets believed.

bounded the same way its sibling was: it requires local write access to the weight cache, and
  `checkpoint-loader`'s restricted unpickler still blocks code execution regardless, so the worst case
  is a model exported from substituted weights — not RCE. That is worse here than at inference, in one
  specific way: an exported artifact is a FILE that outlives the process and gets shipped somewhere
  else.
beat: done · next: add status

## RULES
<must>
- M1 The export path passes the registry pin to whichever loader it calls, so the raw `.pt` is
  re-compared against the pin immediately before conversion — the same gate the inference path got.
- M2 `resolve_weights`'s signature is unchanged, for the same reason it was unchanged in the parent:
  it is mocked at ~60 test sites and called from three modules.
- M3 An unpinned model — an explicit `weights_path`, or a registry entry with `sha256=None` — keeps
  today's behaviour and still exports.
- M4 No new comparison logic is written. `load_verified_state_dict` already re-verifies when handed a
  digest; this node supplies the value and nothing else.
</must>
<reject>
- R:CONVERTFIRST No unpickling conversion on the export path may run before the pin comparison -> "CONVERTFIRST"
- R:PARTIALCLAIM No node may claim the resolve-to-load window is shut while a path it can see is open -> "PARTIALCLAIM"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who supplies the digest; taking: the registry, via the
  spec being exported — identical to `PyTorchBackend.load`, never the caller and never the file
  -> a second, differently-sourced digest on a second path is how the two paths drift apart.
- A2 [which] covers: S1 · the request does not say which export branches are in; taking: all three.
  `classify` and `obb` already hold the `meta` the pin lives on; the detection branch calls no
  registry lookup today and gains one · probe: no `load_*_weights` call in `export/` omits the digest
  for a pinned spec -> covering the two that are cheap and skipping the one that needs a lookup is
  exactly the shape of a fix that reads as complete and is not.
- A3 [when] covers: S1 · the request does not say where the comparison happens; taking: inside
  `load_verified_state_dict`, already built and gated by `verified-digest-threading` — this node
  changes call sites only -> re-implementing the comparison on the export path would give one rule two
  implementations that can disagree.
- A4 [absent] covers: S1 · the request does not say what an unpinned export does; taking: `None`, and
  today's behaviour — every `-cls` and `-obb` registry entry carries `sha256=None`, so this is the
  common case for two of the three branches -> refusing to export an unpinned model would break
  `yowo export` for every classification and OBB variant.
- A5 [order] covers: S1 · the request does not say where the pin is read; taking: from the same `meta`
  the branch already fetched, not a second registry lookup -> two lookups in one export could observe
  a re-pin inconsistently, and the cheaper thing is also the correct one.
- A6 [experience] covers: S1 · the request does not say what a mismatch says; taking: `verify_digest`'s
  existing message, not wrapped in an export-specific error -> a reader who hits this during an export
  and during a load should see one event described one way.

## PLAN
contract:
  - S1 `export/_exporter.py` passes `raw_digest=` to `load_classify_weights` (:80),
    `load_obb_weights` (:91) and `load_weights` (:99). The detection branch gains the `get()` lookup
    the other two already make. `None` when `spec.weights_path` is set or the entry is unpinned.
strategy: red the three branches first — assert the digest arrives — then thread it.
regression floor: `test_obb_export.py`, `test_export.py` and the INT8/CoreML export tests stay green.

## EDGES
- E1 A file swapped between `resolve_weights` and the loader on the EXPORT path must fail, not convert.
- E2 An unpinned model must still export (M3, A4).
- E3 All three branches, not only the two that already hold `meta` (A2).

## CHECKS
names reconciled to the tests actually built, 2026-09-10. The builder's worktree was cut from `main`
and never saw the names authored here, so it chose its own — see specs/method M7. Three rules the
build left genuinely unbound (M2, A5, R:CONVERTFIRST) were bound at review; their red evidence came
from reverting the exporter, and is recorded in the commit.
- test_export_detection_branch_makes_a_registry_lookup · covers: A2 · the branch that made no
  registry call now makes one.
- test_export_threads_the_registry_pin_for_detection · covers: M1, A1 · the digest reaching the
  loader is the registry's own.
- test_export_threads_the_registry_pin_for_cls_and_obb · covers: A2 · both non-detection branches.
- test_export_threads_the_registry_pin_for_cls_and_obb[classify] · covers: E3 · the classification
  branch passes the same gate as detection.
- test_export_threads_the_registry_pin_for_cls_and_obb[obb] · covers: E3 · the OBB branch likewise.
- test_export_sends_no_pin_for_an_explicit_weights_path · covers: A1, A4 · a user's own checkpoint is
  not compared to the official digest.
- test_export_unpinned_registry_model_still_makes_an_explicit_no_pin_decision · covers: A4 · an
  unpinned entry yields an explicit no-pin decision, not an omission.
- test_export_unpinned_registry_model_still_makes_an_explicit_no_pin_decision[classify] · covers: M3,
  E2 · an unpinned model still exports — the common case, since every -cls entry is sha256=None.
- test_export_unpinned_registry_model_still_makes_an_explicit_no_pin_decision[obb] · covers: M3, E2 ·
  likewise for OBB.
- test_no_new_comparison_logic_was_authored_in_exporter · covers: M4, A3 · the exporter supplies the
  value and never re-implements the check.
- test_integrity_failure_passes_through_export_unwrapped · covers: A6, R:PARTIALCLAIM · one event,
  one message, on either path.
- test_load_verified_state_dict_signature_is_unchanged · covers: M4 · this node adds a caller, not a
  parameter.
- test_export_resolve_weights_signature_unchanged · covers: M2 · the mock sites stay valid.
- test_export_reuses_the_meta_it_already_fetched · covers: A5 · one registry lookup, not two.
- test_export_never_converts_before_the_pin_is_compared · covers: R:CONVERTFIRST, E1 · conversion is
  never reached on a mismatch.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
