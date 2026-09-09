---
type: Task
title: The export path verifies its weights the way the inference path does
status: direction
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
beat: scaffold · next: author export-digest-threading's RULES, ASSUMPTIONS and CHECKS, then add freeze export-digest-threading

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
- test_export_detection_threads_the_registry_pin · covers: M1, A1, A5 · the branch that needed a new
  lookup actually makes it.
- test_export_classify_and_obb_thread_the_pin · covers: M1, A2, E3 · the two branches that already
  hold meta.
- test_export_refuses_a_file_swapped_after_resolution · covers: M1, R:CONVERTFIRST, E1 · conversion is
  never reached.
- test_export_unpinned_model_still_exports · covers: M3, A4, E2 · the common case for cls and obb.
- test_export_writes_no_new_comparison_logic · covers: M4, A3 · the export path calls the parent's
  gate rather than reimplementing it.
- test_export_resolve_weights_signature_unchanged · covers: M2 · the mock sites stay valid.
- test_export_mismatch_uses_the_resolution_message · covers: A6 · one event, one message.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
