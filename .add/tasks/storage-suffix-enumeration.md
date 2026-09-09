---
type: Task
title: The storage rule names its classes, like every other entry in the boundary
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/_weights.py
  - tests/unit/test_checkpoint_loader.py
gives:
  - S1 the storage rule's removal from `find_class`, and the boundary that remains
generated: { by: add/3.5.0, at: 2026-09-09 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:3c7c166f078112a2", receipt: /tasks/storage-suffix-enumeration.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:SUFFIX=confirm|R:DEADRULE=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:f751d2c951f0cc0c", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:187b7b694542c9bc" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:20267e5c1c1841b0" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/storage-suffix-enumeration.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:00a38d0ca2cd11c0", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:a41683f9b0ca9495" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/storage-suffix-enumeration.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/storage-suffix-enumeration.d/runs/2.md, brief: "sha256:a41683f9b0ca9495", reason: "The last shape-based rule is gone; find_class is exact set, then stub prefixes, then refuse. The evidence is the regression floor rather than the new checks: 53 passed, all 10 pinned variants load, 499/499 tensors identical to the pre-change loader, and the legacy code-execution probe is still refused before execution. The build verified the manifest's no-storage claim itself - 53 observations, zero storage classes - and went further, establishing empirically that torch's zip reader intercepts typed-storage names before find_class sees them, so the rule was only ever reachable on the legacy non-zip path. That refines A3 rather than contradicting it. It also caught its own false-red from a zip-format fixture and rebuilt it on raw pickle before finalizing." }
advised_by: security-reviewer
---
## CARD
goal: `torch.<X>Storage` is admitted by an enumerated list of storage classes, not by a name suffix.
why: found by the security residue lens while verifying `narrow-loader-allowlist`, as the last
  un-enumerated entry in the trust boundary that task just narrowed everywhere else.
  - `arch/_weights.py` — `_ALLOWED_STORAGE_SUFFIX = "Storage"` admits any `torch` attribute whose name
    ends in `Storage`, via `getattr(torch, name)`.
  - For a real checkpoint the rule is never even consulted: torch's own `UnpicklerWrapper` intercepts
    storage names before ours sees them. It only fires for a name torch did NOT intercept.
  - The reachable shape is `UntypedStorage`, which a checkpoint can ask to construct at an
    attacker-chosen size. That is memory exhaustion, not code execution — `checkpoint-loader`'s
    unpickler still blocks the latter.

bounded, and the bound is why this is a separate node rather than a HARD-STOP on
`narrow-loader-allowlist`'s gate:
  - no code executes, so the worst case is a process that dies allocating;
  - it requires a checkpoint that already passed digest verification, or an unpinned model;
  - and A3 of `narrow-loader-allowlist` explicitly KEPT the suffix rule as a closed generated family,
    a reading a human confirmed on 2026-09-09. Reversing it inside that node would have been a silent
    widening of a frozen decision.

the measurement that would enumerate it already exists: `scripts/measure_checkpoint_globals.py`
  records every global the corpus names, and reported that no storage class reached our `find_class`
  at all across the 10 pinned variants. So the enumerated list is likely to be small or empty, and
  the honest question this node answers is whether the rule can simply be deleted.
beat: done · next: add status

## RULES
<must>
- M1 No suffix rule survives in `find_class`. The boundary admits torch names by exact
  `(module, name)` pair only; `_ALLOWED_STORAGE_SUFFIX` is removed, not narrowed.
- M2 All 10 pinned detection variants still load, the classification and OBB paths still load, and
  tensor equivalence against the pre-change loader is unchanged.
- M3 If a storage class is ever genuinely needed, it is added to the exact set carrying the
  observation that put it there — the same change-request path as every other entry.
</must>
<reject>
- R:SUFFIX No trust rule may admit by the SHAPE of a name rather than by the name -> "SUFFIX"
- R:DEADRULE No rule may remain in the trust boundary on the strength of an assumption that it is
  needed, once a measurement says it is not -> "DEADRULE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may reinstate the rule; taking: a maintainer via
  change-request carrying an observation — identical to `_ALLOWED_TORCH` and `_ALLOWED_EXACT`
  -> a boundary with one rule anyone may restore is a boundary with a hole anyone may restore.
- A2 [which] covers: S1 · the request does not say whether to enumerate or delete; taking: DELETE.
  `scripts/checkpoint_globals_manifest.json` records 53 globals observed across 10 digest-verified
  checkpoints and 25 in-process models, and NOT ONE is a storage class — torch's own
  `UnpicklerWrapper` intercepts `*Storage` names before our `find_class` ever sees them · probe:
  removing the branch outright leaves all 10 pinned variants loading -> enumerating a set that
  measurement says is empty would leave a rule in the boundary that exists only to look thorough.
- A3 [when] covers: S1 · the request does not say what the rule was ever reachable FOR; taking: only
  a name torch did not intercept, where it would call `getattr(torch, "<X>Storage")` and could
  construct an attacker-sized `UntypedStorage` — memory exhaustion, never code execution, since
  `checkpoint-loader`'s unpickler still blocks that -> stating the severity precisely is what keeps
  this a `quick` node instead of a second HARD-STOP.
- A4 [absent] covers: S1 · the request does not say what happens if a real checkpoint DOES name a
  storage class after the removal; taking: it is refused by name, loudly, with the same message every
  other unlisted name gets, and re-measured deliberately -> the failure is visible and one line to
  fix, which is the trade `narrow-loader-allowlist` already made and a human already confirmed.
- A5 [order] covers: S1 · the request does not say what `find_class` looks like afterwards; taking:
  exact set, then stub prefixes, then refuse — one branch shorter than today
  -> removing a rule must not leave the remaining order subtly different from what the tests bind.
- A6 [experience] covers: S1 · the request does not say who notices; taking: nobody, if it is right —
  the visible outcome is that nothing changes for any shipped model, which is precisely why the
  regression floor and not the new check is the load-bearing evidence here
  -> a deletion whose only proof is a new assertion proves the assertion, not the deletion.

## PLAN
contract:
  - S1 `_ALLOWED_STORAGE_SUFFIX` and its branch in `_RestrictedUnpickler.find_class` are removed. The
    comment recording why torch storages never reach us stays, as the reason the rule is gone.
strategy: the regression floor IS the evidence. Run the corpus and the loader suite BEFORE and AFTER
  and show they are identical; the new checks only bind the rule's absence.
regression floor: `test_checkpoint_loader.py`, `test_arch_model.py`, `test_classify_weights.py`,
  `test_weight_integrity.py`, and `scripts/verify_every_pinned_variant_loads.py` — all green, and
  `scripts/verify_checkpoint_equivalence.py` unchanged at 499/499.

## EDGES
- E1 A checkpoint naming `torch.FloatStorage` directly through our `find_class` is refused (M1).
- E2 A look-alike (`torch.NotAStorage`, `torch.Storage`) is refused by name, no `getattr` attempted.
- E3 All 10 pinned variants plus the cls and obb paths still load — the load-bearing evidence (A6).

## CHECKS
- test_no_suffix_rule_survives_in_find_class · covers: M1, R:SUFFIX, R:DEADRULE · no `endswith` over a
  torch name remains anywhere in the boundary.
- test_a_storage_name_is_refused_by_name · covers: M1, A4 · refused without `getattr` being
  attempted on the torch module.
- test_a_storage_name_is_refused_by_name[real-storage-class] · covers: E1 · a checkpoint naming
  `torch.FloatStorage` through our `find_class` is refused. Only reachable on the LEGACY non-zip
  path: torch's zip reader intercepts typed-storage names first, established empirically by the
  build — a refinement of A3, not a contradiction of it.
- test_a_storage_name_is_refused_by_name[fictitious-lookalike] · covers: E2 · `torch.NotAStorage`
  is refused by name with no `getattr` attempted on the torch module.
- test_allowlist_is_unchanged_by_the_removal · covers: M3, A2 · the exact set gained nothing; a
  deletion must not smuggle an addition.
- test_every_pinned_variant_still_loads[yolo11n] · covers: M2, E3 · the regression floor, which is the
  real evidence for a deletion.
- test_every_pinned_variant_still_loads[yolo26x] · covers: M2 · the largest variant.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
