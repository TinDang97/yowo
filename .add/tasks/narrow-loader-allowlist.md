---
type: Task
title: Enumerate the torch layer classes the allowlist admits, instead of a prefix
status: direction
depth: standard
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/arch/_weights.py
  - tests/unit/test_checkpoint_loader.py
  - scripts/
gives:
  - S1 the enumerated torch-class allowlist in `arch/_weights.py`, and the refusal for anything outside it
  - S2 `scripts/measure_checkpoint_globals.py` — the measurement that produced the list
depends_on:
  - /tasks/checkpoint-loader.md
  - /tasks/ci-weight-fixture.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:da482b07bcbb4659", receipt: /tasks/narrow-loader-allowlist.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|R:PREFIX=confirm|R:GUESSED=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:d1c08876f3db2e82", binding: "sha256:e9a79d98e3503d91" }
advised_by: security-reviewer
---
## CARD
goal: The checkpoint allowlist names the torch layer classes it admits, rather than admitting a namespace.
why: `checkpoint-loader` froze `torch.nn.modules.` as a PREFIX rule, which admits any class in that namespace. Every one is data-bearing so nothing is currently wrong, but it is the broadest entry in the trust boundary and the only one that is not enumerated. Narrowing it needs the real set measured across all ten shipped variants — which is why this depends on `ci-weight-fixture` for reachable checkpoints, rather than being guessed from the single `yolo11n.pt` available when the loader was built.
beat: scaffold · next: author narrow-loader-allowlist's RULES, ASSUMPTIONS and CHECKS, then add freeze narrow-loader-allowlist

## RULES
<must>
- M1 The unpickler admits torch layer classes from an enumerated `(module, name)` set. No
  `torch.nn.modules.` prefix rule survives, and neither does the `torch.nn.parameter` prefix beside it.
- M2 Every entry in that set was OBSERVED, not reasoned about: either named by one of the 10
  digest-pinned detection checkpoints, or constructed by our own `YOLOModel` / `ClassifyModel` /
  `OBBModel` across every size, walked in process.
- M3 A `torch.nn.modules.*` name that is not enumerated is refused exactly as any other unlisted name
  is, with a message naming the class and pointing at the change-request path.
- M4 The measurement is reproducible and recorded: a committed script regenerates the set, and the
  digest of every checkpoint it measured is recorded beside the list.
- M5 All 10 pinned detection variants, and the classification and OBB paths, still load after the
  narrowing.
</must>
<reject>
- R:PREFIX No trust rule may admit a namespace where it could name a class -> "PREFIX"
- R:GUESSED No entry may be added without a recorded observation behind it -> "GUESSED"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may widen the list; taking: a maintainer, through
  a change-request that carries a new observation — the same rule `checkpoint-loader` already set for
  `_ALLOWED_EXACT` -> a list anyone may append to mid-build is not a boundary.
- A2 [which] covers: S1, S2 · the request does not say which corpus defines the set; taking: the 10
  digest-pinned detection variants, fetched through `resolve_weights` so every measured byte was
  verified, UNION an in-process sweep of our own three model classes across all five sizes. The
  `-cls` and `-obb` registry entries carry `sha256=None`, so measuring them would mean fetching
  unverified bytes to define a security boundary; the arch sweep covers their layer types instead
  (`Linear`, `Dropout`, `AdaptiveAvgPool2d`, `Flatten`) from code we own · probe: the classification
  and OBB load paths stay green
  -> measuring a trust boundary from unverified bytes lets the corpus choose the allowlist, and
  measuring detection alone would refuse every classification checkpoint (E1).
- A3 [when] covers: S1 · the request does not say where the boundary falls for the neighbouring rules;
  taking: `torch.*Storage` stays a suffix rule — it is a closed, generated family — while
  `torch.nn.parameter` becomes the exact pair `("torch.nn.parameter", "Parameter")`, because as a
  prefix it also admits `torch.nn.parameterfoo`
  -> removing one prefix while leaving a second is R:PREFIX with fewer characters.
- A4 [absent] covers: S1 · the request does not say what an unlisted torch class does; taking: refused,
  with the same `ModelLoadError` shape as any other unlisted name (human decision, 2026-09-09). A
  future upstream variant using a new layer type therefore fails loudly and is re-pinned deliberately
  -> a warn-and-admit path makes the enumeration advisory, which is what it already is today.
- A5 [order] covers: S1 · the request does not say the check order in `find_class`; taking: exact set,
  then storage suffix, then stub prefixes, then refuse — a single set lookup replaces the prefix scan
  -> `find_class` runs once per global in the pickle stream; the narrowing must not make it slower.
- A6 [experience] covers: S1 · the request does not say who reads the refusal; taking: someone whose
  new upstream checkpoint just stopped loading, so the message names the exact class and points at
  this node -> "not a weights primitive" alone tells them nothing they can act on, and the tempting
  next move is to look for a flag that disables the check.
- A7 [which] covers: S2 · the request does not say where the measurement runs; taking: a maintainer
  script under `scripts/`, not CI — the corpus is ~440 MB and a per-run download would be the thing
  someone eventually disables · probe: the unit suite asserts the committed list against the recorded
  manifest and downloads nothing -> putting the measurement in CI is how the check gets turned off.

- A8 [who] covers: S2 · the request does not say who runs the measurement; taking: a maintainer
  re-pinning or adding a variant, on a machine with network and disk for the corpus — never a
  contributor on an ordinary build and never a CI job (A7)
  -> a script anyone runs by accident becomes a 440 MB surprise in someone's test run.
- A9 [when] covers: S2 · the request does not say when it is re-run; taking: whenever a pin changes or
  a variant is added to the registry — the two events that can introduce a new class
  -> re-measuring on no schedule at all lets the recorded manifest drift silently away from the pins.
- A10 [absent] covers: S2 · the request does not say what a checkpoint it cannot fetch or verify means;
  taking: the script FAILS and emits nothing — never a partial set
  -> a set measured over 7 of 10 variants would narrow the allowlist past what the shipped models need,
  and would look identical to a complete one.
- A11 [order] covers: S2 · the request does not say how the output is ordered; taking: sorted by
  `(module, name)`, so re-measuring after a pin bump produces a clean diff
  -> an unordered dump makes "what changed" unanswerable, which is the only question the re-run asks.
- A12 [experience] covers: S2 · the request does not say who reads the output; taking: a maintainer
  deciding whether to widen the list, so each measured class is printed with the variants that named it
  -> a bare set of names gives no basis for the judgement the change-request (A1) demands.

## PLAN
contract:
  - S1 `_ALLOWED_TORCH: frozenset[tuple[str, str]]` replaces `_ALLOWED_PREFIXES`, checked by the same
    set lookup as `_ALLOWED_EXACT`. `_ALLOWED_STORAGE_SUFFIX` and `_STUBBED_PREFIXES` unchanged. Each
    entry keeps the "why it is here" comment convention `checkpoint-loader` established.
  - S2 `scripts/measure_checkpoint_globals.py` — resolves each pinned variant through
    `resolve_weights`, drives `torch.load` with a find_class that RECORDS rather than refuses, unions
    the in-process arch sweep, and prints both the set and the digest of every checkpoint measured.
  - S3 a recorded manifest committed beside the list: which digests were measured, and when.
strategy: measure FIRST — the list cannot be authored before it exists — then narrow, then the refusal
  path. The red tests are written against the narrowing, not against the measured contents.
regression floor: `test_checkpoint_loader.py`, `test_arch_model.py`, `test_classify_weights.py`,
  `test_weight_integrity.py` stay green; `tests/integration` still loads yolo26n.

## EDGES
- E1 A checkpoint naming `torch.nn.modules.linear.Linear` — no detection variant uses it, the
  classification head does (A2).
- E2 A look-alike prefix (`torch.nn.parameterfoo.X`, `torch.nn.modulesX.Y`) must be refused (A3).
- E3 A checkpoint naming `torch.nn.modules.module.Module` itself.
- E4 All 10 pinned variants still load (M5).

## CHECKS
- test_no_rule_admits_a_torch_namespace · covers: M1, R:PREFIX · neither `_ALLOWED_PREFIXES` nor any
  `startswith` over a torch namespace survives.
- test_allowlist_matches_the_recorded_measurement · covers: M2, M4, R:GUESSED · every entry traces to a
  recorded observation; no entry is unaccounted for.
- test_unlisted_torch_class_is_refused · covers: M3, A4, E3 · an unlisted torch layer is refused like any other name.
- test_refusal_names_the_class_and_the_change_path · covers: M3, A6 · the message is actionable.
- test_lookalike_torch_prefixes_are_refused · covers: A3, E2 · `torch.nn.parameterfoo.X` does not slip through.
- test_classification_layers_are_admitted · covers: A2, E1 · the cls head loads.
- test_measurement_script_downloads_nothing_in_the_unit_suite · covers: A7 · the corpus never enters CI.
- test_every_pinned_variant_still_loads · covers: M5, E4 · integration tier, not the unit suite.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
