---
type: Task
title: A deprecation mechanism, used, and a written stability policy
status: done
depth: standard
sensitivity: architecture
milestone: m5-declare-ga
scope:
  - src/yowo/_deprecation.py
  - src/yowo/types.py
  - src/yowo/__init__.py
  - docs/stability-policy.md
  - tests/unit/test_deprecation.py
gives:
  - S1 the deprecation mechanism — how a public name is marked, and what a caller sees
  - S2 the first name marked with it, and the proof that the warning fires
  - S3 the written stability policy — what "public" means here and how long a marked name survives
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: interview, authority: human, interview: "sha256:98c72b73ac20ed3e", receipt: /tasks/deprecation-policy.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|R:SILENT_REMOVAL=confirm|R:BROKEN_BY_WARNING=confirm|R:HANDLIST=confirm" }
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:2ed32151fe17a716", binding: "sha256:5c2d49cc61b91d2c" }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "Build found four things the Direction did not foresee, three of them defects in checks or tooling. (1) STACKLEVEL IS 4, NOT 3, and it had to be measured rather than reasoned: the chain is warn_deprecated, then __post_init__, then the __init__ the dataclass decorator GENERATES (which lives in <string>), then the caller. A value of 3 blames types.py, which tells a user nothing about their own code. (2) TWO CHECKS COUNTED A STRING WHERE THEY MEANT A CALL. One asserted the mechanism module mentions DeprecationWarning exactly once and got 3, because the docstring explains it; the other -- inherited from export-result-contract -- reported types.py as a second mechanism because its docstring says the name warns. Both now walk the AST and count actual warnings.warn(DeprecationWarning) call sites. (3) A CHECK FROM THE PREVIOUS NODE CORRECTLY FIRED. test_this_node_invents_no_deprecation_mechanism asserted no DeprecationWarning existed anywhere in src/, which was right while the mechanism was undesigned. This node legitimately built one, so the guard was updated to state the same INTENT against the world that now exists -- exactly one mechanism, and no second one outside _deprecation.py -- rather than deleted or weakened. (4) MY MUTATION HARNESS PRODUCED FALSE VERDICTS. Two policy mutations read as 'killed' while the check was already red for an unrelated regex bug of my own. A mutation killed by an already-failing check is not a verdict -- the same defect as reading 'no tests ran' as green, one polarity over. The harness now asserts a GREEN BASELINE before trusting any row. SWEEP RESULT, stated honestly: 8 mutations, 6 killed, 2 survived. The two survivors are BADLY-TARGETED MUTATIONS rather than weak checks: each replaces only the first of 3 and 2 occurrences in the policy document, so the property it tests genuinely still holds afterwards. The valuable property -- that the policy and the DEPRECATIONS record cannot drift apart -- is falsifiable and was killed by the 'deprecation record emptied' mutation." }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:2ed32151fe17a716", binding: "sha256:5c2d49cc61b91d2c" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:1ac070079e8fb49f" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/deprecation-policy.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:ee253a555c38b22c", binding: "sha256:5c2d49cc61b91d2c" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:923fcf8240770623" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/deprecation-policy.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: human, outcome: PASS, receipt: /tasks/deprecation-policy.d/runs/2.md, brief: "sha256:923fcf8240770623" }
advised_by: task-planner
---
## CARD
goal: A public name can be retired without breaking anyone: it warns, keeps working, and the warning says what to use instead — and a test proves the warning fires.
why: There is **zero `DeprecationWarning` anywhere in `src/`**, verified 2026-09-16. The package is at 2.5.0 on PyPI with 191 public names across 19 modules, so semver already promises stability, and there is no channel to announce a planned removal through.
  THIS IS ALREADY BLOCKING THREE THINGS, none hypothetical. (1) `yowo.types.ExportResult` is a 7-field public dataclass, in `yowo.__all__`, unit-tested, and constructed by NOTHING in `src/` — `export_model` returns the 24-field `ExportMetadata`. It cannot be retired because removal is a break with no warning channel (issue #84). (2) `engine.py:143` records a public type change — widening `PRECISION_UNKNOWN` from `str` to `str | None` to match `memory_pct: float | None` on the same dataclass — explicitly "deferred to the deprecation policy", a policy that does not exist. (3) m4 was constrained to ADDITIVE-ONLY for this reason, recorded in its milestone on 2026-09-16: it may add names and fix behaviour, never remove or rename one.
  THE BOX MAKES THE MECHANISM THE GATE, not the prose: "at least one `DeprecationWarning` in `src/`, with a test asserting it fires. The written stability policy accompanies it as a review item." So a policy document alone does not discharge it; something must actually warn.
beat: done · next: add status

## RULES
<must>
- M1 A helper exists that emits a `DeprecationWarning` naming the name, when it was deprecated, when it will be removed, and what to use instead.
- M2 At least one public name uses it, and a check asserts the warning actually fires.
- M3 A deprecated name keeps working: still importable, still constructible, behaviour unchanged.
- M4 The warning points at the CALLER's line, not at this package's internals.
- M5 A written stability policy states what counts as public, how long a deprecated name survives, and which semver step removes it.
- M6 Deprecations are recorded in one structure a check can enumerate, so one cannot be removed early or forgotten.
</must>
<reject>
- R:SILENT_REMOVAL no public name is removed without having warned in a released version first -> "SILENT_REMOVAL"
- R:BROKEN_BY_WARNING marking a name deprecated never changes what it does -> "BROKEN_BY_WARNING"
- R:HANDLIST no check over the deprecations reads a hand-maintained copy of them -> "HANDLIST"
</reject>

## ASSUMPTIONS
- A1 [which] covers: S2 · the box says "at least one"; taking the first subject to be `ExportResult`, because it is the only public name already established as returned by nothing and it already carries a docstring saying so · probe: is anything in `src/` constructing it? · found: NO — the only hits are its definition in `types.py`, the import and `__all__` entry in `__init__.py`, and tests -> if wrong and something constructs it, the package would warn its own users
- A2 [who] covers: S1, S2 · the box does not say whose code the warning is for; taking the audience to be a package CONSUMER on 2.5.0, not a contributor, so the message must name a replacement and a version rather than a rationale -> if wrong and it is for contributors, a comment would have done and the machinery is waste
- A3 [when] covers: S2 · the box does not say when the warning fires; taking it as fired on CONSTRUCTION rather than on import, because nothing returns `ExportResult` so the only way to use it is to build one, and an import-time warning would need module `__getattr__` and a restructure of where the class is defined · probe: does the dataclass allow `__post_init__`? · found: YES — `@dataclass(frozen=True)` with no slots -> if wrong and importers must be warned too, someone who imports and never constructs hears nothing
- A4 [absent] covers: S1 · the box does not say what happens when the removal version arrives; taking it that the mechanism RECORDS a removal version but never enforces it by raising, because a library that starts raising on a date is worse than one that keeps warning -> if wrong, names accumulate forever and the policy is decorative
- A5 [order] covers: S1 · the box does not say how many times a warning fires; taking Python's default once-per-location as correct and NOT overriding it, because a warning inside a loop that fires every iteration gets silenced wholesale by the user -> if wrong, a caller who only runs the line once in a rare branch may never see it
- A6 [experience] covers: S1, S2 · the box does not say what a user sees; taking a `DeprecationWarning`, which CPython hides by default outside `__main__`, as correct rather than a louder `UserWarning`, because that is the category the ecosystem filters on and library authors do not get to decide a consumer's noise level -> if wrong, most users never see it until their test suite turns warnings into errors
- A7 [who] covers: S3 · the box does not say who the policy is for; taking the reader to be someone deciding whether to depend on this package, so the policy states what is public and what is not — the 191 names are not all equal -> if wrong and it is an internal contributor doc, it belongs in CONTRIBUTING.md
- A8 [which] covers: S3 · the box does not say what "public" means; taking it as a name in a module's `__all__`, because that is the only definition already mechanically checkable here (`scripts/check_public_surface.py` exists) -> if wrong, an underscore-free name not in `__all__` is treated as private while users treat it as public
- A9 [when] covers: S3 · the box does not say how long a name survives; taking it as at least one MINOR release with a warning before removal in the next MAJOR, because 2.5.0 is already published and semver is what an adopter is relying on -> if wrong and this is too slow, dead names sit in the surface for a whole major cycle
- A10 [absent] covers: S3 · the box does not say what happens to a name nobody deprecated but everybody stopped using; taking silence as meaning it stays public, because the alternative is removal by neglect -> if wrong, the surface only ever grows
- A11 [order] covers: S2 · the box does not say whether the first deprecation must be the most deserving; taking `ExportResult` as first because it is the clearest case rather than the most valuable, since the mechanism is the deliverable and the subject is its proof -> if wrong, effort goes to a name whose retirement matters less
- A12 [experience] covers: S3 · the box calls the policy a REVIEW ITEM and the mechanism the gate; taking the policy as a document that must exist and be accurate but is not itself mechanically checked beyond existing and naming the pieces -> if wrong, the policy drifts from the mechanism with nothing to notice
- A13 [absent] covers: S1 · the box does not say what a caller does who wants the old behaviour without the warning; taking it that no suppression hook is provided, because `warnings.filterwarnings` is the standard one and inventing another is surface -> if wrong, users suppress our category wholesale and miss later ones
- A14 [order] covers: S2 · the box does not say whether the warning may change what the deprecated name DOES; taking behaviour as strictly unchanged, because a deprecation that also changes semantics is two breaks announced as one -> if wrong, nothing

- A15 [which] covers: S1 · the box does not say what KIND of marking the mechanism supports; taking it as covering a public NAME only -- a class, function or constant -- and NOT a parameter, a return type or a behaviour, because those need different machinery and the box asks for one warning that fires · probe: what does the first subject need? · found: a name -> if wrong, `engine.py:143`'s deferred `str` -> `str | None` widening still has no channel and waits again
- A16 [when] covers: S1 · the box does not say when a deprecation is RECORDED relative to when it warns; taking the record as the single source -- the warning is built from it at call time rather than written out separately -- so a message and its record cannot disagree -> if wrong and messages are hand-written per site, the record becomes documentation of what the warnings once said
- A17 [absent] covers: S2 · the box does not say what happens if the first subject turns out to be wrong to deprecate; taking the mark as REVERSIBLE -- deleting the record and the `__post_init__` call restores the name untouched, because nothing else changes -> if wrong and marking is one-way, choosing the first subject becomes a much heavier decision than the box implies
- A18 [order] covers: S3 · the box does not say whether the policy must precede or follow the mechanism; taking the policy as written LAST, describing what exists, because this node's own `why:` records a policy cited by two source comments for months while never existing -> if wrong and the policy should lead, the mechanism risks being shaped by an unreviewed document

## PLAN
decided-by-the-author: Four decisions were put to the author on 2026-09-16 and ANSWERED, not timed out. (1) SURVIVAL: a deprecated name warns for at least one MINOR release and is removed no sooner than the next MAJOR. This promises nothing new -- it makes explicit what semver already implies for a published 2.5.0 -- and gives adopters a whole major cycle. Two minors was considered and rejected as leaving dead names in the surface that m5's GA declaration would inherit. (2) THE WARNING FIRES ON CONSTRUCTION, not on import. Nothing returns `ExportResult`, so constructing one is the only way to use it; an import-time warning would require moving the class behind a private name so module `__getattr__` fires, restructuring `types.py` for one name, and would fire on this repo's own test imports and on `check_public_surface.py`'s introspection -- noise with no corresponding use. (3) PUBLIC MEANS A NAME IN A MODULE'S `__all__`. It is the only definition already mechanically checkable here, and `scripts/check_public_surface.py` already enforces it. (4) A RECORDED REMOVAL VERSION IS A PROMISE, NOT AN ENFORCEMENT: when it arrives the name keeps warning and removal stays a human decision. Raising at runtime was rejected because it converts a deprecation into a surprise break for anyone who upgraded for unrelated reasons; failing the build was rejected because it ties a release to unrelated cleanup.
contract: `src/yowo/_deprecation.py` publishes a frozen `Deprecation` record (name, since, removed_in, instead) and `warn_deprecated(record)`, plus `DEPRECATIONS`, a mapping a check can enumerate. `yowo.types.ExportResult` gains a `__post_init__` that calls it — the class, its seven fields, its `__all__` entry and its behaviour are otherwise untouched. `docs/stability-policy.md` states what is public (a name in a module `__all__`), that a deprecated name warns for at least one minor release and is removed no sooner than the next major, and that the removal version recorded in `DEPRECATIONS` is a promise rather than an enforcement.
strategy: The mechanism and its record first, because the box gates on the mechanism. `ExportResult` second, as its first subject and its proof. The policy document last, so it describes something that exists rather than something intended — this node's own `why:` records a policy that was cited for months while not existing.

## EDGES
- E1 `DeprecationWarning` is hidden by CPython's default filters outside `__main__`; the check must set its own filter rather than rely on the ambient one, or it passes without the warning ever being visible.
- E2 the warning must point at the CALLER's line — a `stacklevel` that names `types.py` tells a user nothing about their own code.
- E3 nothing inside `src/` may use a deprecated name, or the package warns its own users about its own internals.
- E4 a removal version that has already passed — the record must still be loadable and still warn.
- E5 a deprecated name must keep working under `python -W error::DeprecationWarning` for anyone who has NOT opted into that, i.e. the warning is the only change.
- E6 the package claims `requires-python >=3.8`; the mechanism must not use syntax newer than that.

## CHECKS
- test_the_deprecation_warning_actually_fires · covers: M2, A3, E1 · the box's gate, and it must set its own filter because CPython hides this category by default
- test_the_warning_names_the_replacement_and_both_versions · covers: M1, A2 · a warning that says only "deprecated" leaves the caller with no action
- test_a_deprecated_name_still_works_exactly_as_before · covers: M3, A14, E5, R:BROKEN_BY_WARNING · the whole point is that nothing breaks yet
- test_the_warning_points_at_the_callers_line · covers: M4, E2 · a stacklevel naming our own module is useless to the person who must act
- test_nothing_inside_the_package_uses_a_deprecated_name · covers: A1, E3 · otherwise the package warns its own users about its own internals
- test_every_deprecation_is_enumerable_from_one_record · covers: M6, R:HANDLIST · a check that reads a second copy of the list cannot notice the first one changing
- test_a_removal_version_that_has_passed_still_warns_rather_than_raises · covers: A4, E4, R:SILENT_REMOVAL · a library that starts raising on a date is worse than one that keeps warning
- test_the_stability_policy_exists_and_names_the_mechanism_it_describes · covers: M5, A7, A8, A9, A12 · this node's own why records a policy cited for months while not existing
- test_the_mechanism_marks_a_name_and_not_a_parameter_or_a_behaviour · covers: A15 · ADDED during build: engine.py's deferred `str` -> `str | None` widening is STILL deferred after this node ships, and someone will otherwise assume the policy unblocked it
- test_the_mechanism_parses_under_the_claimed_python_floor · covers: E6 · the package claims 3.8 and `scripts/check_public_surface.py` already enforces that claim elsewhere

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
