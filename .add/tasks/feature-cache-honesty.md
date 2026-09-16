---
type: Task
title: Document the failure mode where users meet it; set the default to match
status: done
depth: standard
milestone: m4-honest-deployment
scope:
  - src/yowo/cache/
  - src/yowo/config.py
  - README.md
  - docs/user-guide.md
  - docs/experiments/
  - tests/unit/test_cache.py
  - tests/unit/test_feature_cache_recall.py
  - scripts/measure_feature_cache.py
gives:
  - S1 the FeatureCache recall contract — when a cached entry may answer a query, and the smallest change the fingerprint is guaranteed to see
  - S2 the stated bound and the savings figure, at every place a user meets the feature
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: human, direction: "sha256:4cfd208f8ea915ac", binding: "sha256:f462ede9487468e4" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:3b1e46caf9702314" }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: human, direction: "sha256:8df0f318ce353a53", binding: "sha256:0658a82398b736fa" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:236f441f35bc58de" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/feature-cache-honesty.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: process, outcome: PASS, receipt: /tasks/feature-cache-honesty.d/runs/1.md, brief: "sha256:0d9b67dfae08afd6" }
advised_by: security-reviewer
---
## CARD
goal: The cache's recall hole is bounded by construction and stated as a measured number wherever a user meets the feature, and the savings figure beside it is one somebody measured.
why: Measured 2026-09-16, all on the shipped default threshold of 0.01. The fingerprint is `current_tensor.mean(axis=(2, 3))` — three numbers for a 640x640 RGB frame — so the cache decides whether a scene changed from its average brightness alone. A 90x90 px object at maximum contrast is invisible (1.98% of the frame); at a realistic 0.15 contrast the invisible object is 165x165 px (6.65%); at 0.30 it is 116x116. A frame that is black over white matches a uniform grey one EXACTLY, because both average 0.5 — nothing about that failure is small. `check_and_load` has no shape check at all, so a 640x640 entry answers a 320x320 query and hands back 80x80 features for an input needing 40x40, and a B=1 entry answers a B=4 query because `np.abs((1,3) - (4,3))` broadcasts to (4,3) rather than raising. That batch case is on a SHIPPED path: `cache=True` appears in exactly two presets and one of them, `(CUDA_HIGH, VIDEO)`, also sets `batch_size=4`, so any video whose frame count is not a multiple of 4 ends on a partial batch. The guard that would stop the shape half exists — `_similarity.py:26` returns 1.0 on a shape mismatch — and `frame_similarity` has ZERO callers in `src/`. It is not untested dead code; `tests/unit/test_cache.py:32-42` tests it, which is how a shape guard stayed green and unreachable while the live path had none. The README states "60-85% compute savings" at `README.md:345` and `src/yowo/cache/README.md:3` with no experiment doc behind it, unlike every other headline figure in this repo.
beat: done · next: add status

## RULES
<must>
- M1 A cached entry never answers a query whose preprocessed tensor shape differs from the entry's, in any dimension.
- M2 A change confined to one region of the frame is compared against that region, not against an average spanning the whole frame.
- M3 The smallest change the fingerprint can miss is stated as a number that was measured, at every place a user meets the feature.
- M4 A compute-savings figure appears in the docs only if a dated experiment doc measured it after the fingerprint this release ships.
- M5 A test fails if the measured blind spot grows beyond the number the docs state.
- M6 The shipped comparison path is the one with the shape guard; no guard exists in this module that the live path does not use.
- M7 No preset enables the feature cache on a device where enabling it was measured to cost time.
- M8 Where the docs state a saving they state, beside it, the blind spot that buys it — the two are one number seen from opposite sides.
</must>
<reject>
- R:BROADCAST two differently-shaped fingerprints are never broadcast into agreement -> "BROADCAST"
- R:DILUTE a local change is never averaged across the whole frame before being compared to the threshold -> "DILUTE"
- R:UNBACKED no compute-savings figure appears in any document without a dated experiment behind it -> "UNBACKED"
- R:DEADGUARD no guard in this module is reachable only from tests -> "DEADGUARD"
- R:MEASURED_LOSS a default never turns on a feature that was measured to be slower on that device -> "MEASURED_LOSS"
- R:HALFTRADE a saving is never stated without the cost that buys it -> "HALFTRADE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the box does not say whose frames these are; taking the caller to be sequential inference over ONE source — a camera or a video — because `check_and_load` is keyed by `source_id` and `pipeline/__init__.py:199` already disables the cache for mixed-source batches -> if wrong and one source_id can carry unrelated scenes, source identity is not enough to make an entry answerable and the key needs more than a name
- A2 [who] covers: S2 · the box says "every place a user meets the feature"; taking that to be `README.md`, `src/yowo/cache/README.md`, `docs/user-guide.md` and the `cache` field's own docstring in `config.py` — the four places the string is reachable without reading the implementation · probe: a check must enumerate those places and fail when one states no bound -> if wrong and a fifth surface exists, the box is true of four and false of the feature
- A3 [which] covers: S1 · the box does not say which changes must be caught; taking the guarantee to be about a change CONFINED TO A CELL at a stated contrast, not about every possible change, because a difference spread thinly enough to leave every cell mean intact is invisible to any pooled fingerprint and saying otherwise would be the same overclaim in a new size -> if wrong and the guarantee must be per-pixel, no pooled fingerprint can hold it and the cache has to compare tensors
- A4 [which] covers: S2 · the box does not say which number to state; taking BOTH the max-contrast bound and a realistic-contrast bound, because a single number at max contrast reads as the whole story and is the most flattering one available -> if wrong, one extra number is noise
- A5 [when] covers: S1 · the box does not say when an entry stops being answerable; taking shape equality as an absolute precondition checked BEFORE any distance is computed, rather than a large distance, because a shape change is not a matter of degree -> if wrong and a resize should be tolerated by re-pooling, the cache would return features of the wrong spatial size and the head would consume them
- A6 [absent] covers: S1 · the box does not say what a first-ever frame for a source means; taking absence of an entry as a plain miss with no warning, because it is the normal first iteration of every stream -> if wrong, every stream logs a warning on frame one
- A7 [absent] covers: S2 · the box does not say what to do when no experiment has been run yet; taking an absent measurement to mean the figure MUST NOT appear, rather than appearing with a caveat, because a hedged number is still the number a reader quotes -> if wrong, the docs lose a figure users were relying on
- A8 [order] covers: S1 · the box does not say what happens when shape and content both change; taking shape to be checked first and to short-circuit, so a shape mismatch can never be rescued by a small content distance -> if wrong, the two orderings differ only in wasted work, not in verdict
- A9 [order] covers: S2 · the box does not say which number leads where a bound and a saving appear together; taking the BOUND first and the saving second, because the saving is why someone enables the feature and the bound is what it costs them -> if wrong, the ordering is cosmetic
- A10 [experience] covers: S1, S2 · the box does not say who is harmed; taking the harmed party to be an operator whose detector silently returns nothing for an object it can see, so the failure must be stated as "an object this big may not be detected" in pixels rather than as a similarity threshold in abstract units · probe: the stated number must be in pixels at a named contrast, not a threshold value -> if wrong and the audience is a tuner rather than an operator, the threshold semantics matter more than the pixel size
- A11 [when] covers: S2 · the box does not say when the stated number goes stale; taking it that the number is pinned to the fingerprint that produced it, so a check must recompute it and fail on drift rather than trusting the prose · probe: changing the grid size must fail a check, not merely change behaviour -> if wrong, the docs and the code drift apart exactly as the 60-85% figure already has

## DECISIONS
Two decisions shaped this node and BOTH WERE ANSWERED BY THE USER on 2026-09-16,
before any code was written. Recorded here because the three decisions on the
previous node had to be taken in silence, and the difference matters.

- HOW FAR THE FINGERPRINT FIX GOES. Asked with three options: close the shape
  and batch holes only; close them AND replace the global mean with a coarse
  spatial grid; or that plus defaulting the cache off everywhere. ANSWERED:
  close the holes and add the grid. The middle option is the one the box's word
  "bounded" requires — closing only the shape and batch holes leaves a
  black-over-white frame matching uniform grey, which is not a bounded failure.
  The cache stays ON in the two presets that set it; this node does not change
  that default, and the user was shown that it would not.
- THE 60-85% SAVINGS FIGURE. Asked: re-measure and restate, withdraw the number
  and describe the mechanism, or leave it as someone else's problem. ANSWERED:
  re-measure and restate. So the figure must come from the fingerprint THIS
  release ships, not from the one that produced the current claim, and it gets
  a dated `docs/experiments/` entry like every other headline figure here.

A THIRD DECISION, asked DURING the build because the measurement contradicted
the premise of the first. Shown the table below, the user chose to drop
`cache=True` from the `(APPLE_SILICON, VIDEO)` preset and to leave
`(CUDA_HIGH, VIDEO)` alone, and to replace the "60-85%" claim with the full
grid rather than a single sentence.

    device  thr     off       on        saving    blind spot at that threshold
    cpu     0.01    34.50ms   40.22ms   -16.6%    40x40 px
    cpu     0.05    33.17ms   18.55ms   +44.1%    92x92 px
    cpu     0.10    36.02ms   17.14ms   +52.4%    130x130 px
    mps     0.01     6.27ms   19.52ms  -211.3%    40x40 px
    mps     0.05     6.27ms   12.08ms   -92.7%    92x92 px
    mps     0.10     6.38ms   11.75ms   -84.1%    130x130 px

The hit path copies roughly 6.4 MB of neck features host-to-device per hit and
a miss copies them back, which is why a faster device loses harder. "60-85%"
is not reachable at any threshold measured; the best is +52.4%, CPU only, at a
130x130 px blind spot. RESIDUAL, recorded rather than assumed: `(CUDA_HIGH,
VIDEO)` keeps `cache=True` and NOBODY HAS MEASURED IT — there is no CUDA
device here, and changing a default on a guess is the thing this milestone
exists to stop.

ACCEPTED COST, stated plainly because the user accepted it: a per-cell maximum
at the same 0.01 threshold registers a given scene change roughly 8x higher
than a whole-frame average did, so the cache will hit less often and save less.
The saving is what the measurement is for. A stricter fingerprint that never
hits would satisfy every check here except `test_an_identical_frame_still_hits`,
which exists for exactly that reason.

## PLAN
contract: `_similarity.py` stops exporting a comparator nothing calls. It publishes `spatial_fingerprint(tensor, grid)` returning `(B, C, G, G)` per-cell means computed from an integral image, so every pixel lands in exactly one cell whatever `H` and `W` are, and `fingerprint_distance(a, b)` which REFUSES a shape mismatch before comparing and otherwise returns the MAXIMUM per-cell absolute difference rather than the mean over the frame. `FeatureCache` stores the entry's tensor shape beside its fingerprint and refuses a query whose shape differs. `similarity_threshold` keeps its name and its 0.01 default but changes meaning — worst cell, not whole-frame average — and that is documented as a change, not slipped in.
strategy: The shape and batch refusals first: they are unbounded holes and they are cheap. The grid fingerprint second, because it changes hit rate and therefore the savings figure. The measurement last, because the number the docs state must come from the code this release ships — `scripts/measure_feature_cache.py` produces it and a dated `docs/experiments/` entry records it, which is the convention every other headline figure in this repo already follows.

## EDGES
- E1 `H` or `W` not divisible by the grid (a 641-pixel input) — every pixel must fall in exactly one cell and no cell may be empty.
- E2 the final partial batch of a video at `batch_size=4`, which is the shipped `(CUDA_HIGH, VIDEO)` preset — a B=2 query against a B=4 entry.
- E3 a frame byte-identical to the cached one must still HIT, or the feature is worthless.
- E4 a black-over-white frame against a uniform grey one — identical global mean, nothing in common — must MISS.
- E5 a source_id seen for the first time — a plain miss, no warning.
- E6 a frame smaller than the grid (an 4x4 input at grid 8) — no division by zero and no silent collapse to a whole-frame average.
- E7 the `(CUDA_HIGH, VIDEO)` preset, which keeps `cache=True` on a device nobody here can measure — it must be recorded as unmeasured, not quietly treated as measured-good.

## CHECKS
- test_a_shape_change_is_refused_before_any_distance_is_computed · covers: M1, A5, A8 · stores a 640x640 entry and queries with 320x320, asserting a miss rather than 80x80 features for a 40x40 query
- test_a_batch_change_is_refused_instead_of_broadcast · covers: M1, R:BROADCAST, E2 · stores B=4 and queries B=2, the shipped preset's final partial batch, asserting the numpy broadcast that made them compare equal cannot happen
- test_a_change_confined_to_one_cell_is_not_averaged_away · covers: M2, R:DILUTE, E4 · a black-over-white frame against uniform grey at an identical global mean must MISS, where the shipped fingerprint reports them identical
- test_the_measured_blind_spot_is_no_larger_than_the_documented_number · covers: M5, M3, A11 · sweeps object size until the cache stops hitting and fails if the measured pixel bound exceeds what the docs state
- test_every_place_the_feature_is_documented_states_the_bound · covers: M3, A2, A10 · reads the four documents and fails when one describes the cache without the pixel bound at a named contrast
- test_no_savings_figure_appears_without_a_dated_experiment · covers: M4, R:UNBACKED, A7 · greps the docs for a percentage beside the cache and requires a `docs/experiments/` file that measured it
- test_the_shipped_path_uses_the_guard_this_module_defines · covers: M6, R:DEADGUARD · asserts no comparator in `yowo.cache` is reachable only from tests, which is how a shape guard stayed green and unused
- test_every_pixel_lands_in_exactly_one_cell · covers: E1, E6 · fingerprints a 641x641 and a 4x4 input and asserts the cell partition covers the frame without empty cells or a divide by zero
- test_an_identical_frame_still_hits · covers: E3 · the cache must remain useful; a stricter fingerprint that never hits would satisfy every other check here
- test_a_first_frame_for_a_source_is_a_plain_miss · covers: A6, E5 · no entry, no warning, just a miss
- test_no_preset_enables_the_cache_on_a_device_measured_to_lose · covers: M7, R:MEASURED_LOSS, E7 · reads the preset table and fails if a device the experiment recorded as slower still ships with the cache on
- test_a_stated_saving_is_accompanied_by_the_blind_spot_that_buys_it · covers: M8, R:HALFTRADE · fails when a document gives a cache saving percentage without a pixel bound in the same section

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
