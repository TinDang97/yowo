---
type: Task
title: Postprocess cost bounded independent of scene content
status: done
depth: deep
milestone: m4-honest-deployment
sensitivity: architecture
scope:
  - src/yowo/postprocess/
  - src/yowo/config.py
  - src/yowo/engine.py
  - src/yowo/obb_engine.py
  - src/yowo/cli/_main.py
  - src/yowo/_convenience.py
  - src/yowo/benchmark/_baseline.py
  - tests/unit/test_bounded_postprocess.py
gives:
  - S1 the bounds as configuration — `max_nms`/`max_det` on `InferenceConfig`, `top_k`/`max_det` on `OBBConfig`, and the path each takes from config to its NMS call
  - S2 the bounded behaviour — which detections survive when a bound binds, and the latency ceiling at each path's real anchor count
  - S3 what the mAP gate records as having determined its number
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-16, act: freeze, authority: plan, direction: "sha256:1c246b36aeff9806", binding: "sha256:75e6eee7fa0bb0c6" }
  - { by: "builder", at: 2026-09-16, act: replan, authority: process, note: "Build found four things the Direction did not foresee. (1) An EXISTING test, test_no_top_k_field, asserts OBBConfig must NOT have a top_k field -- its docstring says 'unlike ClassificationConfig', so it exists to stop two different meanings of top_k colliding on sibling configs (class-label ranking vs a candidate cap). The OBB bound was renamed top_k -> max_nms, which respects that decision AND makes both paths symmetric: max_nms + max_det on each. A good catch by a test that was written for a different reason. (2) MapBaseline's loader validates on the field.type STRING, so 'int | None' fell through to the string branch and rejected the bound; it now handles an optional int, because None means unbounded and is a real value -- rejecting it would make the unbounded configuration unrecordable. (3) The fixture's own _rerecord note points at a printout that rounds to 6 decimals while the fixture stores 17, so the documented procedure could not reproduce the artifact it documents; a full-precision line was added. (4) Two more hand-maintained lists of the milestone's recurring defect class: the required-field parametrize list in test_map_baseline.py (now derived from dataclasses.fields) and _MUST_MATCH in _baseline.py, whose own comment records this exact miss happening to map_50/map_75 before -- a new check now fails if a recorded determinant is not compared. MUTATION SWEEP: 8 mutations, 4 survived on the first pass and ALL FOUR were genuinely weak checks rather than double-guarding -- the axis post-cap was never asserted, the OBB check counted probiou CALLS (which max_det alone bounds) rather than their WIDTH (which only the pre-cap bounds), the tie check tested repeat-stability which argpartition satisfies, and the OBB stability check compared two calls of the same mutated function. All eight now killed." }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:0b26d23d68a0ca8f" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/bounded-postprocess.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: plan, direction: "sha256:3016756b1e4d0c36", binding: "sha256:75e6eee7fa0bb0c6" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:342df8f67f4d3a5c" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/bounded-postprocess.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-16, act: refreeze, authority: plan, direction: "sha256:77fc9770308da48d", binding: "sha256:75e6eee7fa0bb0c6" }
  - { by: "cli", at: 2026-09-16, act: brief, authority: process, brief: "sha256:7f7f7adcb02638ab" }
  - { by: "process:run", at: 2026-09-16, act: run, authority: process, outcome: PASS, receipt: /tasks/bounded-postprocess.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-16, act: gate, authority: plan, outcome: PASS, receipt: /tasks/bounded-postprocess.d/runs/3.md, brief: "sha256:7f7f7adcb02638ab" }
advised_by: edge-reliability-operator
---
## CARD
goal: Postprocess latency at each path's real anchor count is bounded by configuration rather than by scene content, on both the axis-aligned and the oriented path — and the bound that ships is defensible by measurement.
why: Measured 2026-09-16 by three parallel read-only audits. Both paths are unbounded in TIME, and the milestone box's account of why is wrong in five separate ways, each of which would have sent the build somewhere useless.
  THE COST LAW (axis-aligned), measured: cost ~= 2.0 ns x C x K + 0.6 us x C, where C is candidates surviving the confidence filter and K is boxes surviving NMS. It is O(C*K), NOT O(C^2) — which is why 8400 candidates collapsing to 1666 boxes costs 5.56 ms while 8400 collapsing to 8395 costs 51.64 ms (my own measurement, `_class_aware_nms` alone; the agent measured 131-190 ms for the full decode). Because K <= C, ONE pre-NMS cap on C bounds NMS, the lexsort and the `BoundingBox` loop together. `max_det` alone leaves 123.67 ms and bounds none of the O(C*K) term.
  THE COST LAW (oriented), measured: `_nms_rotated` is O(K_kept x N) probiou pairs, worst case O(N^2). But the binding constant is that `probiou_matrix(1 x M)` costs 83 us even at M=1 — ~20 separate elementwise torch ops, dispatch-bound below M~2000. So ANY Python-loop design has a floor of 83us x kept: 25 ms at max_det=300, 83 ms at max_det=1000.
beat: done · next: add status

## RULES
<must>
- M1 At each path's real anchor count with every anchor surviving the confidence filter, postprocess completes within a declared ceiling.
- M2 That ceiling is asserted at the anchor count the SHIPPED runtime actually produces, established from code, not from the box.
- M3 A bound that does not bind changes nothing: at the shipped defaults the detections are identical to unbounded.
- M4 When a bound does bind, it discards only lower-scoring candidates — never a higher-scoring one in favour of a lower.
- M5 The mAP gate records every input that determined its number, the new bounds included.
- M6 The oriented bound holds at the DEFAULT confidence, not only under a non-default flag.
- M7 Each bound is reachable from config, environment and CLI on the path it governs, or its absence there is named.
</must>
<reject>
- R:SYNC_STORY the oriented fix is never described, in code or commit, as removing a device sync -> "SYNC_STORY"
- R:NSQUARED no design whose peak memory grows with the square of the anchor count -> "NSQUARED"
- R:UNRECORDED_DETERMINANT the mAP gate never compares a number across a determinant it does not record -> "UNRECORDED_DETERMINANT"
</reject>

## ASSUMPTIONS
- A1 [which] covers: S2 · the box says "each path's real anchor count" and names 21504 for the oriented path; taking the real count to be 8400 for BOTH paths · probe: read the registry and run a forward pass · found: CORRECTED — `_make_obb_meta` pins `input_height=640, input_width=640` (`_registry.py:258-275`), and a live forward gives `(1,20,8400)` at 640 and `(1,20,21504)` only at 1024. `OBBConfig` has NO imgsz field; the `imgsz=640` at `config.py:401` is `ExportConfig`, export-only. The shipped runtime CANNOT produce 21504 -> a check written against 21504 would assert a bound on a configuration no user can reach
- A2 [which] covers: S2 · the box attributes the oriented cost to an `int(order[i].item())` device sync per candidate; taking that as FALSE on the shipped path · probe: is the tensor ever on a device? · found: REFUTED — `obb_engine.py:231` is `torch.from_numpy(raw_output)` and `infer()` returns `NDArray[np.float32]`, so the tensor is ALWAYS CPU. The sync costs 0.619 us/call = 0.15% of a 3.3 s call, and removing it changed nothing beyond noise (3306 vs 3297 ms). The math is ~70%. The claim WOULD hold on MPS (33874 -> 5249 ms at N=1000), which is a trap to avoid, not today's bug -> building the box's fix would have cost days and bought 0.15%
- A3 [which] covers: S2 · the box says the exposure is "reachable with one documented flag: `--confidence 0.01`"; taking that as refuted and the real trigger as different · probe: measure conf 0.01 on real images, and find what actually pins C · found: REFUTED — conf 0.01 costs 1.37 ms on the worst of 300 real COCO images; 68 ms needs C~5800, never observed. The real trigger is `--confidence 0.0`: validation permits the CLOSED interval (`config.py:173`), the filter is `>=` (`_nms.py:277`), and the CLI option is a bare `type=float` with no floor (`cli/_main.py:164`), so EVERY anchor survives regardless of scene — 0.14 -> 24.57 ms on one real image. On the ORIENTED path no flag is needed at all: 2935 ms at the default conf=0.25 with a dense scene -> the box points at a flag that is not the hole
- A4 [absent] covers: S2 · the box records memory as "FINE, concern RETIRED at 0.697 KB/candidate"; taking that as true of TODAY'S loop and false of the obvious replacement · probe: measure peak RSS of a vectorized rotated NMS · found: CONFIRMED-AND-QUALIFIED — today's loop is 10.5 MB at 8400, but the ultralytics-style `triu` vectorization peaks at 1694 MB at 8400 and 6791 MB at 21504, and it CHANGES RESULTS (strictly more aggressive than sequential greedy: 11 kept -> 4 on a clustered N=200) -> "memory is retired" invites exactly the fix that reintroduces it
- A5 [which] covers: S1 · the box treats `cv2.dnn.NMSBoxes(top_k=)` as a free latency win; taking it as a RESULT-CHANGING pre-NMS cap · probe: run it against overlapping and disjoint boxes · found: CONFIRMED by my own run — `top_k=4` returned 2 boxes, not 4. It truncates the score-sorted CANDIDATE list before suppression, so low-scoring DISJOINT boxes are discarded outright. It is `max_nms`, not `max_det` -> a 70x win recorded as free is a recall cut nobody costed
- A6 [who] covers: S1, S2 · the box does not say whose latency this is; taking the harmed party to be an edge deployment on a fixed frame budget, where a 131 ms spike on one dense frame drops frames, rather than a batch job where mean throughput is what matters -> if wrong and the reader is a batch user, a bound that cuts recall to protect p99 is the wrong trade
- A7 [who] covers: S1 · the box does not say who sets these; taking the bounds to be user-facing configuration with defaults that bind, not internal constants, because the default must be revisable without a code change -> if wrong and they should be constants, the whole thread-through below is wasted
- A8 [when] covers: S1 · the box does not say when the bound applies; taking it as applied per IMAGE inside postprocess, not per batch, because a per-batch cap would let one dense image starve the others · probe: ultralytics ships a per-batch `time_limit` that abandons later images mid-batch -> if wrong, a batch of 8 shares one budget and image 8 silently returns nothing
- A9 [absent] covers: S1 · the box does not say what "no bound" means; taking `None` to mean unbounded and to remain reachable, so today's behaviour is available byte-for-byte, and NOT `0`, which reads as "keep nothing" · probe: this repo already ruled this way once — `rtsp-reconnect-correctness.md:63` (A13) chose `None` for unbounded, distinct from `0` -> if wrong, the capability that exists today is removed with no replacement
- A10 [order] covers: S2 · the box does not say what order survivors come back in; taking confidence-descending as already guaranteed (`_nms.py:191-198`, `np.lexsort` with primary `-scores[kept]`) and REQUIRED, because a post-NMS `max_det` that truncates an unordered list would discard arbitrarily -> if wrong, `max_det` silently keeps whichever boxes happened to sort first
- A11 [order] covers: S2 · the box does not say how ties are resolved at the cap boundary; taking exact score ties as a real hazard, because the mAP fixture asserts bit-identical results across darwin/arm64 and ubuntu/x86-64 · probe: compare numpy `argpartition` against cv2's `stable_sort` on a tie-heavy input · found: THEY DISAGREE — symmetric difference 6 of 300. `argpartition` is repeat-stable on one numpy build but is not contract-stable across versions. Likewise `topk()` may order ties differently from `argsort` on the oriented path -> a faster pre-cap that silently reorders ties breaks the cross-platform bit-identity the mAP gate rests on
- A12 [experience] covers: S2 · the box does not say what a user should see when a bound binds; taking it as SILENT — no warning per frame — because a bound that binds on every frame of a dense video would emit a warning per frame and drown the log, and the bound is a declared configuration rather than an error -> if wrong, a user whose recall silently dropped has nothing to tell them why
- A13 [experience] covers: S3 · the box does not mention the mAP gate; taking the reader of a baseline to be someone comparing two numbers months apart, for whom an unrecorded determinant is indistinguishable from a real regression -> if wrong and baselines are never compared across config changes, the record costs a format break for nothing
- A14 [when] covers: S3 · the box does not say when the baseline is re-recorded; taking it as re-recorded IN THIS NODE, because the gate must pass on the same commit that introduces the bounds -> if wrong, main is red between the two commits
- A15 [absent] covers: S3 · the box does not say what an OLD baseline means once the fields exist; taking a missing field as UNLOADABLE rather than defaulted-to-unbounded, because a baseline recorded under a bound and one recorded without would otherwise be indistinguishable · probe: `load_baseline` already refuses a record missing any field (`_baseline.py:106-113`) -> if wrong, every user with a stored baseline must re-record
- A16 [who] covers: S3 · the box does not say who owns the baseline format; taking it as this repo's own fixture, not a published artifact other projects consume, so a format break costs a re-record here and nothing outside -> if wrong, a downstream consumer's baseline stops loading with no deprecation channel
- A17 [order] covers: S1 · the box does not say which bound applies first; taking pre-NMS `max_nms`/`top_k` BEFORE suppression and post-NMS `max_det` AFTER, because reversing them makes `max_det` bound nothing and `max_nms` cost the full O(C*K) -> if wrong the two knobs are applied in an order where only one does anything
- A18 [experience] covers: S1 · the box does not say whether these are two knobs or one; taking them as two with different jobs — `max_nms` is where the LATENCY is, `max_det` is where the COMPATIBILITY safety is (measured: max_nms 131->3.7 ms, max_det alone 131->123.67 ms) — so collapsing them into one would either not bound or over-cut -> if wrong, users face two knobs where one would do

- A19 [which] covers: S3 · the box does not say WHICH determinants the gate must record; taking it as the two new bounds only, not a general audit of every input, because `MapBaseline` already names the others (model, backend, device, precision, confidence, iou, image set) and widening the field set further is a separate decision with its own cost · probe: `_baseline.py:55-78` lists the existing required fields -> if wrong and other determinants are also missing, the gate stays partly blind after this node claims it is not
- A20 [when] covers: S2 · the box does not say WHEN the ceiling is asserted — on every run, or only in a dedicated benchmark; taking it as a unit check that runs in ordinary CI on every PR, with a ceiling generous enough to absorb a slower shared runner, because a bound checked only in a nightly benchmark regresses silently for a day · probe: this machine showed up to 2.2x run-to-run variance under thermal load, and CI is ubuntu x86-64 and unmeasured -> if wrong, the ceiling is either flaky in CI or so loose it asserts nothing
- A21 [order] covers: S3 · the box does not say whether a baseline's field ORDER or its absence is what identifies it; taking the record as a mapping where a missing key is fatal and key order is irrelevant, matching how `load_baseline` already reads it · probe: `load_baseline` refuses a record missing a field (`_baseline.py:106-113`) -> if wrong and order is significant, re-recording reorders keys and every stored baseline stops matching

## PLAN
decided-by-me: Three decisions were put to the author on 2026-09-16 and timed out after 300s. They are MY calls, not an approval, and each is cheap to revisit. (1) DEFAULTS THAT BIND: axis-aligned `max_nms=1000` / `max_det=300`, oriented `top_k=2048` / `max_det=1000`. Measured mAP cost on this repo's own pinned COCO 500 at the gate's exact config is -0.000053, roughly 100x inside the +/-0.005 band, for a 35x worst-case latency cut; the oriented defaults are 1000 rather than ultralytics' 300 because 300 cuts recall to 39.2% of objects on DOTA-scale scenes while 1000 holds 82-84% at 35-51 ms. An unbounded-by-default option was available and rejected because the ORIENTED path reaches 2935 ms at the DEFAULT confidence, so shipping it unbound leaves the worst exposure open to someone who never sets a flag. (2) THE mAP BASELINE: add the bounds to `MapBaseline` as REQUIRED fields and re-record `coco_map_baseline.json`. This breaks a shipped file format and there is no third option — not adding them leaves the gate comparing a number across a determinant it cannot see, which is the defect class the file's own docstring was written against. (3) `--confidence 0.0`: leave validation ALONE. With `max_nms` bound it is no longer a latency cliff, and rejecting 0.0 would remove a currently-valid public configuration with no deprecation mechanism in this repo to announce it through.
contract: `InferenceConfig` gains `max_nms: int | None = 1000` and `max_det: int | None = 300`; `OBBConfig` gains `top_k: int | None = 2048` and `max_det: int | None = 1000`. `None` means unbounded and keeps today's behaviour byte-for-byte. Each threads config -> env -> CLI -> convenience fn -> the postprocess signature, which is public (`yowo.postprocess` is in `scripts/check_public_surface.py`). On the axis-aligned path `max_nms` caps candidates by score BEFORE `_class_aware_nms` and `max_det` truncates after the existing lexsort. On the oriented path `top_k` caps before the greedy loop and `max_det` breaks out of it. `MapBaseline` gains both bounds as required fields and the fixture is re-recorded in this node.
strategy: The measurements first, because every default here rests on one and the box's own numbers did not survive re-verification. The axis-aligned path second — it is the one with a labelled corpus to measure mAP against. The oriented path third, where the bound must be checked with a DENSE synthetic tensor: a check fed sparse output passes on a broken implementation. The baseline re-record last, so the gate and the bounds land together.

## EDGES
- E1 N=0 on the oriented path — today returns a `(0,)` **int64** tensor on `boxes.device`; a naive `torch.tensor([])` returns float and breaks the `.tolist()` indexing at `_obb_nms.py:197`.
- E2 probiou self-IoU is **0.999532, never 1.0** — two byte-identical boxes score 0.9995, so any threshold >= 0.9996 makes duplicates un-suppressible. A test asserting `IoU == 1.0` would be wrong.
- E3 degenerate zero-area boxes (`w=h=0`) — the `+ eps` at `_obb_nms.py:84` prevents div-by-zero and returns 0.9995, no NaN. Must not regress.
- E4 exact score ties straddling the cap boundary, on BOTH paths — numpy `argpartition` vs cv2 `stable_sort` disagree 6-of-300; `topk()` vs `argsort` may too.
- E5 `--confidence 0.0` — every anchor survives regardless of scene, C pinned at the anchor count forever.
- E6 `auto_letterbox=True` on a 16:9 source — the tensor is 384x640 and the count is **5040**, not 8400, so the ceiling must be stated against the maximum and not assumed constant.
- E7 a randomly-initialised model via the public `ModelBuilder` / custom-`num_classes` surface — emits a uniform score everywhere, so C = the full anchor count at ANY threshold.
- E8 `OBBConfig.num_classes` above ~2000 — `_obb_nms.py:189` uses a FIXED `* 10000.0` class offset that destroys float32 coordinate resolution (measured: class 5000 gives stored dx=0.0000 and probiou 0.999532, a false suppression). The axis-aligned path is already adaptive (`_nms.py:161-163`). Found while probing; not created here.
- E9 a bound set to `None` — must be indistinguishable from today, on both paths.
- E10 a bound of 1 — must return at most one detection without dividing by zero or emptying the result.

## CHECKS
- test_the_axis_aligned_path_is_bounded_at_the_full_anchor_count · covers: M1, M2, E7 · every one of 8400 anchors survives the confidence filter, which is the only input that exercises the O(C*K) term at its ceiling
- test_the_oriented_path_is_bounded_at_the_full_anchor_count · covers: M1, M2, M6, R:NSQUARED · fed a DENSE raw tensor at the default confidence, because a check fed sparse output passes on a broken implementation
- test_the_real_anchor_count_is_read_from_the_shipped_registry_not_assumed · covers: M2, A1 · the box said 21504 for the oriented path and the shipped runtime cannot produce it
- test_an_unset_bound_changes_nothing · covers: M3, A9, E9 · a bound that alters results when it is None would be a silent behaviour change for every existing user
- test_a_bound_discards_only_lower_scoring_candidates · covers: M4, A10, A17 · the ordering guarantee is what makes a post-NMS truncation defensible rather than arbitrary
- test_ties_at_the_cap_boundary_are_broken_deterministically · covers: A11, E4 · the mAP fixture asserts bit-identical results across two platforms, and a faster pre-cap that reorders ties breaks exactly that
- test_the_map_baseline_records_the_bounds_that_determined_its_number · covers: M5, A13, A15, R:UNRECORDED_DETERMINANT · a gate that cannot see a determinant cannot fail on it
- test_both_bounds_are_reachable_from_config_env_and_cli · covers: M7, A7 · a knob that exists only on the dataclass is not reachable by the user the default was chosen for
- test_the_oriented_path_has_no_device_sync_to_remove · covers: R:SYNC_STORY, A2 · pins the measured fact that the tensor is always CPU, so nobody re-derives the box's wrong diagnosis
- test_the_bounded_oriented_path_does_not_allocate_quadratically · covers: R:NSQUARED, A4 · the vectorized replacement peaks at 6.8 GB at 21504, and the box records memory as a retired concern
- test_an_empty_or_single_candidate_survives_both_bounds · covers: E1, E10 · the empty oriented return is int64 on the input device and a float replacement breaks indexing downstream
- test_probiou_self_similarity_is_not_one · covers: E2, E3 · records the 0.999532 self-IoU and the zero-area guard so a later threshold change cannot silently make duplicates un-suppressible
- test_a_letterboxed_source_has_fewer_anchors_than_the_ceiling · covers: E6 · the ceiling is a maximum, not a constant, and a check that assumed 8400 always would pass vacuously on a 16:9 stream
- test_every_determinant_the_baseline_records_is_also_compared · covers: M5, R:UNRECORDED_DETERMINANT, A19 · ADDED during build: `_MUST_MATCH` is hand-maintained and `_baseline.py` already records this exact miss happening to `map_50`/`map_75`, so a recorded-but-uncompared field is a live defect class here, not a hypothetical
- test_the_class_offset_stride_follows_the_boxes_not_a_fixed_constant · covers: E8 · ADDED during build: the fixed 10000.0 stride was a live false-suppression bug above ~class 2000, and the axis path already solved it
- test_the_opencv_top_k_argument_is_not_a_free_truncation · covers: A5 · ADDED during build: the box records this as a free win and nothing else would stop a reader adopting it as one
- test_the_bound_applies_per_image_not_per_batch · covers: A8 · ADDED during build: ultralytics ships a per-batch budget that abandons later images, so the alternative reading is one somebody actually shipped
- test_the_ceiling_is_checked_in_ordinary_ci_not_only_in_a_benchmark · covers: A20 · ADDED during build: a bound verified only nightly regresses silently for a day
- test_the_baseline_record_is_read_by_name_not_by_position · covers: A21 · ADDED during build: a re-record rewrites the file and key order must not decide whether the gate still loads
- test_confidence_zero_is_bounded_rather_than_rejected · covers: A3, E5 · the decision was to leave validation alone, so the bound must be what makes 0.0 survivable

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
