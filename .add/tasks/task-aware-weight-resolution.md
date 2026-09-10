---
type: Task
title: resolve_weights returns the weight for the spec's task, and every registered model is pinned
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/models/
  - tests/unit/
  - scripts/
gives:
  - S1 the mapping from a ModelSpec to the weight file that spec actually needs
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:6b2ccad889487674", receipt: /tasks/task-aware-weight-resolution.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:WRONGASSET=confirm|R:UNMEASURED=confirm|R:COLLIDE=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:3c856bae5573c791", binding: "sha256:403f71e509d37f74" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:6ed5f34052fa8401" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/task-aware-weight-resolution.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/task-aware-weight-resolution.d/runs/1.md, brief: "sha256:bf04efc2f1875b53", reason: "Twelve checks green on a bound receipt, every rule proven. Verified independently rather than on the builder's word: a classify spec now resolves yolo11n-cls.pt with pin c62d41bf9625, an obb spec yolo11n-obb.pt with b62898ebf389, detect and taskless are unchanged, and a typo'd task raises 'Unknown task classifiy for model yolo11n. Registered tasks: classify, detect, obb' instead of silently fetching a detection weight - A4 and M6 as the human chose them. R:UNMEASURED is discharged by method, not by assertion: all 25 pins were written programmatically from scripts/weight_digests.json, which the measuring script produced by downloading each registered URL and hashing what it served, and the 10 detection digests already in the registry reproduce from that sweep exactly - so the 15 new ones rest on a method proved against known-good answers. The builder proved M4/R:COLLIDE on disk across all 25 entries rather than assuming weight_stem uniqueness, and left the two pin checks deliberately red rather than inventing a digest, which is the correct refusal. MY ERROR, recorded: the frozen node and its interview were authored, interviewed, advised and frozen but never committed, so the builder's worktree could not see them and it read them out of the shared checkout. It said so. This is the same class as method M7 in a new form - it is not enough for the direction to be frozen, it has to be COMMITTED before a worktree is cut. Both are committed now. Known follow-on outside this node's scope: resolving and digest-verifying a real -cls weight now succeeds and the checkpoint loader then refuses it, because the file names torchvision.transforms.transforms.Compose and the allowlist was measured from the ten detection checkpoints only. That is narrow-loader-allowlist's declared change-request path and needs its own measurement. Two further findings recorded for the user: engine.py:_resolve_model_meta and export/_exporter.py each still carry their own task dispatch with no unknown-task branch, so three copies of one decision now exist where get_for_task was built to be the only one." }
advised_by: artifact-integrity-steward
---
## CARD
goal: `resolve_weights` returns the asset the spec's TASK names, and every model in every registry carries a SHA-256 measured from the file that URL actually serves.
why: `resolve_weights` calls `get(spec.family, spec.size)` at `models/_weights.py:124` regardless of `spec.task`, so a `classify` or `obb` spec resolves the DETECTION checkpoint. `engine.py:447` is the call site every engine uses, so `yowo classify SOURCE --model yolo11n-cls` with no `--weights` cannot work. Reproduced 2026-09-10: loading the detection `yolo11n.pt` into a classification model raises `RuntimeError: Shape mismatches` at `backbone.c2psa.cv1.conv.weight`, and into an OBB model raises the same after WARNING about 42 missing keys — "model may produce incorrect results". Had the shapes lined up, OBB would have loaded a silently wrong model behind a warning. Classification shipped in v2.2.2 and OBB in v2.4.0; neither has ever worked through the registry. Three independent signs it was never exercised: the maintainer's weight cache holds only detection files, every test in `test_classification_engine.py` patches `yowo.engine.resolve_weights` to `/fake/cls.pt`, and `tests/integration/` has no classify coverage at all. This also explains the `sha256=None` on all 10 `-cls` and all 5 `-obb` entries cleanly — those pins were never added because those URLs were never fetched. All 15 URLs are correct and live; HEAD-probed 2026-09-10, every one 200, 5.8 MB to 118.4 MB.
beat: done · next: add status

## RULES
<must>
- M1 `resolve_weights` returns the asset the spec's task names: `detect` the detection weight, `classify` the `-cls` weight, `obb` the `-obb` weight. The task selects the registry; family and size select the entry within it.
- M2 Every model in every registry carries a pinned SHA-256, measured by downloading that exact URL and hashing the bytes it served.
- M3 The check asserting M2 iterates EVERY registry. `list_available()` returns only the 10 detection metas, and a check that walks it cannot see the 15 entries this task exists to pin.
- M4 A weight's cache path is distinct per task, so a detection and a classification weight for the same family and size cannot overwrite each other on disk.
- M5 `resolve_weights(spec, cache_dir=None) -> Path` is unchanged in signature and return type. Roughly sixty tests mock it and three callers depend on the shape.
- M6 An unknown or unregistered task fails loudly, naming the task and what is registered. It never silently falls back to detection — that fallback IS this defect.
</must>
<reject>
- R:WRONGASSET Returning a weight whose task differs from the spec's, on any branch. -> "WRONGASSET"
- R:UNMEASURED A digest recorded from anywhere but a download of that exact URL — copied from another variant, taken from a local file of unknown origin, or transcribed from a third party. -> "UNMEASURED"
- R:COLLIDE Two tasks resolving to one cache path. -> "COLLIDE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose call sites must keep working; taking every existing caller — `engine.py:447`, `export/_exporter.py:94`, `benchmark/_runner.py:74`, and the ~60 tests that mock it — since M5 forbids a signature change and a task-blind caller must keep resolving detection -> if wrong, a caller that relied on always getting the detection weight breaks silently.
- A2 [which] covers: S1 · the request does not say which registries are in scope; taking all three — `_REGISTRY`, `_CLS_REGISTRY`, `_OBB_REGISTRY` — because pinning ten of twenty-five is what produced a check that reads as complete and covers a tenth -> if wrong, effort is spent pinning assets nobody resolves.
- A3 [when] covers: S1 · the request does not say where the task is read; taking `spec.task` at the top of `resolve_weights`, BEFORE the registry lookup, so one dispatch decides both the URL and the pin and they cannot disagree -> if wrong and the task is consulted after the lookup, the pin and the URL can come from different entries, which is worse than today.
- A4 [absent] covers: S1 · the request does not say what an absent or unrecognised `spec.task` means; taking `None` or missing as `detect`, matching `ModelSpec`'s own default, but an unrecognised STRING as an error naming what is registered -> if wrong, a typo'd task silently downloads a detection weight, which is exactly today's bug wearing a different hat. · probe: a bogus task must raise and name the registered tasks.
- A5 [order] covers: S1 · the request does not say what happens when `spec.weights_path` is set and a task is given; taking `weights_path` first and unchanged — an explicit file is the user's own and no registry entry describes it -> if wrong, a user's fine-tuned checkpoint is compared against an official digest and every custom run refuses.
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking someone who typed `--model yolo11n-cls` and got a detection weight — the message must name the task, the model, and the URL it went to, so the mismatch is visible rather than surfacing 200 lines later as a shape error in a conv layer -> if wrong, the next person debugs the architecture instead of the resolution.

## PLAN
contract: `resolve_weights(spec, cache_dir=None) -> Path`, unchanged. Internally it dispatches on `spec.task` to `get` / `get_cls` / `get_obb` and uses THAT meta for both the URL and the pin. The 15 missing digests are measured by a script that downloads each registered URL and records the SHA-256 of the bytes served, alongside the URL, the date and the byte count — the same evidence shape `reproducible-sdist` records for its build digests. The measuring script is committed so the numbers can be regenerated rather than trusted.

## EDGES
- E1 `ModelSpec(family=yolo11, size=n, task="classify")` — resolves `yolo11n-cls.pt`, not `yolo11n.pt`. The reproduction.
- E2 The same for `task="obb"`, which today warns about 42 missing keys before failing on shape.
- E3 `task="detect"` and a spec with no task at all — both unchanged, still the detection weight.
- E4 A detection and a classification weight for one family and size, both cached — distinct paths, neither overwriting the other.
- E5 An unrecognised task string — raises, naming the task and the registered ones.
- E6 `spec.weights_path` set alongside a non-detect task — returns the user's file, no pin, no registry lookup.
- E7 A pinned `-cls` weight that fails verification — refused exactly as a detection weight is; the new entries inherit the existing control rather than getting a weaker one.

## CHECKS
all in `tests/unit/test_task_aware_resolution.py`.
- test_a_classify_spec_resolves_the_cls_weight · covers: M1, A3, E1 ·
  the reproduction: a `task="classify"` spec resolves `yolo11n-cls.pt`, not the detection weight,
  and carries that entry's own pin.
- test_an_obb_spec_resolves_the_obb_weight · covers: M1, E2 ·
  the branch that today warns about 42 missing keys before failing on shape.
- test_a_detect_spec_and_a_taskless_spec_are_unchanged · covers: A1, A4, E3 ·
  every existing caller keeps resolving what it resolved before.
- test_no_branch_returns_a_weight_for_another_task · covers: R:WRONGASSET ·
  swept over every registered task, not only the two that are broken today.
- test_every_registry_entry_carries_a_pinned_digest · covers: M2, A2 ·
  all 25 entries across all three registries, not the 10 `list_available()` returns.
- test_the_pinned_digest_check_cannot_pass_by_walking_one_registry · covers: M3 ·
  the check above is proved to see every registry, so it cannot read as complete while covering
  a tenth — the exact way the old one passed.
- test_every_recorded_digest_names_its_measurement · covers: R:UNMEASURED ·
  each new pin is traceable to a download of that URL, with date and byte count.
- test_a_detection_and_a_classify_weight_do_not_share_a_cache_path · covers: M4, R:COLLIDE, E4 ·
  one family and size, two tasks, two files on disk.
- test_an_unrecognised_task_raises_naming_what_is_registered · covers: M6, A4, A6, E5 ·
  a typo'd task fails at resolution naming the task and the model, not 200 lines later as a shape
  error in a conv layer.
- tests.unit.test_task_aware_resolution::test_resolve_weights_signature_is_unchanged · covers: M5 ·
  the ~60 mock sites and three real callers stay valid. Qualified with its module: this name is
  also defined in `test_verified_digest_threading.py`, and a bare citation resolves to AMBIGUOUS
  the moment a receipt runs both files (method M10).
- test_an_explicit_weights_path_wins_over_any_task · covers: A5, E6 ·
  a user's own checkpoint is returned unpinned, whatever the task says.
- test_a_pinned_cls_weight_that_fails_verification_is_refused · covers: E7 ·
  the new entries inherit the existing integrity control rather than getting a weaker one.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
