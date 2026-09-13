---
type: Task
title: Read the cgroup limit, not the host; OOM ladder for non-PyTorch
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/hardware/
  - src/yowo/backends/
  - src/yowo/engine.py
  - src/yowo/tune/_profile.py
  - tests/unit/
gives:
  - S1 yowo.hardware.effective_cpu_count() — the CPU budget this process may actually use
  - S2 yowo.hardware.detect_system_memory_mb() — the memory budget, capped by the cgroup limit
  - S3 the engine's memory-pressure monitor on a NON-CUDA backend, and HealthReport.memory_pct with it
depends_on:
  - /tasks/real-backend-smoke.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "unrecorded", at: 2026-09-13, act: interview, authority: human, interview: "sha256:981178b07454bd86", receipt: /tasks/container-aware-sizing.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A13=confirm|A14=confirm" }
  - { by: "unrecorded", at: 2026-09-13, act: interview, authority: human, interview: "sha256:981178b07454bd86", receipt: /tasks/container-aware-sizing.d/interviews/2.md, answers: "R:HOSTCOUNT=confirm|R:RAISES=confirm|R:STARVE=confirm|R:PHANTOMLADDER=confirm" }
  - { by: "Tin Dang", at: 2026-09-13, act: freeze, authority: human, direction: "sha256:d3767431a46d58a6", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:b84615ae505e28ad" }
  - { by: "builder", at: 2026-09-13, act: replan, authority: process, note: "CHECKS naming drift, corrected so the gate binds the tests that exist: the declared test_every_thread_sizing_site_reads_the_effective_count split during build into two checks with different strengths — test_no_sizing_site_derives_a_budget_from_the_host (the host probes are GONE) and test_every_sizing_site_calls_the_effective_count (the replacement is NAMED). A covers: key on a name no test carries binds nothing, which is lesson M12; the split is recorded rather than left to read as covered. Sixteen further checks were added during build and are declared here so their coverage binds too — chiefly test_the_pytorch_load_path_configures_threads, which exists only because the previous node's mutation sweep proved a helper can be verified in isolation while nothing on the real path calls it." }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:72d0038f45ac1963" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/container-aware-sizing.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:e0ece11f051576a2", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:9bba79a68a64e999" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/container-aware-sizing.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-13, act: refreeze, authority: human, direction: "sha256:51e82dd36427ecbc", binding: "sha256:60b3aed15d2d819b" }
  - { by: "cli", at: 2026-09-13, act: brief, authority: process, brief: "sha256:ad8820dd13fe1346" }
  - { by: "process:run", at: 2026-09-13, act: run, authority: process, outcome: PASS, receipt: /tasks/container-aware-sizing.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-13, act: gate, authority: human, outcome: PASS, receipt: /tasks/container-aware-sizing.d/runs/3.md, brief: "sha256:ad8820dd13fe1346" }
advised_by: edge-reliability-operator
---
## CARD
goal: Every sizing decision reads the limit the process was given, not the machine it landed on.
why: Measured 2026-09-13 in `docker run --cpus=2 --memory=512m` (cgroup v2): the container grants
  `cpu.max` = `200000 100000` (2.0 CPUs) and `memory.max` = 512 MB. `os.cpu_count()` returns **6**
  and `len(os.sched_getaffinity(0))` returns **6** — affinity sees a cpuset, never a quota — so
  `torch.set_num_threads(cpu_count // 2)` spawns 3 compute threads against a 2-CPU quota, and the
  same arithmetic runs in `_onnx.py` and `_tensorrt.py`. `/proc/meminfo` reports **12017 MB**
  against a 512 MB limit: a 23x overstatement of the budget, feeding `Device.memory_total_mb` and
  every capacity decision downstream. And `_start_oom_monitor` returns early unless the device is
  CUDA (`engine.py:589`), so a CPU container has no memory ladder at all — it is OOM-killed rather
  than degraded.
beat: done · next: add status

## RULES
<must>
- M1 `effective_cpu_count()` returns the CPU budget the cgroup grants where a quota is present, and
  every site that sizes threads or keys a cache on a CPU count reads it instead of `os.cpu_count()`:
  `backends/_pytorch.py`, `backends/_onnx.py`, `backends/_tensorrt.py`, and `tune/_profile.py`.
- M2 `detect_system_memory_mb()` returns the cgroup budget where a memory limit is present — total
  capped by the limit, available derived from limit minus current usage — and returns the host
  reading unchanged where no limit exists.
- M3 Both readers understand cgroup v2 AND cgroup v1, take their cgroup root as a parameter so a
  fixture directory can drive every case on any OS, and return the host reading rather than raising
  on a missing, unreadable or malformed file.
- M4 A non-CUDA backend running under a cgroup memory limit gets the same three-tier recovery ladder
  CUDA gets, driven by `memory.current / memory.max`, and `HealthReport.memory_pct` reports that
  fraction instead of `None`.
- M5 The milestone box records the measured host-vs-container numbers, and names the tune-cache
  invalidation this change causes rather than leaving a user to discover it.
</must>
<reject>
- R:HOSTCOUNT no thread-sizing site and no cache key derives a CPU budget from `os.cpu_count()` or
  affinity alone once a cgroup quota is readable -> "HOSTCOUNT"
- R:RAISES neither reader raises out of a missing, unreadable or malformed cgroup file -> "RAISES"
- R:STARVE an absent or unlimited limit never sizes the process down — not to 1 CPU, not to 0 MB
  -> "STARVE"
- R:PHANTOMLADDER the non-CUDA ladder never starts when no memory limit is readable, so it can
  never degrade against a denominator it invented -> "PHANTOMLADDER"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1,S2 · the box does not say whose cgroup is read — the process's own, or the
  container's; taking the reading that the root handed to the reader IS the budget, because a
  container shares one cgroup across every process in it and yowo creates no sub-cgroup of its own
  -> a sidecar sharing the limit is unaccounted and the budget is overstated for both.
  · probe: the shipped reader must take its root as a parameter, so the caller names the cgroup.
- A2 [who] covers: S3 · the box does not say who may trigger degradation on a non-CUDA backend;
  taking the reading that the existing three-tier ladder applies unchanged, because nothing in
  `_apply_oom_recovery` reads a CUDA handle -> tiers tuned against VRAM fire against host RAM.
  · probe: the non-CUDA path must reach `_apply_oom_recovery` with a fraction, not a new ladder.
- A3 [which] covers: S1 · the box does not say which limit is "the cpu count" when a quota and a
  cpuset both bind; taking the MINIMUM of affinity and ceil(quota), since either alone is a ceiling
  the process cannot exceed -> a cpuset-only container is sized to the host and oversubscribes.
  · probe: a quota of 2 on a 6-CPU host must yield 2, and affinity must still cap with no quota.
- A4 [which] covers: S2 · the box does not say which of MemTotal and the cgroup limit wins when both
  read; taking the MINIMUM -> the 23x overstatement survives on any host whose limit exceeds RAM.
- A5 [which] covers: S3 · the node's title says "non-PyTorch", which is not the boundary that
  matters; taking the reading that EVERY non-CUDA device gets the ladder — pytorch-cpu included,
  since it is OOM-killed identically -> the most common container deployment keeps no ladder.
- A6 [when] covers: S1,S2 · the box says "where present" and does not say which cgroup version that
  admits; taking BOTH, v2 first then v1 -> a v1 host (Jetson's stock JetPack among them) reads the
  host and the box is false exactly where this project deploys.
  · probe: a v1 fixture must yield the same numbers as its v2 equivalent.
- A7 [when] covers: S3 · the box does not say when the non-CUDA ladder starts; taking the reading
  that it starts ONLY when a memory limit is readable, because with no limit the denominator would
  be the host's MemTotal — the same lie this node exists to delete -> an unlimited container gets
  no ladder, which is correct: it has no bound to breach.
- A8 [absent] covers: S1 · the box does not say what an absent or unlimited quota means; taking it
  as "no container limit", falling back to affinity then `os.cpu_count()` -> every bare-metal host
  and every laptop is sized to one thread.
- A9 [absent] covers: S2 · cgroup v1 writes a huge sentinel rather than the word `max` for "no
  limit"; taking any value at or above 2**62 as unlimited -> a v1 host reads an 8-exabyte budget as
  its cap, and `min()` then hides the real host reading behind a number no machine has.
- A10 [absent] covers: S3 · the box does not say what an unreadable `memory.current` means mid-run;
  taking it as "no reading this tick" — skip, never degrade -> a transient read error is escalated
  into a batch halving and a DEGRADED health state.
- A11 [order] covers: S1,S2 · n/a · both surfaces are single idempotent reads returning one value;
  there is no sequence to order and no tie to break.
- A12 [order] covers: S3 · the box does not say what orders the tiers when one reading crosses
  several; taking the existing top-down ladder unchanged, one action per poll -> a container evicts
  streams AND halves its batch on a single tick, overshooting the recovery.
- A13 [experience] covers: S1,S2 · the box does not say who reads these and what would make it hard;
  taking the reading that the caller is a backend sizing threads and must NOT have to know cgroups
  exist -> the contract `os.cpu_count()` had is lost and every call site grows a branch.
- A14 [experience] covers: S3 · the box does not say what the operator diagnosing a container OOM
  sees; taking the reading that the log line must name the budget it read and where it came from
  -> a halved batch size appears with no stated cause, which is the defect `metrics-truth` closed.

## PLAN
contract: A new `yowo/hardware/_cgroup.py` publishes three pure readers — `read_cpu_quota(root)`,
  `read_memory_limit_bytes(root)`, `read_memory_usage_bytes(root)` — each returning `None` for
  "no limit here", each trying cgroup v2 then v1, each swallowing every OSError and parse failure.
  `effective_cpu_count(root)` composes the quota with affinity and is exported from
  `yowo.hardware`. `detect_system_memory_mb()` gains a cgroup cap over its existing host read, its
  `(0, 0)`-on-failure contract intact. `BaseEngine._memory_pressure()` becomes the single place
  that answers "what fraction of the budget is in use" — CUDA via torch, otherwise via the cgroup,
  `None` when neither is knowable — and both `_oom_monitor_loop` and `health_report` read it, so
  the ladder and the health surface can never disagree. `_start_oom_monitor` starts whenever
  `_memory_pressure()` returns a number, and not otherwise.

## EDGES
- E1 cgroup v2 `cpu.max` = `max 100000` on a 6-CPU host -> 6, never 1.
- E2 cgroup v1 `cpu.cfs_quota_us` = `-1` -> no limit, fall through to affinity.
- E3 cgroup v1 `memory.limit_in_bytes` = `9223372036854771712` -> unlimited, not an 8 EB budget.
- E4 a fractional quota, `cpu.max` = `150000 100000` (`--cpus=1.5`) -> 2 by ceiling, never 0 and
  never a float handed to `range()`.
- E5 a cgroup root that does not exist at all (macOS, bare metal) -> host reading, no exception.
- E6 malformed `cpu.max` — empty, one field, non-numeric, a zero period -> treated as no limit.
- E7 `memory.current` above `memory.max` -> available clamps at 0, never negative.
- E8 a cpuset narrower than the quota -> the narrower of the two wins.

## CHECKS
- test_effective_cpu_count_reads_the_quota_not_the_host · covers: M1, A3 · a v2 fixture granting 2.0
  on a 6-CPU host yields 2.
- test_no_sizing_site_derives_a_budget_from_the_host[_pytorch] · covers: M1, R:HOSTCOUNT · no host
  probe derives a budget in `backends/_pytorch.py`.
- test_no_sizing_site_derives_a_budget_from_the_host[_onnx] · covers: M1, R:HOSTCOUNT · nor in
  `backends/_onnx.py`.
- test_no_sizing_site_derives_a_budget_from_the_host[_tensorrt] · covers: M1, R:HOSTCOUNT · nor in
  `backends/_tensorrt.py`.
- test_no_sizing_site_derives_a_budget_from_the_host[_profile] · covers: M1, R:HOSTCOUNT · nor in
  `tune/_profile.py`, the cache key. One line per case: a parametrised check cited by its bare name
  binds nothing, which is lesson M12 and the reason this node's first gate attempt was refused.
- test_every_sizing_site_calls_the_effective_count[_pytorch] · covers: M1 · and it NAMES the
  cgroup-aware helper, so deleting the host probe cannot be satisfied by deleting sizing entirely.
- test_every_sizing_site_calls_the_effective_count[_onnx] · covers: M1 · likewise.
- test_every_sizing_site_calls_the_effective_count[_tensorrt] · covers: M1 · likewise.
- test_every_sizing_site_calls_the_effective_count[_profile] · covers: M1 · likewise.
- test_the_onnx_session_is_built_from_the_cgroup_budget · covers: M1 · the real
  `_build_session_options`, driven under a fixture cgroup, yields the container's thread counts.
- test_the_pytorch_load_path_configures_threads · covers: M1 · `load()` calls the helper. Added
  during build: the previous node's mutation sweep proved a helper can pass every check in
  isolation while nothing on the real path calls it.
- test_memory_limit_reads_the_cgroup_not_the_host · covers: M2 · 512 MB granted reads as 512 MB.
- test_memory_usage_reads_back · covers: M2 · current usage is the numerator the ladder needs.
- test_detect_system_memory_mb_takes_a_root · covers: M3, A1 · the caller names the cgroup here too.
- test_the_reader_root_is_a_parameter_not_a_constant · covers: A1 · the probe A1 declared.
- test_the_default_root_is_the_real_one · covers: A1 · and it still defaults to /sys/fs/cgroup.
- test_a_quota_that_is_not_a_quota_reads_as_no_limit · covers: E1, E6, R:RAISES · every malformed
  form of `cpu.max` reads as no limit, parametrised one case per id.
- test_cgroup_v1_quota_of_minus_one_is_no_limit · covers: E2 · the v1 sentinel.
- test_v2_wins_when_both_layouts_are_present · covers: A6 · a hybrid host reads as v2.
- test_no_limit_never_sizes_the_process_down · covers: R:STARVE · absent, `max` and `-1` all leave
  the host reading alone.
- test_the_pressure_is_usage_over_the_cgroup_limit · covers: M4 · the fraction is the container's.
- test_an_unlimited_cgroup_is_the_same_as_no_cgroup · covers: R:PHANTOMLADDER · `max` is not a budget.
- test_the_box_does_not_claim_a_ladder_on_an_unlimited_host · covers: M5, R:PHANTOMLADDER · the
  box's claim matches the shipped gate.
- test_the_pytorch_backend_sizes_threads_from_the_cgroup · covers: M1 · loading through the real
  backend path under a patched effective count calls `torch.set_num_threads` with the cgroup's
  number — the wiring, not the helper.
- test_the_tune_fingerprint_changes_with_the_cgroup_quota · covers: M1 · two quotas on one host
  produce two fingerprints, so a tuned profile cannot be reused across unequal containers.
- test_memory_total_is_capped_by_the_cgroup_limit · covers: M2, A4 · a 512 MB limit against a
  12017 MB host reports 512.
- test_memory_available_is_limit_minus_current · covers: M2 · usage subtracts from the limit, not
  from the host's MemAvailable.
- test_no_limit_returns_the_host_reading_unchanged · covers: M2, R:STARVE, E1, E2 · absent, `max`
  and `-1` all leave both numbers exactly as the host reported them.
- test_v1_and_v2_fixtures_agree · covers: M3, A6 · the same limit expressed in either layout yields
  identical numbers.
- test_readers_take_a_root_and_never_raise · covers: M3, R:RAISES, A1, E5, E6 · a missing root, an
  unreadable file, and every malformed form return None rather than raising.
- test_the_v1_unlimited_sentinel_is_not_a_budget · covers: A9, E3 · PAGE_COUNTER_MAX reads as no
  limit.
- test_a_fractional_quota_rounds_up_to_a_whole_cpu · covers: E4 · 1.5 yields 2.
- test_a_narrower_cpuset_still_caps · covers: E8, A8 · affinity below the quota wins, and no quota
  at all falls back to the host count.
- test_usage_above_the_limit_clamps_at_zero · covers: E7 · available never goes negative.
- test_a_non_cuda_engine_under_a_limit_runs_the_ladder · covers: M4, A2, A5 · a CPU engine with a
  cgroup limit at 85% halves its batch through the real `_apply_oom_recovery` path.
- test_a_non_cuda_engine_with_no_limit_starts_no_monitor · covers: R:PHANTOMLADDER, A7 · no thread,
  no degradation, no invented denominator.
- test_health_report_memory_pct_is_the_cgroup_fraction · covers: M4 · a container's health surface
  reports the measured fraction where it used to report None.
- test_an_unreadable_usage_mid_run_does_not_degrade · covers: A10 · a failed tick is skipped.
- test_one_tier_fires_per_poll · covers: A12 · a reading above every threshold takes one action.
- test_the_degradation_log_names_the_budget_it_read · covers: A14 · the operator can tell the cgroup
  ladder from the CUDA one.
- test_effective_cpu_count_is_exported_as_a_plain_int · covers: A13 · the contract `os.cpu_count()`
  had is preserved for callers.
- test_box_9_records_the_measurement_and_the_invalidation · covers: M5 · the milestone box carries
  the measured numbers and states the tune-cache invalidation.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
