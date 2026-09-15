---
type: Task
title: A licensed evaluation dataset, sourced and reachable from CI
status: direction
depth: standard
sensitivity: security
milestone: m3-prove-it
scope:
  - tests/
  - docs/
gives:
  - S1 tests.support.datasets.fetch_verified(archive, cache_dir, *, session, sleep, clock) -> Path — a digest-verified archive, or nothing
  - S2 tests.support.datasets.extract_zip_safely(archive, dest, *, members_under, max_total_bytes) -> Path — traversal-refusing unpack
  - S3 tests.support.datasets.ensure_coco_val2017(cache_dir, *, session, keep_archives) -> Path — the assembled dataset root, complete or absent
  - S4 tests.support.datasets.pinned_subset_ids() -> tuple[int, ...] — the 500 pinned ground-truth image ids
  - S5 tests/fixtures/coco_val2017_subset500.tsv — the in-repo subset manifest, checkable without running anything
  - S6 coco_val2017_root — the integration fixture that fails rather than skips
  - S7 sample_image_path — the existing bus.jpg fixture, now digest-verified
  - S8 docs/datasets.md — provenance, licence and the local fetch path
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "process:run", at: 2026-09-15, act: run, authority: process, outcome: PASS, receipt: /tasks/accuracy-dataset.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-15, act: interview, authority: human, interview: "sha256:8c80c8a2d60946b7", receipt: /tasks/accuracy-dataset.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|A19=confirm|A21=confirm|A20=confirm|R:UNPINNED=confirm|R:TRAVERSAL=confirm|R:HALFCACHED=confirm|R:GREENSKIP=confirm|R:RESELECT=confirm" }
  - { by: "Tin Dang", at: 2026-09-15, act: freeze, authority: human, direction: "sha256:cea1c98c8b34fceb", binding: "sha256:ce8cafb5b72a5550" }
advised_by: security-reviewer
---
## CARD
goal: Make COCO val2017 fetchable, digest-verified, traversal-safe and reachable from CI, with a 500-image subset pinned in the repo, so the repaired mAP evaluator has something real to measure.
why: The project publishes accuracy claims it cannot measure — PR #49 fixed `evaluate_coco_map` and left it with no data to eat; every downstream accuracy gate in m3 waits on this node.
beat: verify · PROCESS DEVIATION RECORDED 2026-09-15: the build ran BEFORE the freeze was stamped. The executing agent authored direction, recorded 49 red-first checks (0 skipped), then built to green — so red-first held and the direction was not shaped to fit the code — but `add freeze` was never marked, because the agent's own boundary treats the freeze as a human seam it must not touch and that boundary outranked the instruction delegating it. The orchestrator's spawn prompt was wrong to delegate a human seam. Direction was reviewed and frozen retroactively at human authority; the receipt predates the stamp and that is visible here rather than tidied away.

## RULES
<must>
- M1 Every archive byte used is compared against a SHA-256 pinned in this repo — on first fetch AND on every cache hit — before anything reads, unpacks or hands it to a test.
- M2 Every fetch is bounded: a connect timeout, a read timeout between chunks, a per-attempt wall-clock deadline that fits inside its CI job, and at most 3 attempts with exponential backoff.
- M3 A fetch publishes atomically: bytes land in a temp file unique to this process and thread, are verified there, and only then are renamed onto the cache path.
- M4 Extraction refuses any archive member that would write outside the destination root, and validates every member before writing any of them.
- M5 The dataset is complete or absent: `ensure_coco_val2017` returns a root only once a completion marker naming its full invalidation set is present and re-derived; an unmarked, half-extracted or partly-deleted tree is rebuilt, never used.
- M6 The pinned subset is exactly `sorted(ground_truth_image_ids)[:500]` — the selection `benchmark/_evaluator.py:199-202` already makes — is recorded in the repo as a readable manifest, and is identical across independent runs.
- M7 An unobtainable dataset FAILS under CI with a message naming each archive, its URL, its pinned digest and a copy-pasteable fetch command; under CI it never reports a skip.
- M8 `sample_image_path` verifies `bus.jpg` against a pinned digest on fetch and on cache hit, exactly as the weights are verified through `resolve_weights`.
- M9 The documentation states where the data comes from, the licence terms it carries, that it is not redistributable as a project asset, how to fetch it locally, and what a contributor without it will see.
- M10 Every pinned URL is HTTPS and TLS verification is never disabled on a fetch.
</must>
<reject>
- R:UNPINNED bytes are unpacked, read, or handed to a test without a digest comparison against a repo-pinned value -> "UNPINNED"
- R:TRAVERSAL an archive member escapes the destination root — by absolute path, by `..`, or by being a symlink -> "TRAVERSAL"
- R:HALFCACHED an interrupted fetch or extraction leaves an artifact that a later run accepts as complete -> "HALFCACHED"
- R:GREENSKIP a missing dataset under CI is reported as a skip or a pass -> "GREENSKIP"
- R:RESELECT the evaluated subset is derived from predictions, from directory listing order, or from anything but sorted ground-truth image ids -> "RESELECT"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1,S2,S3,S6 · the request does not say whether the fetch runs under a service identity or needs a credential; taking it as an anonymous public HTTPS GET, no auth header, no credential ever placed in the URL -> if COCO later gates the bucket, CI fails closed with a transport error instead of leaking a token, and the pins must be re-sourced by hand. · found: the bucket answers an unauthenticated GET today (evidence: `curl -sSI https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip` -> `HTTP/1.1 200 OK`, `Server: AmazonS3`, `Content-Length: 815585330`).
- A2 [who] covers: S4,S5,S7,S8 · the request does not say who may re-pin a digest or regenerate the subset manifest; taking a re-pin to be a deliberate human commit, never an automatic refresh on mismatch -> if wrong, an auto-repin makes every integrity check tautological and R:UNPINNED unfalsifiable. NOTE this diverges DELIBERATELY from `resolve_weights`, which warns and re-fetches once on a cache-hit mismatch under a recorded human decision (`_weights.py`, 2026-09-08): a weight URL is immutable and versioned, so a refetch there resolves the mismatch, whereas these archive URLs are stable and a refetch would return the same bytes and loop.
- A3 [which] covers: S1,S3 · the request does not say WHICH annotations archive carries the val labels; taking `annotations_trainval2017.zip` (241 MB, train+val) and extracting only `annotations/instances_val2017.json` from it -> if wrong we transfer 241 MB to use 20 MB. · found: no val-only annotations object is published — `annotations_val2017.zip` and a bare `instances_val2017.json` both return 404 from the bucket, `annotations_trainval2017.zip` returns 200 (evidence: `curl -sS -o /dev/null -w '%{http_code}' -I <url>` over all three, 2026-09-15).
- A4 [which] covers: S2,S6 · the request does not say which archive members are needed; taking every member under `val2017/` and only `annotations/instances_val2017.json` -> if wrong, a later task wanting captions or keypoints refetches 241 MB, because the archives are deleted after extraction (see PLAN).
- A5 [which] covers: S4,S5 · the request does not say which 500 of the 5000; taking `sorted(getImgIds())[:500]`, byte-for-byte the selection `benchmark/_evaluator.py:199-202` makes, so the fixture and the evaluator cannot disagree -> if wrong, the manifest documents a set the evaluator does not actually score. · probe: the in-repo manifest equals the first 500 sorted image ids of the real `instances_val2017.json`, and every file it names exists under `val2017/`.
- A6 [which] covers: S7,S8 · the request does not say which sample image or which document; taking the existing `bus.jpg` at the bytes it serves today, and one new `docs/datasets.md` rather than a section bolted onto the user guide -> if wrong, a re-hosted `bus.jpg` turns the integration tier red until a human re-pins it, which is the intended trade (and the mismatch message says which of the two causes it is, so a CDN re-encode does not read as an incident).
- A7 [when] covers: S1,S2,S3 · the request does not say how long a fetch may take before it is a failure; taking 30 s to connect, 60 s between chunks, and a 1800 s wall-clock deadline PER ATTEMPT (a retry restarts from byte 0 — there is no Range resume, so the worst case is 3x that), 3 attempts, two backoff gaps of 2 s and 4 s, overridable via `YOWO_DATASET_FETCH_DEADLINE_S` -> if wrong, a link slower than ~600 KB/s sustained is failed when it would have finished, and the override is the escape hatch. The deadline MUST stay strictly below the CI job timeout: above it, GitHub cancels the job first and the contributor gets a bare cancellation carrying none of M7's guidance. `CI_JOB_TIMEOUT_MINUTES_REQUIRED = 45` is the number handed to whoever writes the workflow, and a check binds the two together.
- A8 [when] covers: S4,S5,S6 · the request does not say whether the subset boundary is inclusive; taking `[:500]` — 500 images, id 56545 included, id 57027 excluded -> if wrong, every published mAP is measured over a different denominator than the one documented, and the number stops being comparable to itself across releases.
- A9 [when] covers: S7,S8 · the request does not say when a pin goes stale; taking a pin as valid until a check fails, and a failing pin as a human re-pin decision recorded in `docs/datasets.md`, never a silent refresh -> if wrong, upstream republishing quietly changes what "the same test" measures.
- A10 [absent] covers: S3,S6 · the request does not say what a missing dataset means; taking absent-under-CI as a FAILURE naming both archives and the fetch commands, and absent-locally as a skip, matching the existing `_unavailable()` policy -> if wrong, either CI proves nothing (Q3) or every contributor without 1.07 GB of disk hits a wall.
- A11 [absent] covers: S1,S2 · the request does not say what a half-written temp file or half-extracted tree means; taking any artifact without a completion marker naming the full invalidation set as ABSENT and rebuilding it -> if wrong, a killed fetch is cached as success forever, which is the `if dest.exists()` defect this persona exists to stop.
- A12 [absent] covers: S4,S5 · the request does not say what an absent, empty or unreadable manifest means; taking it as a hard error, never an empty subset -> if wrong, an empty subset evaluates zero images and zero images score a vacuous pass, which is the exact failure this milestone exists to stop.
- A13 [absent] covers: S7,S8 · the request does not say what an absent `bus.jpg` pin means; taking the pin as mandatory with no unpinned path, unlike `resolve_weights`'s `sha256 is None` warn-and-continue -> we lose an escape hatch, but that hatch exists for user-registered models whose digest cannot be known, and this URL is ours to pin. The residual risk is real and named: unlike the weights' immutable versioned release URL, this is an unversioned path on a marketing site behind a CDN, so a re-encode we cannot predict turns the tier red — which is why the mismatch message distinguishes "upstream republished, re-pin" from "corrupted or substituted" rather than escalating both.
- A14 [order] covers: S1,S2,S3 · the request does not say the order of verify versus unpack; taking verify-then-rename-then-extract, never extract-then-verify -> if wrong, unverified bytes are written across the filesystem before they are rejected, which is where the traversal surface actually bites. This ordering, not the traversal rules, is the control the security argument rests on. · probe: extraction is never reached for bytes that failed their digest.
- A15 [order] covers: S4,S5 · the request does not say what orders the manifest; taking ascending numeric image id — the order `sorted()` itself produces — one `id<TAB>file_name` line each -> if wrong, a reader checking the manifest by eye cannot reproduce the selection, which is the whole reason it is in the repo.
- A16 [order] covers: S1,S3,S6 · the request does not say what happens when two runs race for one cache (pytest-xdist, or two CI jobs sharing a restore); taking a temp name carrying both pid and thread id plus `os.replace`, so the loser's bytes are discarded and no reader ever observes a partial -> if wrong, two workers interleave into one file and both verify it.
- A17 [experience] covers: S3,S6,S7 · the request does not say who receives a failure; taking the recipient as a contributor mid-`pytest` who has never heard of COCO, so the message carries both archive names, both URLs, both pinned digests, the cache location and the deadline override -> if wrong they get `FileNotFoundError: val2017` and file an issue instead of fetching 1.07 GB.
- A18 [experience] covers: S8 · the request does not say where the document lives; taking `docs/datasets.md`, linked from CONTRIBUTING's testing section, because that is the page a contributor already opens before running tests -> if wrong, the licence note exists and nobody ever meets it.
- A19 [experience] covers: S4,S5 · the request does not say how a reader checks WHICH 500 without running anything; taking a plain-text `id<TAB>file_name` manifest committed to the repo, with its own digest pinned beside it, readable in a browser diff -> if wrong, "pinned" degrades to "trust the code", and a silent re-selection is invisible in review.
- A21 [order] covers: S7,S8 · the request does not say what order bus.jpg's verification takes relative to its publication, nor what a reader of the doc must meet first; taking the same verify-then-rename order the archives use for the image, and putting the licence and non-redistribution terms into CONTRIBUTING at the point it links the doc rather than only at the foot of the page -> if wrong, a contributor fetches 1.07 GB of Flickr-licensed imagery before meeting the terms that govern it.
- A20 [experience] covers: S1,S2 · the request does not say what a failure mid-transfer or mid-unpack must tell its reader; taking it that a refused member names WHICH member and why, and that an exhausted fetch names the URL, the attempt count and the last transport error -> if wrong, the two loudest failure paths in this node report "DatasetIntegrityError" with no way to tell a hostile archive from a flaky link.

## PLAN
contract: `tests/support/datasets.py` publishes a frozen `RemoteArchive(name, url, sha256, size_bytes)`
and the functions over it. `fetch_verified` streams to a temp sibling unique to this process and
thread under connect/read/wall-clock bounds with 3 attempts and two backoff gaps, hashes the temp
file, compares it to the pin, and only then `os.replace`s it onto the cache path. A mismatch raises
`DatasetIntegrityError`, removes the bytes, and is never retried. On a cache hit it re-verifies rather
than trusting the path — and distinguishes a file of unknown provenance (replaced once, as
`resolve_weights` does) from one this process verified and which has since changed (refused outright).
`extract_zip_safely` validates every member against the RESOLVED root before writing any of them —
prefix matching is not containment, since `"val2017/../../x".startswith("val2017/")` is true.
`ensure_coco_val2017` composes them, then writes a completion marker LAST.
`pinned_subset_entries` reads `tests/fixtures/coco_val2017_subset500.tsv`, verifies it against its own
pin, and refuses absent/empty rather than returning nothing. `session`, `sleep` and `clock` are part of
S1's contract, not leaked scaffolding: they are what make the retry and deadline behaviour assertable
without sleeping through it.

error taxonomy: `DatasetIntegrityError` (wrong bytes, never retried) and `DatasetUnavailableError`
(transport, deadline, absence — retried) both derive from `DatasetError`. Collapsing the two is how a
digest mismatch gets retried into success.

what is kept: the archives are DELETED once extracted. Their digests survive in the completion marker
as source provenance. What is re-verified on every cache hit is the tree we still have — the
annotations file that decides which images are scored, plus the image count. That is not weaker than
re-hashing 1.07 GB of zip; it is the only check that can see corruption occurring AFTER extraction,
and it is the only thing that detects a partially-deleted tree. It also keeps the per-run cost low
enough that nobody switches it off. Hard constraint behind the choice: `~/.cache/yowo/weights` and
`~/.cache/yowo/test-assets` are already cached by three CI jobs against GitHub's 10 GB per-repo
budget, and 1.07 GB of archives beside a 787 MB tree would evict the weight store.

not done, deliberately: all 5000 images are extracted, not only the pinned 500. Extracting 500 would
save ~700 MB but would make `load_coco_dataset(root, subset=N)` fail for any N > 500 with an opaque
missing-JPEG error, and `evaluate_coco_map`'s selection semantics are explicitly out of scope.
Per-file digests for the retained JPEGs are a real strengthening this node does not do — recorded as
residue, not silently dropped.

scope: `tests/`, `docs/`. `.github/workflows/ci.yml` is SHARED and was dropped from `scope:` — the CI
job is handed to the orchestrator as YAML in the report, together with a cache-key change the existing
workflow needs (see EVIDENCE).

regression floor: the four quality-gate commands stay clean and no existing test changes its verdict.
`evaluate_coco_map`'s selection semantics are not touched.

## EDGES
- E1 A digest mismatch on an ALREADY-CACHED archive — the cache-hit path, which runs every subsequent time and which a download-time-only check never protects.
- E2 A stream that stalls mid-body: the connect timeout has already passed, bytes arrived, and then it trickles.
- E3 A zip member that is a symlink. CPython's `zipfile` writes such a member as a regular file, so this is not a live escape THERE — but `docs/datasets.md` documents a manual fetch using `unzip`, which does honour symlinks, and S2 is a published surface other nodes may point at unpinned archives.
- E4 Two processes fetching the same archive into one cache directory at once.

## CHECKS
- test_digest_mismatch_refuses_the_archive · covers: M1,R:UNPINNED · bytes that hash to anything but the pin raise, and nothing is left behind
- test_a_mismatch_is_never_retried_into_success · covers: M1,R:UNPINNED · a digest failure aborts on the first attempt
- test_cache_hit_is_verified_not_trusted · covers: M1,E1 · a cached file corrupted after it was written is refused, decided locally with zero further requests
- test_reuse_is_decided_by_digest_not_by_path_existence · covers: M3,E1 · a non-archive file already at the destination is replaced, not handed back; once correct it is reused with zero requests
- test_extraction_never_runs_on_unverified_bytes · covers: M1,A14,R:UNPINNED · extraction is never reached for bytes that failed their digest
- test_fetch_bounds_connect_and_read · covers: M2 · every request carries a (connect, read) timeout pair, not None
- test_a_stalled_stream_is_abandoned_at_the_deadline · covers: M2,E2 · a body still trickling past the deadline is abandoned, driven by an injected clock rather than real sleeping
- test_transport_failure_retries_with_backoff_then_gives_up · covers: M2 · exactly 3 attempts, sleeps of 2 s and 4 s, then one error naming the URL
- test_backoff_has_one_gap_fewer_than_it_has_attempts · covers: M2 · three attempts have two gaps; a third value would be a sleep never performed
- test_the_deadline_fits_inside_its_declared_job_timeout · covers: M2,M7 · a deadline above its job timeout can never fire
- test_the_deadline_is_overridable · covers: M2 · a slow link raises it rather than giving up; a non-numeric value is refused
- test_a_wrong_sized_object_aborts_before_the_transfer · covers: M2 · content-length is a pre-flight, never an integrity check
- test_interrupted_fetch_leaves_no_artifact_a_later_run_accepts · covers: M3,R:HALFCACHED · a truncated response leaves neither a cache file nor a temp file
- test_concurrent_fetches_never_publish_a_partial · covers: M3,E4 · a watcher polling the destination during two racing fetches never observes a short file
- test_extraction_refuses_a_member_escaping_the_root[../escaped.txt] · covers: M4,R:TRAVERSAL · a parent-directory member is refused, not silently rewritten
- test_extraction_refuses_a_member_escaping_the_root[../../escaped.txt] · covers: M4,R:TRAVERSAL · a two-level escape is refused
- test_extraction_refuses_a_member_escaping_the_root[val2017/../../escaped.txt] · covers: M4,R:TRAVERSAL · an escape that survives a prefix filter is refused
- test_extraction_refuses_a_member_escaping_the_root[/absolute/escaped.txt] · covers: M4,R:TRAVERSAL · an absolute member is refused
- test_extraction_refuses_a_symlink_member · covers: M4,E3 · a member whose mode carries S_IFLNK is refused before any member is written
- test_half_extracted_tree_is_rebuilt_not_used · covers: M5,R:HALFCACHED · a tree with images and annotations but no marker reports incomplete
- test_completion_marker_names_both_pinned_digests · covers: M5 · a marker written under a different pin does not satisfy the current one
- test_every_marker_field_invalidates_the_tree[marker_version] · covers: M5 · the marker's own format version is compared
- test_every_marker_field_invalidates_the_tree[images_archive_sha256] · covers: M5 · the images pin is compared
- test_every_marker_field_invalidates_the_tree[annotations_archive_sha256] · covers: M5 · the annotations pin is compared
- test_every_marker_field_invalidates_the_tree[subset_manifest_sha256] · covers: M5 · re-pinning the subset invalidates the tree
- test_every_marker_field_invalidates_the_tree[subset_size] · covers: M5 · changing how many are scored invalidates the tree
- test_every_marker_field_invalidates_the_tree[annotations_json_sha256] · covers: M5 · the kept annotations digest is compared
- test_every_marker_field_invalidates_the_tree[image_count] · covers: M5 · the kept image count is compared
- test_a_complete_tree_missing_its_annotations_is_not_complete · covers: M5,R:HALFCACHED · deleting the annotations reopens the tree for rebuild
- test_a_partially_deleted_image_tree_is_not_complete · covers: M5,R:HALFCACHED · the count is the only witness once the archives are gone
- test_corrupted_annotations_are_caught_on_the_cache_hit_path · covers: M1,M5,E1 · post-extraction corruption of the file that decides the subset is caught
- test_pinned_subset_is_the_sorted_ground_truth_prefix · covers: M6,R:RESELECT · selection over a reversed id list returns ascending order and lands on the documented boundary (139 first, 56545 last, 57027 out)
- test_pinned_subset_is_identical_across_independent_runs · covers: M6 · two separate interpreters, each given the ids shuffled differently, agree byte for byte
- test_manifest_matches_its_pinned_digest · covers: M6,R:RESELECT · the committed manifest hashes to its pin; an edited one is refused
- test_a_missing_manifest_raises_rather_than_returning_nothing · covers: M6,A12 · absent is a hard error, never an empty subset
- test_an_empty_manifest_raises_rather_than_returning_nothing · covers: M6,A12 · empty is a hard error, never a vacuous pass
- test_missing_dataset_fails_under_ci_and_never_skips · covers: M7,R:GREENSKIP · under CI the outcome is DatasetUnavailableError and specifically not pytest's Skipped
- test_missing_dataset_skips_locally · covers: M7 · without CI a contributor gets a skip rather than a wall
- test_failure_message_names_the_archive_and_how_to_get_it · covers: M7 · both names, both URLs, both pins and the cache location
- test_sample_image_is_digest_verified_on_fetch · covers: M8 · wrong bytes from the image URL are refused and nothing is cached
- test_sample_image_cache_hit_is_verified · covers: M8,R:UNPINNED · a corrupted cached bus.jpg is refused
- test_the_integration_fixture_pins_the_sample_image · covers: M8,R:UNPINNED · the conftest routes bus.jpg through the verified fetch, not a bare requests.get
- test_docs_record_provenance_licence_and_fetch_path · covers: M9 · both URLs, both pins, CC BY 4.0, the Flickr terms, the non-redistribution note, the subset size
- test_contributing_links_the_dataset_doc · covers: M9 · a contributor meets the licence before running tests
- test_tls_verification_is_never_disabled · covers: M10 · no fetch passes verify=False
- test_every_pinned_url_is_https · covers: M10 · no pinned URL is plain HTTP
- test_pinned_subset_matches_the_real_annotations · covers: A5,M6,R:RESELECT · integration: the manifest equals sorted(getImgIds())[:500] of the real instances_val2017.json
- test_the_selection_agrees_with_the_evaluator_itself · covers: M6,R:RESELECT · integration: pycocotools' own getImgIds is asked, not a re-implementation
- test_every_pinned_image_is_present_on_disk · covers: M5,M6 · integration: every pinned file exists in the tree
- test_the_file_names_match_the_ids_they_are_pinned_against · covers: M6 · integration: both halves of a row agree with the annotations
- test_load_coco_dataset_reads_the_tree_this_fixture_builds · covers: M5 · integration: the layout produced is the layout the evaluator expects
red-first: every check MUST fail first.

Deliberately NOT gated (true before the build, kept as guidance): `test_a_wholesome_archive_still_extracts`
(stops "refuse everything" satisfying the traversal checks — and it earned its place, catching a real
over-refusal where `writestr` members carry permission bits with no file-type bits),
`test_manifest_rows_are_ascending_and_unique` and `test_the_bus_image_pin_is_a_real_sha256` (shape
guards on data this node commits). Binding a Must to a check that never went red is M12/Q10 in another
form.

## EVIDENCE
receipt: runs/1.md — 54/54 reported, exit 0, freshness content, 2026-09-15
gate: <PASS | RISK-ACCEPTED | HARD-STOP>  — NOT this agent's to mark

### Residue (three lenses)
- SECURITY — no HARD-STOP. No credential, token or authorization boundary is touched. The fetch is
  anonymous HTTPS with verification on; bytes are digest-checked before extraction; extraction
  validates every member against the resolved root and refuses absolute, `..`, symlink and
  non-regular members before writing any. MEASURED: CPython's `zipfile` already sanitises names and
  never creates a symlink, so this closes no exploitable hole today — it refuses instead of silently
  rewriting, and it holds if the extractor is ever swapped (the documented manual path uses `unzip`,
  which DOES honour symlinks).
  RULED 2026-09-15: sensitivity RAISED `data` -> `security`. A 1.07 GB fetch from a third party plus
  an archive unpack executing inside CI is a supply-chain trust edge, and the CLAUDE.md floor makes
  security a HARD-STOP lens rather than an advisory one. Raised before any interview answer existed,
  so nothing was cleared by the edit.
  REVIEWED 2026-09-15 by the orchestrator, reading `_check_member`, `extract_zip_safely` and
  `fetch_verified` directly rather than accepting the report. Verdict: no HARD-STOP. Four controls
  confirmed by reading, not by summary — containment decided by `Path.resolve()` and not by a string
  prefix (the code carries the counter-example itself: `"val2017/../../x".startswith("val2017/")` is
  True); symlink detection via `stat.S_IFMT(external_attr >> 16)` with the correct note that a DOS-era
  zip has no type bits at all, so "has any mode" would have been the wrong question; every member
  validated BEFORE any member is written; and digest-verify-then-`os.replace`, so a reader that finds
  the cache path finds bytes that already passed.
  ONE RESIDUAL the report did not name: the zip-bomb bound sums `info.file_size`, the DECLARED size
  from the central directory, which an adversarial archive can understate. Harmless for COCO, whose
  content is fixed by the digest pin — but `extract_zip_safely` is a published surface whose own
  docstring invites other nodes to point it at archives carrying no pin, and for those the bound is
  advisory. Named here rather than fixed: bounding the DECOMPRESSED stream belongs to whichever node
  first extracts an unpinned archive.
- CONCURRENCY — temp files carry pid AND thread id; publication is `os.replace`, atomic on POSIX, so
  two processes racing cannot interleave. The in-process `_verified` map is lock-guarded. A watcher
  polling the destination during two racing fetches observes no short file. Residual: the process-race
  case is argued from `os.replace` atomicity plus the unique temp name, not directly executed.
- ARCHITECTURE — one new package, `tests/support/`, imported as `tests.support.datasets` (the
  `tests.utils` convention, not the fragile `pythonpath` one). The committed manifest went to the
  existing `tests/fixtures/`, not a second data directory. `evaluate_coco_map`'s selection semantics
  are untouched; the fixture mirrors them rather than reimplementing them.

### Residue NOT done, named rather than dropped
- Per-file SHA-256 for the retained 5000 JPEGs. The tree is attested by count plus the annotations
  digest; a corrupted individual JPEG would not be caught. Extending the manifest to
  `id<TAB>file_name<TAB>sha256` would close it for the pinned 500.
- The process-race is not executed as a test, only argued.

### The pins, and how they were obtained
Computed from a real download on 2026-09-15 — not copied from anywhere — then corroborated three
independent ways. COCO publishes no checksums of its own.

| object | bytes | sha256 |
|---|---:|---|
| `val2017.zip` | 815585330 | `4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05` |
| `annotations_trainval2017.zip` | 252907541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `bus.jpg` | 137419 | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| `coco_val2017_subset500.tsv` | 11390 | `872ac411cf65f03234a725df9711e9ced36d1c8156161ab0b7ae63bd23336aad` |
| `instances_val2017.json` (extracted) | 19987840 | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |

1. size == the `Content-Length` the bucket reports;
2. every zip member's CRC verifies (`unzip -t`, both archives clean);
3. both reproduce S3's own ETag — annotations is single-part, so its ETag is a plain MD5 and matches
   `f4bbac642086de4f52a3fdda2de5fa2c` directly; images is a 98-part multipart upload and recomputing
   the multipart ETag over 8 MiB parts reproduces `d366be60d3dc737327160d62453e3973-98`.
`bus.jpg` was fetched twice fresh and agreed with a pre-existing cached copy from an earlier session.

### TLS finding — do not let this be "tidied"
`https://images.cocodataset.org/...` FAILS certificate verification: the bucket is fronted by a cert
issued for `s3.amazonaws.com`. That is why most recipes silently downgrade to plain HTTP. The
path-style `https://s3.amazonaws.com/images.cocodataset.org/...` reaches byte-identical objects
(matching ETags) over TLS that validates, and is what is pinned.

### Change request the orchestrator must apply (`.github/workflows/ci.yml`, not this node's file)
The existing cache key `yowo-weights-sha256-${{ hashFiles('src/yowo/models/_registry.py') }}` covers
`~/.cache/yowo/test-assets`, where `bus.jpg` lives — but its pin now lives in
`tests/support/datasets.py`. A re-pin would therefore NOT change the key, stale bytes would be
restored, verification would fail, and CI would stay red on every run until the cache evicted, with no
contributor-reachable way to clear it. The key must become
`hashFiles('src/yowo/models/_registry.py', 'tests/support/datasets.py')` in BOTH the `weights` and
`backend-smoke` jobs. The new accuracy job needs `timeout-minutes: 45` to sit above the 1800 s
per-attempt fetch deadline.

## LESSONS
- A check that asserts "did not raise Skipped" by catching `BaseException` passes on an AttributeError from a function that does not exist yet — a green report for a check that never ran the policy. Name the expected exception type. -> add learn test
- `pytest.skip` called inside code under test SKIPS the test asserting it — Q3 one layer down, where the skip is invisible because it is reported against the very check written to forbid it. Capture the outcome, do not let it propagate. -> add learn test
- CPython's `zipfile.extractall` already strips `..`, leading separators and drive letters, and writes a symlink member as a regular file. A traversal check asserting "nothing escaped" is green before any fix; the property worth binding is that a hostile member is REFUSED rather than silently rewritten. -> add learn persona:artifact-integrity-steward
- An internal timeout above its CI job timeout can never fire: the platform kills the job first and every carefully-written failure message is replaced by a bare cancellation. Bound the inner deadline to the outer one with a check. -> add learn build
