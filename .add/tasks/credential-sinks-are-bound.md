---
type: Task
title: Every sink named in box 6 is bound by a check that fails when the credential reaches it
status: done
depth: deep
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/io/
  - src/yowo/pipeline/
  - src/yowo/cache/
  - tests/unit/
gives:
  - S1 the four sinks m1 box 6 names: a log, an exception message, a result payload, a cache key
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:a41e2918ad084905", receipt: /tasks/credential-sinks-are-bound.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A7=confirm|A6=confirm|R:UPSTREAM=confirm|R:SOURCESCAN=confirm|R:TICKBOX=confirm|R:SELFANSWER=confirm" }
  - { by: "cli", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:de97963a693df659", binding: "sha256:6053c5d4353b5da1" }
  - { by: "cli", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:0c2e33cd49affebc", binding: "sha256:6053c5d4353b5da1" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/credential-sinks-are-bound.d/runs/1.md }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:d5792eeff7fe57e1" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/credential-sinks-are-bound.d/runs/2.md }
  - { by: "cli", at: 2026-09-10, act: refreeze, authority: human, direction: "sha256:baccf3fa3270375c", binding: "sha256:6053c5d4353b5da1" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:7a477ed502d04bd3" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/credential-sinks-are-bound.d/runs/3.md }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/credential-sinks-are-bound.d/runs/4.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/credential-sinks-are-bound.d/runs/4.md, brief: "sha256:7a477ed502d04bd3", reason: "19 checks green, zero skipped, every rule bound. Re-verified by me rather than taken on report: suite 2368 passed / 11 skipped, ruff clean, pyright 0 errors, tests/unit/test_rtsp_redaction.py byte-identical to its gating commit (sha256 8a8f4f8f..048404e7, zero-line diff) as M7/A7/E8 require. THE BUILD FOUND A LEAK THE NODE HAD NOT TRACED, which is the point of building rather than asserting. I traced Frame.source_id; the stream_id a caller passes to FrameCollector.add_stream and DetectionRouter.register is very often the camera URL, credentials and all, and it reached SEVEN log sites, a RuntimeError, dict keys, the bridge thread name and the routing callback RAW. Fixed at the boundary per M5 — pipeline/_ids.py, safe_stream_id() at four public entry points. No sink learned to redact. MUTATION-VERIFIED BY ME, three mutations because the four sinks have three origins: reverting the boundary call in _collector reddens 5; making safe_stream_id return its argument reddens 5; the :346 Frame.source_id mutation reddens 9 including both cache-key checks. All 19 green restored. Each of the four rejects is refuted by an injected offender: source.safe_url, inspect.getsource, a redact_url tautology, and ticking box 6 each go red. I CORRECTED MY OWN FROZEN MUST AND THE ERROR WAS MINE. M2 said every sink check reddens under the single :346 mutation. It cannot: E2 of this same node already named :322 for the exception sink, and the log and callback sinks originate at add_stream. A Must demanding one mutation redden all four sinks is one correct code cannot satisfy, and the build was right to say so rather than contort the checks to fit. M2 and E1/E1b/E1c now name all three mutations. THE BUILD ALSO CAUGHT ITSELF THREE TIMES, disclosed rather than buried: its _refute helper scored its own liveness guards as refutations; its log-sink liveness was met by non-sink records, so deleting :196 reddened only 1 check and it initially called that clean — the exact misread this node exists to prevent, now reddening 5; and its own fix introduced a regression, since redact_url collapses credential-free unparseable authorities to one fixed string and the id is a dict key, so two streams would have cross-delivered detections through DetectionRouter.register's silent overwrite. Fixed at the cause with the @ short-circuit and bound by test_redaction_happens_once_at_the_boundary. THREE RISKS IN EVIDENCE, because the PLAN said no behaviour change was expected and that was wrong: callbacks now receive the redacted id; credential-only-differing cameras collapse to one id, loudly in add_stream and silently in register; and the @ short-circuit is a deliberate security boundary decision, sound because RFC 3986 delimits userinfo with @. BOX 6 STAYS OPEN AND A CHECK NOW ENFORCES THAT. redact_url preserves the query string, and the mixed form is the damning one — I measured rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2 coming back as rtsp://10.0.0.5:554/s?token=SECRET&password=hunter2. The userinfo IS stripped, so the @ is gone and the string wears the visual signature of a redacted URL while carrying the literal password. It defeats eyeball review of a log and defeats a reviewer diffing before against after, because the diff shows redaction happening. Pre-existing — _redact.py last touched at 3d0594e, in neither of the build's commits — and outside this node's frozen scope, which fixes the userinfo form. It takes its own node." }
advised_by: security-reviewer
---
## CARD
goal: A credentialed RTSP URL cannot reach a log, an exception message, a result payload or a cache key without a check going red.
why: Box 6 says "asserted by a check using a credentialed RTSP URL". Ten checks in `test_rtsp_redaction.py` claim to do that. Measured 2026-09-10 by mutation: changing `_source.py:346` from `source_id=self._safe_url` to `source_id=self._url` — putting the raw password into every emitted result AND into the feature-cache dict key — leaves ALL TEN GREEN. Only the exception sink binds; mutating `:322` does go red.
  The reason is that two of the four "sink" checks assert the same thing, and it is not a sink. `test_source_id_carries_no_credential` and `test_cache_key_carries_no_credential` have BYTE-IDENTICAL bodies — both are `_no_credential(source.safe_url)`. They read the redacting attribute, never the payload that is emitted or the key the cache actually stores. A third, `test_reconnect_timeout_message_carries_no_credential`, is method M9 exactly: `assert "{self._url}" not in inspect.getsource(...)`, which proves a token is not spelled one particular way. And the log sink has no check at all.
  The sinks are real and reachable, not hypothetical. `pipeline/__init__.py:196` is `logger.warning("Stream %r failed during pipeline run: %s", sid, exc)` and `:192` interpolates the same id into a `RuntimeError`. `cache/__init__.py:151` is `self._last_fingerprints[source_id] = fp`, followed by `self._store.store(source_id, ...)` — a dict key and a store key, which outlive log scrubbing entirely. `Frame.source_id` reaches every result through `postprocess/_classify.py:67`, `_obb_nms.py:175` and `:227`, and `obb_engine.py:231`.
  The code is CORRECT today. This node changes almost nothing about behaviour; it makes the guarantee enforced instead of coincidental. Box 6 is currently unticked for the right reason and must not be ticked on the checks that exist.
next: bind each sink at the sink, then re-measure the same mutation.

## RULES
<must>
- M1 Each of the four sinks box 6 names is bound by at least one check that reads THE SINK ITSELF — the string a logger emitted, the exception that propagated, the identifier on an emitted result, the key the cache stored — and not an attribute that happens to be redacted upstream of it.
- M2 Each sink check goes RED under the mutation that feeds THAT sink, and the mutations are recorded. CORRECTED post-build, and the correction is mine: I froze this Must as "every one of those checks goes RED when `_source.py:346` is changed", which asserts that one mutation reaches all four sinks. It does not, and E2 of this same node already said so by naming `:322` for the exception sink. Measured: the `:346` mutation reddens 9 of 15 — the result payload and both cache keys — and leaves the log and exception checks green, because those sinks have different origins. The four sinks have THREE origins: `_source.py:346` (`Frame.source_id` -> result payload -> cache key), `_source.py:322` (the exception), and the `stream_id` argument to `FrameCollector.add_stream` / `DetectionRouter.register` (the log record, the `RuntimeError`, the dict keys, the thread name, the callback). A Must that demands one mutation redden all four is a Must that cannot be satisfied by correct code, and the build was right to say so rather than contort the checks to fit it.
- M3 No check in this node proves a guarantee by reading source text. A test that asserts a token is or is not spelled somewhere survives a rewrite that keeps the behaviour and changes the spelling, and survives deleting what it names (method M9).
- M4 A credentialed URL is used END TO END: constructed with a real user and password, driven through the real code path, and the assertion is made on what came out — never on a value the test computed itself.
- M5 The redaction stays at the boundary. The fix for any leak found is that the sink receives an already-safe value, not that each sink learns to redact — one redaction site, so a sink added tomorrow is safe by construction.
- M6 The raw URL remains available to whatever must connect with it. Redacting the value passed to `cv2.VideoCapture` would break every RTSP stream, and a check pins that it is still the credentialed one.
- M7 `tests/unit/test_rtsp_redaction.py` is NOT edited. It belongs to a frozen, gated node; its ten checks are weak but none is wrong, and a node that fixes weak checks by editing another node's checks is one refactor away from fixing them by deleting them. The new checks are added alongside, and the weak ones become harmless redundancy. If they should later go, that is a change-request against `rtsp-credential-redaction`, carrying this node's mutation evidence.
</must>
<reject>
- R:UPSTREAM Binding a sink by asserting a redacting attribute — `source.safe_url` — instead of the sink's own output. That is the defect being fixed and it must not be reintroduced under a new name. -> "UPSTREAM"
- R:SOURCESCAN Any `inspect.getsource` / string-in-source assertion standing as the binding for a rule. -> "SOURCESCAN"
- R:TICKBOX Ticking m1 box 6 on checks that survive the M2 mutation. -> "TICKBOX"
- R:SELFANSWER A check that asserts a credential is absent from a string the test itself redacted. -> "SELFANSWER"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose credential; taking an operator's RTSP camera password, the case box 6 names, which reaches a shared log and a shared result store rather than staying on one machine -> if wrong and only local use matters, the checks are stricter than needed, which costs nothing.
- A2 [which] covers: S1 · box 6 names four sinks and the exception one already binds; taking all four anyway, re-binding the exception sink at the sink rather than trusting the check that exists -> if wrong, one redundant check.
- A3 [when] covers: S1 · the request does not say whether a leak is caught at write time or at read time; taking write time — the assertion is on what the sink received, because a value already in a log file or a cache key cannot be unsent · probe: drive a real failure through `run_pipeline` with a credentialed stream id and read the record the logger actually emitted -> if wrong, a leak is detected only after it has already been persisted.
- A4 [absent] covers: S1 · the request does not say what happens when a sink is not exercised — no pipeline failure, no cache write; taking a check that FAILS rather than skips if it could not reach the sink, since a skipped credential check reads exactly like a passed one in a suite summary -> if wrong, a green suite where a sink was never visited. · probe: assert the sink was actually written to before asserting what it contains.
- A5 [order] covers: S1 · the request does not say what happens when the same credentialed URL reaches two sinks; taking each sink asserted independently, because the byte-identical bodies in `test_rtsp_redaction.py` are how one assertion came to stand for two sinks -> if wrong, some duplication between checks, which is the correct direction to be wrong in here.
- A7 [absent] covers: S1 · the request does not say what to do with the weak checks already there; taking ADD-alongside, never edit — see M7 -> if wrong, ten redundant checks remain in the suite, which costs a second of runtime and no correctness. · probe: `git diff` must show zero changes to `tests/unit/test_rtsp_redaction.py`.
- A6 [experience] covers: S1 · the request does not say who reads a failure; taking a maintainer who must know WHICH sink leaked and what reached it, so each message names the sink and shows the redacted-vs-raw comparison rather than asserting `not in` and printing nothing -> if wrong, a red check that says a credential leaked somewhere.

## PLAN
contract: no behaviour change is expected. A new `tests/unit/test_credential_sinks.py` binds each of the four sinks at the sink: a real `logging` capture around a `run_pipeline` failure carrying a credentialed stream id; the `SourceError` that actually propagates out of `_open_cap`; the `source_id` on a `Frame` emitted by the real read path and carried into a real postprocess result; and the key `FeatureCache` actually stored, read back out of `_last_fingerprints` and the store. `tests/unit/test_rtsp_redaction.py` is not touched at all (M7, A7). Any leak the sink-level checks find is fixed at the boundary, not at the sink.

## EDGES
- E1 The `:346` mutation — `source_id=self._url`. The result-payload and cache-key checks must go red; measured at 9 of 15 red. Today all ten of the parent node's checks survive it, which is why this node exists.
- E1b The `:322` mutation — the exception message. The exception check must go red. Named separately because it is a separate origin (see the M2 correction).
- E1c The boundary mutation — `safe_stream_id` returning its argument unchanged, or the call removed from `add_stream`. The log, `RuntimeError`, dict-key and callback checks must go red; measured at 5 of 15 red in both forms.
- E2 The exception sink — the one that already binds. Its replacement must still go red on the `:322` mutation.
- E3 A pipeline run where nothing fails, so the log sink is never written. The check must fail, not pass and not skip (A4).
- E4 A cache key read back after eviction — the key is a dict key AND a store key, and `cache/__init__.py:151-152` writes both.
- E5 `cv2.VideoCapture` still receives the credentialed URL. If this goes red, the redaction has broken the feature.
- E6 A username with no password, and a malformed URL — both already covered in `test_rtsp_redaction.py`; they must still hold at the sinks, not just at `redact_url`.
- E7 A sink reached with a non-RTSP credentialed URL (an HTTP camera), since `redact_url` is scheme-agnostic and the sinks are not RTSP-specific.
- E8 `tests/unit/test_rtsp_redaction.py` unchanged — asserted, because "the weak check was in the way" is how a weak check becomes no check.

## CHECKS
all in `tests/unit/test_credential_sinks.py`. Every one drives a credentialed URL through
the real code path and asserts on what the SINK received — never on `safe_url`, never on
source text (M1, M3, R:UPSTREAM, R:SOURCESCAN).

- tests.unit.test_credential_sinks::test_the_log_record_the_pipeline_emitted_carries_no_credential · covers: M1, M4, A3 · a real `logging` capture around a `run_pipeline` failure whose stream id is a credentialed URL; the assertion reads `record.getMessage()`. red: nothing asserts the log sink at all today.
- tests.unit.test_credential_sinks::test_the_log_sink_was_actually_written_to · covers: A4, E3 · the probe A4 names — assert a record exists before asserting what is in it, so a run where the sink was never reached FAILS rather than passing quietly. red: no such guard.
- tests.unit.test_credential_sinks::test_the_all_streams_failed_error_carries_no_credential · covers: M1, M4 · `pipeline/__init__.py:192` interpolates the same id into a `RuntimeError` that propagates to the caller. red: unasserted.
- tests.unit.test_credential_sinks::test_the_exception_that_propagates_carries_no_credential · covers: M1, E2, A2 · the `SourceError` caught from the real `_open_cap`, re-bound at the sink rather than trusted to the parent check. red: passes today — this is the one sink that already binds, and it is re-bound so the set is uniform.
- tests.unit.test_credential_sinks::test_the_emitted_frame_identifier_carries_no_credential · covers: M1, M2, M4 · a `Frame` produced by the real read path; the assertion reads `frame.source_id`. red: the M2 mutation leaves every existing check green.
- tests.unit.test_credential_sinks::test_the_identifier_on_a_postprocess_result_carries_no_credential · covers: M1, M2 · the id that survives into a real result object, which is what reaches result JSON. red: no check follows the id past the Frame.
- tests.unit.test_credential_sinks::test_the_key_the_feature_cache_stored_carries_no_credential · covers: M1, M2, E4 · read back out of `_last_fingerprints`, the dict key itself, not the value passed in. red: `test_cache_key_carries_no_credential` reads `safe_url` and never touches the cache.
- tests.unit.test_credential_sinks::test_the_key_the_feature_store_stored_carries_no_credential · covers: M1, E4 · `cache/__init__.py:152` stores under the same id separately; a key in two places is two sinks. red: unasserted.
- tests.unit.test_credential_sinks::test_the_mutation_that_leaks_turns_every_sink_check_red · covers: M2, E1, E1b, E1c · the refutation itself, executed: `source_id=self._url` is applied to a copy of the module, every sink check re-run against it, and each must fail. This is the check that makes M2 a measurement rather than a claim. red: against the current checks it reports that all of them survive.
- tests.unit.test_credential_sinks::test_the_connectable_url_still_carries_the_credential · covers: M6, E5 · what `cv2.VideoCapture` actually received. red: guard — must pass throughout; redacting this would break every RTSP stream.
- tests.unit.test_credential_sinks::test_redaction_happens_once_at_the_boundary · covers: M5 · the count of `redact_url` calls across a full read, measured at the function; one boundary, not one per sink. red: unmeasured.
- tests.unit.test_credential_sinks::test_a_username_with_no_password_is_redacted_at_every_sink · covers: E6 · half a credential is a credential, asserted at the sinks rather than at `redact_url`.
- tests.unit.test_credential_sinks::test_a_malformed_url_leaks_nothing_at_any_sink · covers: E6, A4 · the parse-failure path, at the sinks.
- tests.unit.test_credential_sinks::test_a_credentialed_http_source_is_redacted_at_every_sink · covers: E7 · `redact_url` is scheme-agnostic and the sinks are not RTSP-specific.
- tests.unit.test_credential_sinks::test_the_parent_redaction_checks_are_untouched · covers: M7, A7, E8 · the probe A7 names — `tests/unit/test_rtsp_redaction.py` is byte-identical to the commit that gated `rtsp-credential-redaction`. "The weak check was in the way" is how a weak check becomes no check.
- tests.unit.test_credential_sinks::test_no_check_here_proves_a_guarantee_from_source_text · covers: M3, R:SOURCESCAN · this file contains no `inspect.getsource` assertion. A meta-check, and deliberately so: M3 forbids proving PRODUCTION behaviour by reading source, and the only way to assert that discipline about a test file is to read the test file. red: unasserted.
- tests.unit.test_credential_sinks::test_no_check_here_asserts_on_the_redacting_attribute · covers: R:UPSTREAM · no check reads `safe_url`. That is the defect this node exists to fix and it must not come back under a new name. red: unasserted.
- tests.unit.test_credential_sinks::test_no_check_here_answers_its_own_question · covers: R:SELFANSWER · no assertion is made against a string the test itself passed through `redact_url`. red: unasserted.
- tests.unit.test_credential_sinks::test_box_6_is_not_ticked · covers: R:TICKBOX · m1 box 6 stays `- [ ]`. It cannot be ticked on this node: `redact_url` preserves the query string, so a signed-URL camera still reaches all four sinks with its secret intact. red: unasserted, and this is the check that keeps a real remaining leak from being closed over.
red-first: every check MUST fail first. E1, E1b and E1c name the three mutations that prove it — one per sink origin.

measured before the build, and the reason this node exists: applying `source_id=self._url` to
`_source.py:346` — the raw password into every emitted result and into the feature-cache dict
key — leaves ALL TEN checks in `test_rtsp_redaction.py` GREEN. Mutating `:322` instead does go
red on two. So of the four sinks box 6 names, one binds and three do not.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

RISK, recorded because the PLAN said "no behaviour change is expected" and that turned out
to be wrong. The build found a leak the node had not traced: the `stream_id` a caller passes
to `FrameCollector.add_stream` / `DetectionRouter.register` is very often the camera URL,
credentials and all, and it reached a log record, a `RuntimeError`, dict keys, a thread name
and the routing callback RAW. Fixing it at the boundary (M5) is right, and it changes a
contract:
  (a) callbacks now receive the REDACTED id, not the string the caller passed. A consumer
      that reused the stream id as a reconnect URL now gets a non-connectable one. This is
      the contract change m1's own GROUND/risks section anticipated for `Frame.source_id`,
      arriving on a different field.
  (b) redaction is not injective, so two cameras differing ONLY in credentials now collapse
      to one id. `add_stream` raises `ValueError: Stream already registered`, so that half
      is loud. `register` does not — it silently overwrites, and the surviving stream's
      detections go to the other camera's callback. Documented in `pipeline/README.md`;
      making `register` raise is a behaviour change beyond this node's scope.
  (c) the `@` short-circuit in `safe_stream_id` is itself a security boundary decision, not
      a convenience: `redact_url`'s parse-failure path returns one fixed string, so applying
      it blindly would collapse distinct CREDENTIAL-FREE ids such as `rtsp://host:abc/path`
      onto each other. RFC 3986 userinfo is delimited by `@`, so an id without one cannot
      carry userinfo, and short-circuiting is sound. Bound by
      `test_redaction_happens_once_at_the_boundary`, which reddens when the short-circuit
      is deleted.

OUT OF SCOPE AND STILL OPEN, found by this node and not fixed by it: `redact_url` preserves
the query string. Measured — `rtsp://cam/s?token=SUPERSECRET` -> `rtsp://cam/s?token=SUPERSECRET`,
and `rtsp://u:p@cam/s?auth=SECRET` -> `rtsp://cam/s?auth=SECRET`. A signed-URL camera (HLS,
`?auth=`, `?sig=`) therefore reaches ALL FOUR sinks with its secret intact. Pre-existing and
unchanged by this node — M4 addresses the `user:password@` form only — but a query token is
a credential, so m1 box 6 CANNOT be ticked while this holds. It takes its own node.

## LESSONS
- <lesson> -> add learn <lens>
