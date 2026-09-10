---
type: Task
title: Every sink named in box 6 is bound by a check that fails when the credential reaches it
status: direction
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
- M2 Every one of those checks goes RED when `_source.py:346` is changed to `source_id=self._url`. That single mutation is the node's refutation, it is recorded, and a check that survives it does not count as binding.
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
- E1 The M2 mutation — `source_id=self._url`. Every sink check must go red. Today all ten survive it.
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
- tests.unit.test_credential_sinks::test_the_mutation_that_leaks_turns_every_sink_check_red · covers: M2 · the refutation itself, executed: `source_id=self._url` is applied to a copy of the module, every sink check re-run against it, and each must fail. This is the check that makes M2 a measurement rather than a claim. red: against the current checks it reports that all of them survive.
- tests.unit.test_credential_sinks::test_the_connectable_url_still_carries_the_credential · covers: M6, E5 · what `cv2.VideoCapture` actually received. red: guard — must pass throughout; redacting this would break every RTSP stream.
- tests.unit.test_credential_sinks::test_redaction_happens_once_at_the_boundary · covers: M5 · the count of `redact_url` calls across a full read, measured at the function; one boundary, not one per sink. red: unmeasured.
- tests.unit.test_credential_sinks::test_a_username_with_no_password_is_redacted_at_every_sink · covers: E6 · half a credential is a credential, asserted at the sinks rather than at `redact_url`.
- tests.unit.test_credential_sinks::test_a_malformed_url_leaks_nothing_at_any_sink · covers: E6, A4 · the parse-failure path, at the sinks.
- tests.unit.test_credential_sinks::test_a_credentialed_http_source_is_redacted_at_every_sink · covers: E7 · `redact_url` is scheme-agnostic and the sinks are not RTSP-specific.
- tests.unit.test_credential_sinks::test_the_parent_redaction_checks_are_untouched · covers: M7, A7, E8 · the probe A7 names — `tests/unit/test_rtsp_redaction.py` is byte-identical to the commit that gated `rtsp-credential-redaction`. "The weak check was in the way" is how a weak check becomes no check.
red-first: every check MUST fail first, and M2 names the single mutation that proves it.

measured before the build, and the reason this node exists: applying `source_id=self._url` to
`_source.py:346` — the raw password into every emitted result and into the feature-cache dict
key — leaves ALL TEN checks in `test_rtsp_redaction.py` GREEN. Mutating `:322` instead does go
red on two. So of the four sinks box 6 names, one binds and three do not.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
