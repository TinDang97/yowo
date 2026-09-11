---
type: Task
title: A credential in a URL's query or fragment is redacted, and two cameras stay two cameras
status: done
depth: deep
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/io/
  - src/yowo/pipeline/
  - tests/unit/
gives:
  - S1 `redact_url` and `safe_stream_id` — the two boundaries every credential crosses
generated: { by: add/3.5.0, at: 2026-09-11 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:f6a39a4ed33b58d5", receipt: /tasks/query-string-credentials.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|R:DENYLIST=confirm|R:COLLAPSE=confirm|R:BYTEDRIFT=confirm|R:REVERSIBLE=confirm|R:TICKBOX=confirm" }
  - { by: "cli", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:0d43bd3dc4a716bb", binding: "sha256:3fdca6e11454c863" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:2cfa2a574d6c1f7c" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/query-string-credentials.d/runs/1.md }
  - { by: "cli", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:9daa411a3a424266", binding: "sha256:3fdca6e11454c863" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:d823c7c2ee270d2e" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/query-string-credentials.d/runs/2.md }
  - { by: "cli", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:c75a81cb52dcb221", binding: "sha256:3fdca6e11454c863" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:d6bcec7aeec4cfa2" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/query-string-credentials.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: human, outcome: PASS, receipt: /tasks/query-string-credentials.d/runs/3.md, brief: "sha256:d6bcec7aeec4cfa2", reason: "37 check items green, zero skipped, every rule bound. Suite 2404 passed / 11 skipped, ruff clean, pyright 0 errors. THE MEASURED CASE IS CLOSED: rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2 now yields rtsp://10.0.0.5:554/s?token=&password=#q0734d834 — neither hunter2 nor SECRET survives, at either boundary. REFUTED BY MUTATION, six of them, each executed: keeping query values reddens 15; dropping the identity digest reddens 4; computing the digest after redaction instead of before reddens 3; reverting safe_stream_id's short-circuit to @-only reddens 3; dropping the byte-identical early return reddens 1; sorting the query keys reddens 1. TWO OF MY OWN CHECKS WERE WRONG AND REFUTATION FOUND BOTH, NOT READING. test_query_keys_and_their_order_survive used alpha and beta, already alphabetical, so a sorted() in the implementation was indistinguishable from preserving order and that mutation SURVIVED the first draft; now zulu and alpha, and it reddens. That is the M9 defect in my own work. The same check also asserted a query value was absent by scanning the whole pre-fragment string, which matches the 1 in the host 10.0.0.5 and turned a correct implementation red for the wrong reason; now scoped to the query. I WALKED INTO MY OWN M12 TOO: the first gate refused with thirteen rules unbound because every parametrised check was cited by its bare name — the exact lesson I filed two nodes ago. Citations are now one line per parametrisation. The gate then refused twice more, for A2 and A7, and was right both times. I ALSO INTRODUCED A REGRESSION AND THE PARENT NODES CAUGHT IT: moving urlsplit outside the try broke four checks across test_rtsp_redaction, test_credential_sinks and test_source_factory_redaction, because urlsplit itself raises on an invalid IPv6 authority long before .port is touched. Restored inside the guard, and not one of those checks was weakened to get green. SCOPE HELD DELIBERATELY: the parse-failure path still returns the bare <unparseable url> and still collapses identity there. Appending a digest would have preserved identity but broken an == assertion in a gated node's check, and editing another node's checks to widen my own scope is what the previous node's M7 exists to prevent. Recorded as residual, not fixed. ONE CHANGE-REQUEST APPLIED to credential-sinks-are-bound's test_box_6_is_not_ticked, which held box 6 open on the query leak and asserted that leak reproduced — with its own note saying that if it stopped holding, the follow-up had landed and the reason must be RE-DERIVED rather than left asserting a fiction. It has landed. The check keeps its purpose, now carries the reason that is true, and gains a guard that the closed form has not reopened. BOX 6 STILL CANNOT BE TICKED, and I measured the reason rather than assuming it: rtsp://h/live/S3CR3T-signed/stream comes back unchanged. A path-borne token survives, and redaction cannot fix that one — the path IS the camera's identity, so redacting it would collapse every camera on a host onto one identifier, which is exactly the cross-delivery failure the previous node fixed at the cause. Whether to tick box 6 anyway, naming path-borne secrets as an operator constraint rather than a code guarantee, is a milestone decision this node may not make." }
advised_by: security-reviewer
---
## CARD
goal: A secret carried in a URL's query or fragment is unreadable at every sink, and two cameras that differ only in their query remain two distinct identifiers.
why: `redact_url` strips userinfo and passes the query and fragment through untouched. Measured 2026-09-11:
  `rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2` -> `rtsp://10.0.0.5:554/s?token=SECRET&password=hunter2`
  That is the damning form. The userinfo IS stripped, so the `@` is gone and the string wears the visual signature of a redacted URL while carrying the literal password. It defeats eyeball review of a log, and it defeats a reviewer diffing before against after, because the diff shows redaction happening.
  Worse at the pipeline boundary. `safe_stream_id` short-circuits on `"@" not in stream_id`, so `rtsp://host:abc/p?token=X` is returned BYTE-IDENTICAL and the token reaches all seven log sites, the `RuntimeError`, the dict keys, the thread name and the routing callback. Both functions have an early return that skips query redaction, and `#frag=SECRET` leaks by the same two paths.
  Signed-URL cameras are the normal case for HLS and for a large share of IP cameras: `?auth=`, `?sig=`, `?token=`, `?st=`/`?e=` on CDN-fronted streams. m1 box 6 says "no credential", and a query token is a credential, so this is what holds that box open. `credential-sinks-are-bound` bound the four sinks and recorded this as out of its scope.
next: the sinks are already bound — `_assert_sink_clean` catches this today. What is missing is a URL variant driven through them, and a `redact_url` that redacts.

## RULES
<must>
- M1 Every query value and the fragment are redacted, whatever the key is called. A denylist of credential-shaped key names fails open on the key nobody listed, and the key nobody listed is the one that leaks.
- M2 Redaction preserves identity. Two URLs differing anywhere in query or fragment produce two different results. A short digest of the ORIGINAL query and fragment is appended for exactly this purpose, because the id is a dict key and `DetectionRouter.register` overwrites silently — collapsing two cameras cross-delivers one camera's detections to the other's callback.
- M3 The digest is not a way back to the secret, and where it is weak the node says so rather than implying strength it does not have.
- M4 Query KEYS survive, so an operator can still see the shape of what was there. `?token=&channel=` tells a reader which parameters the camera used; a dropped query tells them nothing.
- M5 Both early returns are fixed. `redact_url`'s `if "@" not in parts.netloc: return url` and `safe_stream_id`'s `if "@" not in stream_id: return stream_id` each skip query redaction today, and a fix to one alone leaves the other leaking.
- M6 A URL with nothing to redact — no userinfo, no query, no fragment — is returned BYTE-IDENTICAL. That is what `credential-sinks-are-bound` had to fix as a regression, and it must not be reintroduced: `rtsp://host:abc/path` must survive unchanged even though it cannot be parsed.
- M7 The four sinks are re-driven with a query-credentialed URL. The sinks are already bound; this node proves they are bound against THIS credential form too, using the same `_assert_sink_clean` rather than a new one.
- M8 The redaction stays at the two boundaries. No sink learns about query strings.
</must>
<reject>
- R:DENYLIST Deciding what to redact from a query by matching key names. -> "DENYLIST"
- R:COLLAPSE Any change that makes two distinct camera URLs produce one identifier. -> "COLLAPSE"
- R:BYTEDRIFT A URL with nothing to redact coming back other than byte-identical. -> "BYTEDRIFT"
- R:REVERSIBLE A digest, encoding or marker from which the original secret can be recovered. -> "REVERSIBLE"
- R:TICKBOX Ticking m1 box 6 before the four sinks are green against this credential form. -> "TICKBOX"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose secret; taking an operator's signed camera URL — HLS `?token=`, `?auth=`, CDN `?st=`/`?e=` — which is the normal way a large share of cameras authenticate, not an exotic case -> if wrong and only userinfo is ever used here, this node costs a digest suffix on ids that have no query at all, which is nothing, since M6 leaves those byte-identical.
- A2 [which] covers: S1 · the maintainer chose redact-every-value-plus-identity-digest over a key denylist and over dropping the query; taking that verbatim -> if wrong, ids carrying a query grow a suffix. · probe: `?channel=1` and `?channel=2` must remain distinct after redaction.
- A3 [when] covers: S1 · the request does not say when the digest is computed; taking BEFORE redaction, over the original query and fragment, since a digest of the redacted form is identical for every URL and preserves no identity at all -> if wrong, every query-bearing camera collapses to one id, which is R:COLLAPSE.
- A4 [absent] covers: S1 · the request does not say what happens to a URL with no query and no fragment; taking byte-identical return with NO digest appended, so nothing that works today changes shape -> if wrong, every existing `source_id` for a file, a directory or a webcam changes, breaking anything keyed on it. · probe: a corpus of credential-free ids round-trips byte-for-byte.
- A5 [order] covers: S1 · the request does not say whether query keys may be reordered; taking original order preserved, because reordering merges `?a=1&b=2` with `?b=2&a=1` and the whole point of M2 is that distinct inputs stay distinct -> if wrong, two orderings of one query collapse.
- A6 [experience] covers: S1 · the request does not say who reads the result; taking an operator reading a log line who must be able to tell WHICH camera failed and see that redaction happened, hence keys kept, values emptied, and a visible digest marker rather than a silently truncated URL -> if wrong, a redacted id is indistinguishable from a camera that genuinely had empty parameters.
- A7 [absent] covers: S1 · the request does not say what a weak digest implies; taking honest documentation — a short digest over a LOW-ENTROPY query such as `?channel=2` is effectively an encoding of it, and that is acceptable because a channel number is not a secret, while a high-entropy token is not recoverable from it -> if wrong, someone reads the digest as a security control rather than an identity control. · probe: the docstring says which of the two it is.

## PLAN
contract: `redact_url` gains query and fragment redaction — every value emptied, keys and order kept, and `#q<digest>` appended when a query or fragment was present. `safe_stream_id`'s short-circuit narrows from "no `@`" to "no `@`, no query, no fragment", so a query-only credential still reaches the redactor. Nothing else changes. `tests/unit/test_query_credentials.py` drives the query form through the SAME `_assert_sink_clean` the sink node already uses, so this proves the existing bindings hold against a new credential form rather than adding a parallel set.

## EDGES
- E1 `rtsp://camop:hunter2@10.0.0.5:554/s?token=SECRET&password=hunter2` — the measured case. Neither `hunter2` nor `SECRET` survives, at any sink.
- E2 `rtsp://h/s?token=SECRET` — no `@` at all. Today `safe_stream_id` returns it byte-identical; it must not.
- E3 `rtsp://host:abc/p?token=X` — no `@`, unparseable authority, query credential. The worst case: it passes through both boundaries untouched today.
- E4 `rtsp://host:abc/path` — no `@`, unparseable, NOTHING to redact. Must stay byte-identical (M6, R:BYTEDRIFT). This is the regression `credential-sinks-are-bound` had to fix.
- E5 `?channel=1` versus `?channel=2` — must remain two identifiers (M2, R:COLLAPSE).
- E6 `rtsp://h/s#frag=SECRET` — the fragment, which leaks by the same two paths as the query.
- E7 `?flag` with no `=`, and `?a=&b=` already empty — no crash, and still distinct from each other.
- E8 `?a=1&b=2` versus `?b=2&a=1` — distinct, because order is preserved and the digest is over the original string.
- E9 A file path, a directory, a webcam index, a plain name — byte-identical, no digest (A4).

## CHECKS
all in `tests/unit/test_query_credentials.py`. The sink check reuses `_assert_sink_clean` from `test_credential_sinks.py` rather than defining a parallel one (M7).
Cited one line per parametrisation: a bare name cannot bind against a reported id ending in
`[param]`, and the gate correctly refused a PASS while thirteen rules read as unbound (method M12,
filed by me two nodes ago and walked into again here).

- tests.unit.test_query_credentials::test_the_measured_mixed_form_leaks_nothing[redact_url] · covers: M1, E1 · the case that caused this node; neither `hunter2` nor the token survives, and the host does.
- tests.unit.test_query_credentials::test_the_measured_mixed_form_leaks_nothing[safe_stream_id] · covers: M1, E1 · the case that caused this node; neither `hunter2` nor the token survives, and the host does.
- tests.unit.test_query_credentials::test_a_query_credential_with_no_userinfo_is_redacted[redact_url] · covers: M1, M5, E2 · no `@`, so both early returns skipped it.
- tests.unit.test_query_credentials::test_a_query_credential_with_no_userinfo_is_redacted[safe_stream_id] · covers: M1, M5, E2 · no `@`, so both early returns skipped it.
- tests.unit.test_query_credentials::test_an_unparseable_authority_with_a_query_credential_is_redacted[redact_url] · covers: M1, M5, E3 · the worst case — untouched by both boundaries.
- tests.unit.test_query_credentials::test_an_unparseable_authority_with_a_query_credential_is_redacted[safe_stream_id] · covers: M1, M5, E3 · the worst case — untouched by both boundaries.
- tests.unit.test_query_credentials::test_the_fragment_is_redacted[redact_url] · covers: M1, E6 · the fragment leaks by the same two paths as the query.
- tests.unit.test_query_credentials::test_the_fragment_is_redacted[safe_stream_id] · covers: M1, E6 · the fragment leaks by the same two paths as the query.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[token] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[sessionId] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[x] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[auth] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[sig] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[st] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[e] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_every_value_goes_whatever_its_key_is_called[zzz_unknown_key] · covers: M1, R:DENYLIST · the key nobody listed is the one that leaks.
- tests.unit.test_query_credentials::test_query_keys_and_their_order_survive · covers: M4, A5, E8 · keys deliberately NOT alphabetical: with `alpha`/`beta` a `sorted()` is indistinguishable from preserving order, and that mutation SURVIVED the first draft.
- tests.unit.test_query_credentials::test_two_cameras_differing_only_in_query_stay_two_identifiers[rtsp://h/s?channel=1-rtsp://h/s?channel=2] · covers: M2, A2, R:COLLAPSE, E5 · the id is a dict key and `register` overwrites silently; a collapse cross-delivers detections.
- tests.unit.test_query_credentials::test_two_cameras_differing_only_in_query_stay_two_identifiers[rtsp://h/s?a=1&b=2-rtsp://h/s?b=2&a=1] · covers: M2, A2, R:COLLAPSE, E5 · the id is a dict key and `register` overwrites silently; a collapse cross-delivers detections.
- tests.unit.test_query_credentials::test_two_cameras_differing_only_in_query_stay_two_identifiers[rtsp://h/s#one-rtsp://h/s#two] · covers: M2, A2, R:COLLAPSE, E5 · the id is a dict key and `register` overwrites silently; a collapse cross-delivers detections.
- tests.unit.test_query_credentials::test_two_cameras_differing_only_in_query_stay_two_identifiers[rtsp://h/s?x=1-rtsp://h/s?x=1&y=2] · covers: M2, A2, R:COLLAPSE, E5 · the id is a dict key and `register` overwrites silently; a collapse cross-delivers detections.
- tests.unit.test_query_credentials::test_the_digest_is_computed_before_redaction · covers: M2, A3 · a digest of the REDACTED form is identical for every URL sharing the same keys and distinguishes nothing.
- tests.unit.test_query_credentials::test_the_digest_does_not_carry_the_secret · covers: M3, R:REVERSIBLE · not raw, not hex, not base64, not url-quoted.
- tests.unit.test_query_credentials::test_the_digest_is_documented_as_an_identity_control · covers: A7 · the probe A7 names. The guarantee here IS the documentation — a reader must not mistake eight hex characters for a security control — so reading the docstring is reading the subject, not a proxy for it.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[rtsp://10.0.0.5:554/stream] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[rtsp://10.0.0.5/stream] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[http://camera.local/feed] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[/var/media/clip.mp4] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[/var/media/] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[webcam:0] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[cam-0] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_a_url_with_nothing_to_redact_is_byte_identical[] · covers: M6, A4, R:BYTEDRIFT, E9 · no digest, no drift, at either boundary.
- tests.unit.test_query_credentials::test_an_unparseable_authority_with_nothing_to_redact_is_byte_identical · covers: M6, R:BYTEDRIFT, E4 · the exact regression `credential-sinks-are-bound` fixed at the cause.
- tests.unit.test_query_credentials::test_degenerate_query_forms_do_not_crash_and_stay_distinct · covers: E7 · `?flag`, `?a=&b=`, a bare `?`, an empty fragment.
- tests.unit.test_query_credentials::test_the_four_sinks_are_clean_against_a_query_credential · covers: M7 · the sinks are already bound; this proves they hold for a new credential form.
- tests.unit.test_query_credentials::test_redaction_still_happens_only_at_the_two_boundaries · covers: M8 · no sink learned about query strings.
- tests.unit.test_query_credentials::test_box_6_is_not_ticked_until_this_is_green · covers: R:TICKBOX · box 6 says 'no credential', and a query token is a credential.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
