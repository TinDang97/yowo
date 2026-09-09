---
type: Task
title: No credential in logs, exceptions, results, or cache keys
status: direction
depth: standard
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/io/_source.py
  - src/yowo/types.py
gives:
  - S1 `Frame.source_id` as published for an RTSP source
  - S2 the exception messages `RTSPStreamSource` raises on open failure and on reconnect timeout
  - S3 the feature-cache key, which is `source_id` (cache/__init__.py:102)
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: security-reviewer
---
## CARD
goal: A camera password cannot be recovered from anything yowo emits — not a result, not an exception, not a key in memory.
why: it leaks today, verbatim, on all four surfaces the exit box names. Demonstrated with
`rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101`:
  - `_source.py:312` — `raise SourceError(f"Cannot open RTSP stream: {self._url}")` → the exception
    carries `hunter2` and travels to every caller, every log handler and every crash reporter.
  - `_source.py:350` — the reconnect-timeout message interpolates the same `_url`.
  - `_source.py:336` — `source_id=self._url`, so every `Frame` and every detection result derived from
    it carries the password into whatever consumes result JSON.
  - `cache/__init__.py:102` — `source_id` IS the feature-cache key, so the password becomes a dict key
    held in memory for the process lifetime.
RTSP credentials are usually long-lived, shared across an estate of cameras, and rarely rotated.
beat: scaffold · next: author rtsp-credential-redaction's RULES, ASSUMPTIONS and CHECKS, then add freeze rtsp-credential-redaction

## RULES
<must>
- M1 No credential appears in `Frame.source_id`, in any exception message raised by a source, or in any
  cache key derived from either.
- M2 Redaction happens at the boundary where the URL is stored, so a surface added later inherits it
  rather than needing its own fix.
- M3 A redacted identifier still distinguishes two different streams, so it remains usable as a key.
</must>
<reject>
- R:LEAK No userinfo component of a source URL may be emitted, logged, raised, returned or used as a
  key, in whole or in part -> "LEAK"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who reads `source_id`; taking: any consumer of result
  JSON, including log shippers and downstream stores yowo has no visibility into — which is why the fix
  is redaction at the source rather than filtering at each sink -> a sink nobody remembered keeps
  receiving the password.
- A2 [which] covers: S1 · the request does not say what is removed; taking: the ENTIRE userinfo
  component, both username and password, replaced by nothing — `rtsp://10.0.0.5:554/path`. A username is
  identifying and half a credential · probe: neither `admin` nor `hunter2` survives redaction
  -> keeping the username leaves an attacker needing only the password, and leaks the account name.
- A3 [when] covers: S1 · the request does not say whether non-RTSP sources are covered; taking: any
  source URL with a userinfo component, since `open_source` accepts HTTP(S) streams too
  -> an HTTP camera URL leaks by the same mechanism through a path nobody checked.
- A4 [absent] covers: S1 · the request does not say what a URL with no credentials becomes; taking:
  it is returned unchanged, byte for byte -> gratuitously rewriting clean URLs breaks existing
  `source_id` values for every file and webcam user.
- A5 [order] n/a · redaction is a pure function of one string.
- A6 [experience] covers: S1 · the request does not say who debugs a stream problem; taking: an operator
  who still needs to tell WHICH camera failed, so host, port and path are preserved
  -> redacting the whole URL makes every camera fault indistinguishable.
- A7 [who] n/a · exception messages have no authorization surface.
- A8 [which] covers: S2 · the request does not say which exceptions; taking: BOTH the open failure
  (`:312`) and the reconnect timeout (`:350`), and any future message, which is why M2 stores the
  redacted form rather than redacting per call site -> a third message added later leaks again.
- A9 [when] covers: S2 · the request does not say whether the connectable URL is still needed; taking:
  the raw URL is retained privately for `cv2.VideoCapture` and never interpolated into a message
  · probe: the object can still connect after redaction -> redacting the URL used to connect breaks
  every RTSP stream, which is a far worse outcome than the leak.
- A10 [absent] covers: S2 · the request does not say what an exception says when the URL is malformed
  and cannot be parsed; taking: fall back to emitting NOTHING of the URL rather than the raw string
  -> a parse failure becomes the one path that leaks, which is exactly where an attacker aims.
- A11 [order] n/a · messages are independent.
- A12 [experience] covers: S2 · the request does not say what an operator sees; taking: the redacted
  URL, so the message still names the failing camera -> "Cannot open RTSP stream" with no identifier
  is unactionable on an estate of cameras.
- A13 [who] n/a · the cache key is internal.
- A14 [which] covers: S3 · the request does not say whether the key changes; taking: the key becomes the
  redacted `source_id`, automatically, because the cache reads `source_id` and does not build its own
  -> a separately-derived key would be a second place to forget.
- A15 [when] covers: S3 · the request does not say what happens to entries cached under an unredacted
  key in a running process; taking: nothing — the cache is in-memory and per-process, so there is no
  migration to perform -> inventing a migration for a cache that does not survive the process.
- A16 [absent] covers: S3 · the request does not say what if two streams differ ONLY by credentials —
  same host, port and path, different accounts; taking: they collide into one cache key, and that is
  accepted: the feature cache is keyed on frame content fingerprint as well, so a collision costs a
  cache miss, not a wrong result · probe: colliding sources do not return each other's features
  -> if the fingerprint check were absent this would be a correctness bug, not a performance one.
- A17 [order] n/a · keys are compared, not ordered.
- A18 [experience] covers: S3 · the request does not say who sees cache keys; taking: anyone with a heap
  dump or a debugger, which is precisely the threat — a password living for the process lifetime as a
  dict key survives log scrubbing entirely -> redacting only the emitted surfaces would leave the
  in-memory copy, which is the hardest one to notice.

## PLAN
contract: a `redact_url(url: str) -> str` helper in `src/yowo/io/`, applied where the URL is STORED
  (`self._url` keeps the connectable form; a new `self._safe_url` carries the redacted one). Every
  emitted surface reads the safe form. `Frame.source_id` becomes the redacted URL.
strategy: redact at the boundary, not at each sink, so M2 holds for surfaces added later. The
  connectable URL never leaves the object.
regression floor: `tests/unit/test_source.py`, `test_cache.py` and `test_streaming.py` stay green —
  `source_id` for files, directories and webcams must be untouched.

## EDGES
- E1 A malformed URL that cannot be parsed must emit nothing of the URL rather than falling back to the
  raw string. The parse-failure path is where a leak would hide.
- E2 A URL with a username but no password (`rtsp://admin@host/path`) still has its userinfo removed —
  a bare username is identifying and is half a credential.

## CHECKS
- test_source_id_carries_no_credential · covers: M1, A2 · neither user nor password survives.
- test_open_failure_exception_carries_no_credential · covers: M1, A8 · the `:312` message.
- test_reconnect_timeout_message_carries_no_credential · covers: M1, A8 · the `:350` message.
- test_cache_key_carries_no_credential · covers: M1, A14 · the key the cache actually stores.
- test_redacted_url_still_identifies_the_stream · covers: M3, A6 · host, port and path survive.
- test_connectable_url_is_retained_privately · covers: A9 · the object can still connect.
- test_clean_url_is_returned_unchanged · covers: A4 · no gratuitous rewriting.
- test_username_without_password_is_still_removed · covers: E2 · half a credential is a credential.
- test_malformed_url_emits_nothing_of_it · covers: E1, A10 · the parse-failure path does not leak.
- test_non_rtsp_scheme_with_userinfo_is_redacted · covers: A3 · HTTP camera URLs too.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
