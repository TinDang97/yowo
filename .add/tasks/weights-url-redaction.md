---
type: Task
title: A credentialed weights URL is not echoed or logged
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/cli/
  - src/yowo/models/
  - tests/unit/
gives:
  - S1 `yowo models` — the listing that prints every registered model's weights URL
  - S2 the weight-download diagnostics in `models/_weights.py` — retries exhausted, and stalled stream
generated: { by: add/3.5.0, at: 2026-09-09 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:3330953313f255ca", receipt: /tasks/weights-url-redaction.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:USERINFO=confirm|R:RAWFALLBACK=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:d8d9a64964eb4441", binding: "sha256:f0152cc85f9b363a" }
  - { by: "cli", at: 2026-09-09, act: brief, authority: process, brief: "sha256:781cb938966bd2d4" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/weights-url-redaction.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-09, act: refreeze, authority: human, direction: "sha256:19bb77f36414f87e", binding: "sha256:f0152cc85f9b363a" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/weights-url-redaction.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-09, act: refreeze, authority: human, direction: "sha256:c71619a834a4718a", binding: "sha256:f0152cc85f9b363a" }
  - { by: "process:run", at: 2026-09-09, act: run, authority: process, outcome: PASS, receipt: /tasks/weights-url-redaction.d/runs/3.md }
  - { by: "Tin Dang", at: 2026-09-09, act: gate, authority: human, outcome: PASS, receipt: /tasks/weights-url-redaction.d/runs/3.md, brief: "sha256:25b4d7dcb732ab08", reason: "11 checks green on a bound receipt. Review returned one R:USERINFO defect the first build missed: _download interpolated (last_exc) unscrubbed, and requests keeps userinfo in PreparedRequest.url, so HTTPError/InvalidSchema/InvalidURL carried the credential. Verified empirically, fixed with _scrub_credential, covered by three tests built from real requests exceptions." }
advised_by: edge-reliability-operator
---
## CARD
goal: A credential in a custom model's weights URL is never echoed, logged or raised.
why: found by the security residue lens while verifying `rtsp-credential-redaction`, in code that task does not own.
  - `cli/_main.py:883` — `yowo models` prints `m.default_weights_url` verbatim for every registered model.
  - `models/README.md:119` — the documented way to register a custom model shows an S3 URL, so users are
    actively guided toward putting a fetchable URL in the registry. `https://KEY:SECRET@bucket.s3...` is
    a completely ordinary shape for one.
  - `models/_weights.py` — download failure paths interpolate the URL into messages.
Built-in models use public ultralytics asset URLs and carry no credential, so this is conditional on a
user registering their own. `redact_url()` already exists and is exported from `yowo.io`, so the fix is
to apply it at the emit points — not to build anything new.

NOT a HARD-STOP on rtsp-credential-redaction's gate, and the reasoning is recorded so it can be
challenged: that task's change introduces no leak and removes four. Blocking its gate over an adjacent
pre-existing issue would leave the RTSP credentials leaking in order to punish a fix.
beat: done · next: add status

## RULES
<must>
- M1 No emit point — stdout, log record, warning or exception message — carries a URL's userinfo
  component.
- M2 A URL carrying no credential is emitted byte for byte unchanged, so existing output, existing
  `source_id` values and existing tests are untouched.
- M3 The registry stores the URL with its credential intact — redaction happens at the emit point, not
  at registration, because `_download` needs the credential to work.
- M4 `models/README.md`'s registration example does not model a credentialed URL, and states the rule.
</must>
<reject>
- R:USERINFO No userinfo component may reach a stream, a log record or an exception message -> "USERINFO"
- R:RAWFALLBACK No emit point may fall back to the raw URL when redaction cannot parse it -> "RAWFALLBACK"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1, S2 · the request does not say whose URL carries a secret; taking: only a
  user-registered model can — the 10 built-ins use public ultralytics asset URLs and carry no
  credential — so this protects the registrant, not the maintainer
  -> scoping to built-ins would leave the only case that has a secret unredacted.
- A2 [which] covers: S1, S2 · the request does not say which emit points are in; taking: the three that
  interpolate a URL — `cli/_main.py:883`, `_download`'s retries-exhausted message, and
  `_attempt_download`'s stall message. `_warn_unpinned` names a weight stem and no URL, so it is
  verified clean rather than changed · probe: no emit point in `cli/` or `models/` interpolates a raw
  URL -> one unredacted path makes the other two decorative.
- A3 [when] covers: S1, S2 · the request does not say when to redact; taking: unconditionally at the
  emit point, on every URL, rather than conditionally when a credential is detected — `redact_url`
  already returns a credential-free URL unchanged (M2), so the conditional buys nothing and is one more
  branch to get wrong -> a "does it look credentialed?" test is exactly the check that misses a shape.
- A4 [absent] covers: S1, S2 · the request does not say what an unparseable URL emits; taking:
  `redact_url`'s existing `"<unparseable url>"`, which emits nothing of the string
  -> falling back to the raw value leaks precisely where a malformed authority hides a credential
  (R:RAWFALLBACK).
- A5 [order] covers: S1 · the request does not say whether the table survives; taking: `yowo models`
  keeps its four columns and its widths, and only the URL cell changes
  -> re-laying out the table would break anyone parsing that output.
- A6 [experience] covers: S1, S2 · the request does not say what an operator needs from a failed
  download; taking: scheme, host, port and path are preserved so the bucket and the object are still
  identifiable -> "download failed" with no identifier is unactionable across a set of private
  buckets, which is why `redact_url` strips userinfo and nothing else.

## PLAN
contract:
  - S1 `models_command` emits `redact_url(m.default_weights_url)`; column layout unchanged.
  - S2 `_download` and `_attempt_download` interpolate `redact_url(url)` into every message they raise.
  - S3 `models/README.md`'s custom registration example uses a credential-free URL and says that a
    credential in `default_weights_url` is redacted wherever it is displayed.
strategy: tests red against the three emit points first, then the one-line substitutions, then the doc.
regression floor: `test_registry.py`, `test_weight_integrity.py`, `test_cli.py` and the `redact_url`
  tests from `rtsp-credential-redaction` stay green.

## EDGES
- E1 Three shapes in one test: `user:pass@`, a bare `user@`, and neither — the third must come back
  identical, byte for byte (M2).
- E2 A malformed authority (`https://h:notaport@b/x.pt`) must emit `<unparseable url>`, never the raw
  string (R:RAWFALLBACK).
- E3 The retries-exhausted message must still name the host and the object path (A6).

## CHECKS
- test_models_command_never_prints_userinfo · covers: M1, R:USERINFO, A2 · a registered credentialed URL
  does not appear in `yowo models` output.
- test_models_command_leaves_clean_urls_byte_identical · covers: M2, A5, E1 · the 10 built-in URLs print
  exactly as they do today.
- test_download_failure_message_redacts_the_url · covers: M1, A2, E3 · retries exhausted, no userinfo,
  host and path still present.
- test_stalled_download_message_redacts_the_url · covers: M1, A2 · the chunk-deadline path.
- test_unparseable_url_never_falls_back_to_raw · covers: R:RAWFALLBACK, A4, E2 · a malformed authority emits `<unparseable url>`, never the raw string.
- test_registry_still_stores_the_credential · covers: M3 · redaction did not damage the fetchable URL.
- test_readme_example_registers_no_credentialed_url · covers: M4 · the doc no longer models a credentialed URL, and states the rule.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
