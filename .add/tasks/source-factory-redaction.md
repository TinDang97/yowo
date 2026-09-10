---
type: Task
title: open_source refuses a credentialed URL without echoing it
status: done
depth: quick
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - src/yowo/io/
  - tests/unit/
gives:
  - S1 every message `open_source` raises that carries the caller's source string
generated: { by: add/3.5.0, at: 2026-09-10 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: interview, authority: human, interview: "sha256:5d58d94a2d506878", receipt: /tasks/source-factory-redaction.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|R:LEAK=confirm|R:PROXY=confirm|R:MANGLE=confirm" }
  - { by: "Tin Dang", at: 2026-09-10, act: freeze, authority: human, direction: "sha256:a28efabb3f9d6f1b", binding: "sha256:0f4c2e0e74c89eec" }
  - { by: "cli", at: 2026-09-10, act: brief, authority: process, brief: "sha256:cd6fb7312cc94273" }
  - { by: "process:run", at: 2026-09-10, act: run, authority: process, outcome: PASS, receipt: /tasks/source-factory-redaction.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-10, act: gate, authority: human, outcome: PASS, receipt: /tasks/source-factory-redaction.d/runs/1.md, brief: "sha256:9ec26624829aa315", reason: "Eleven checks green on a bound receipt, every cited id present, 1:1 with the CHECKS names. Verified independently rather than on the builder's word: all four original reproductions plus the username-only case now hide the credential, and no message reads 'https:/' with one slash - the tell that the value was laundered through Path() before anyone looked at it. One redaction at entry, before the webcam branch, before scheme dispatch, before Path(), so M2 holds structurally and not by three lines happening to be right. The AST half of test_every_raise_in_open_source_reads_the_redacted_form asserts exactly one redact_url call, positioned below every If/Raise/Path statement, with every Raise referencing the redacted name and none of source/source_str/path - that is what makes a later branch inherit the redaction rather than remember it. R:PROXY is bound the hard way: test_no_check_here_asserts_on_redact_url_alone passed its helper assertion before the fix and failed its factory assertion, which is the whole failure shape in one check. RECORDED EXCEPTION to red-first: test_a_credentialed_rtsp_url_still_opens_as_before was green before the change. E4 defines it as a non-regression guard on a path rtsp-credential-redaction already covers; a red result there would have meant that path was already broken, and manufacturing one would be theatre. The builder flagged this rather than weakening or faking it. ACCEPTED TRADE-OFF, confirmed at interview as A3/A4: a path beginning with // whose authority is malformed now reports as <unparseable url>. Checked by hand - ordinary absolute, relative, ./-prefixed, spaced, non-ASCII and @-containing paths all report byte for byte; //user@nas01/share strips the userinfo, which is correct, a UNC credential being a credential. Full suite 2289 passed 11 skipped, ruff, ruff-format and pyright all clean." }
advised_by: security-reviewer
---
## CARD
goal: `open_source` never echoes a userinfo component, on any branch, for any scheme.
why: `rtsp-credential-redaction` closed the leak inside `RTSPStreamSource`, which redacts at the boundary where the URL is stored. `open_source` is upstream of that boundary — it raises before any source object exists — so three of its `SourceError` messages interpolate the caller's raw string. Reproduced 2026-09-10: `Video file not found: https:/admin:hunter2@10.0.0.5/stream.mp4`, `Image file not found: https:/admin:hunter2@10.0.0.5/frame.jpg`, and `Cannot determine source type for: 'http://admin:hunter2@10.0.0.5/video'.` — the last one via `{source!r}`, so quoting does not save it. This is not a newly discovered surface: `rtsp-credential-redaction`'s own A3 names it — "any source URL with a userinfo component, since `open_source` accepts HTTP(S) streams too -> an HTTP camera URL leaks by the same mechanism through a path nobody checked" — and its `R:LEAK` forbids userinfo being "emitted, logged, raised, returned or used as a key, in whole or in part". The check written for A3, `test_non_rtsp_scheme_with_userinfo_is_redacted`, calls `redact_url()` directly and never calls `open_source`, so it asserts redaction on a string that was never the leaking one. An HTTP camera password is the same secret as an RTSP one; the scheme is not the boundary.
beat: done · next: add status

## RULES
<must>
- M1 No message `open_source` raises contains a userinfo component, for any scheme and on any branch — including the branch that reports an unrecognised source type.
- M2 The redaction runs at the factory boundary, once, before any branch can build a message. A branch added later inherits it rather than having to remember it.
- M3 A redacted message still identifies the source: scheme, host, port and path survive. An operator must be able to tell which camera failed.
- M4 `open_source`'s signature and its raised type stay unchanged. The value passed to the underlying source is the caller's original string; only what is EMITTED is redacted.
- M5 The check calls `open_source` and asserts on what it raises. A check that exercises `redact_url` in isolation does not bind this rule — that is the shape that let this through.
</must>
<reject>
- R:LEAK A userinfo component reaching any raised message, log record, or return value from this factory, in whole or in part. -> "LEAK"
- R:PROXY A check that asserts redaction on a helper's output rather than on the factory's own message. -> "PROXY"
- R:MANGLE A repaired message built from a string that `Path()` has already rewritten — `https://` collapsing to `https:/` proves the value was laundered through a path before anyone looked at it. -> "MANGLE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose credential is in scope; taking any caller-supplied source string, since `open_source` is public API reached from `detect()`, the CLI, and the engine -> if wrong and only RTSP callers matter, an http:// camera keeps leaking on the branch its own task already claimed.
- A2 [which] covers: S1 · the request does not say which branches are in scope; taking every `raise` in `open_source`, not only the three that leak today, because M2 is about the boundary and not about three known lines -> if wrong, the node fixes three strings and the next branch reopens the hole.
- A3 [when] covers: S1 · the request does not say whether a webcam index or a bare path should pass through redaction; taking yes — everything goes through one redaction at entry, since a value with no userinfo is returned unchanged and a special case is a branch that can be forgotten -> if wrong, an unparseable local path is rendered as `<unparseable url>` in its own error and becomes harder to debug. · probe: a nonexistent plain path must still name that path in full.
- A4 [absent] covers: S1 · the request does not say what an unparseable source means; taking `redact_url`'s existing `<unparseable url>` sentinel rather than falling back to the raw string, because a parse failure is exactly when the safe assumption is that a secret is in there -> if wrong, a malformed path yields a message that names nothing useful.
- A5 [order] covers: S1 · the request does not say whether redaction precedes or follows scheme dispatch; taking precedes — the redacted form is computed once at entry and every branch reads it, so the `Path()` conversion that produced `https:/` can never touch the value that gets emitted -> if wrong and redaction runs per-branch, R:MANGLE reappears wherever a branch forgets.
- A6 [experience] covers: S1 · the request does not say who reads the failure; taking an operator wiring up a camera who mistyped a host or a path, and who must be able to tell those apart from a bad credential — so the placeholder must read as a deliberate redaction, not as a mangled URL -> if wrong, the operator retypes the password when the host was wrong.

## PLAN
contract: `open_source(source, ...)` unchanged in signature, return type and raised type. At entry, compute the display form of `source` once via `yowo.io._redact.redact_url`; every `SourceError` message in the function interpolates that display form instead of `source`, `path`, or `{source!r}`. The value handed to `RTSPStreamSource`, `VideoFileSource`, `ImageFileSource` and the webcam branch is the caller's original, unmodified.

## EDGES
- E1 `https://admin:hunter2@10.0.0.5/stream.mp4` — a video extension that does not exist. Message must name the host and path, never the credential, and must not read `https:/`.
- E2 `http://admin:hunter2@10.0.0.5/video` — no recognised extension, so the "cannot determine source type" branch. `{source!r}` must not survive.
- E3 `https://admin:hunter2@10.0.0.5/frame.jpg` — the image branch, which leaks by the same mechanism.
- E4 `rtsp://admin:hunter2@10.0.0.5:554/Streaming/Channels/101` — must keep working exactly as today; this node must not regress the path that is already covered.
- E5 A plain local path that does not exist, and a webcam index `"0"` — neither carries userinfo, and both must be reported unchanged and in full (A3).
- E6 A URL with a username and no password — userinfo is still userinfo.

## CHECKS
all in `tests/unit/test_source_factory_redaction.py`; every one calls `open_source` and asserts on
what it raises (M5).
- test_the_video_branch_hides_userinfo · covers: M1, A1, E1 ·
  `https://admin:hunter2@host/stream.mp4` — the credential is gone from the raised message.
- test_the_image_branch_hides_userinfo · covers: M1, E3 ·
  the image extension branch leaks by the same mechanism and must be closed with it.
- test_the_unknown_type_branch_hides_userinfo · covers: M1, E2 ·
  the `{source!r}` branch — quoting does not redact.
- test_a_username_without_a_password_is_still_redacted · covers: R:LEAK, E6 ·
  userinfo is userinfo whether or not a password follows it.
- test_every_raise_in_open_source_reads_the_redacted_form · covers: M2, A2, A5, R:MANGLE ·
  the redacted form is computed once at entry, so no branch interpolates the raw string and no
  message is built from a value `Path()` already rewrote to `https:/`.
- test_a_redacted_message_still_names_host_and_path · covers: M3, A6 ·
  an operator can tell a mistyped host from a bad credential.
- test_a_plain_path_and_a_webcam_index_are_reported_verbatim · covers: A3, E5 ·
  a value with no userinfo comes back unchanged, so the one path costs nothing.
- test_an_unparseable_source_uses_the_sentinel · covers: A4 ·
  a parse failure yields `<unparseable url>`, never the raw string.
- test_a_credentialed_rtsp_url_still_opens_as_before · covers: E4 ·
  the path `rtsp-credential-redaction` already covers must not regress.
- test_open_source_signature_and_raised_type_are_unchanged · covers: M4 ·
  the source object still receives the caller's original string; only what is emitted changes.
- test_no_check_here_asserts_on_redact_url_alone · covers: M5, R:PROXY ·
  the shape that let this leak through a green suite cannot recur in this file.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
