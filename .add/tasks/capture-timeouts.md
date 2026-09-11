---
type: Task
title: Every blocking I/O call bounded by a stated timeout
status: done
depth: standard
milestone: m2-survive-week-two
scope:
  - src/yowo/io/
  - tests/unit/
gives:
  - S1 `RTSPStreamSource(url, *, open_timeout_ms, read_timeout_ms, ...)` — a network capture is constructed with both timeouts, never bare
  - S2 `open_source(source, *, open_timeout_ms, read_timeout_ms, ...)` — the factory carries them to the only source that can use them
  - S3 a structural check that fails on a network capture constructed without both timeouts
depends_on:
  - /tasks/rtsp-credential-redaction.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:c1b7c77b3dd8a220", receipt: /tasks/capture-timeouts.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|R:HARDCODE=confirm|R:SILENTSHORTEN=confirm|R:MOCKONLY=confirm|R:PARTIAL=confirm|R:SCOPECREEP=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:da2d769b6a3e60d1", binding: "sha256:df5eedaf888c7838" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:ba93917a77430ba4" }
  - { by: "builder", at: 2026-09-11, act: replan, authority: process, note: "A2 (default 30000ms) and A4 (None = leave the backend default) resolve to the SAME behaviour, because FFmpeg's default IS ~30s — measured 30.08s. So the default is None, meaning no property is set at all, which preserves today byte-for-byte rather than merely numerically. This also makes 'the caller did not ask for a timeout' detectable, which A8 needs: open_source must refuse the parameters on a non-network source, and it cannot distinguish an explicit 30000 from an unpassed default. The human's choice was 'preserve today, exactly'; None delivers that more exactly than setting the literal number." }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/capture-timeouts.d/runs/1.md }
  - { by: "Tin Dang", at: 2026-09-11, act: refreeze, authority: human, direction: "sha256:f15162e77a2e0781", binding: "sha256:df5eedaf888c7838" }
  - { by: "cli", at: 2026-09-11, act: brief, authority: process, brief: "sha256:f879c32d899ca312" }
  - { by: "process:run", at: 2026-09-11, act: run, authority: process, outcome: PASS, receipt: /tasks/capture-timeouts.d/runs/2.md }
  - { by: "Tin Dang", at: 2026-09-11, act: gate, authority: process, outcome: PASS, receipt: /tasks/capture-timeouts.d/runs/2.md, brief: "sha256:f879c32d899ca312", reason: "15 bound check items green on receipt 2, zero skipped, every Must and Reject bound. Suite 2448 passed / 11 skipped, ruff clean, ruff-format clean, pyright 0 errors. WHAT IS NOW TRUE. How long yowo waits on a wedged camera is a number yowo states. RTSPStreamSource and open_source take open_timeout_ms and read_timeout_ms; every capture for a stream is built through one path, _new_cap, so the periodic reconnect is bounded by the same number as the first open. Unconfigured, NO property is set and FFmpeg's own default stays in force -- today's behaviour byte-for-byte, not merely the same number. THE PREMISE WAS WRONG AND I MEASURED IT BEFORE AUTHORING. m2's card says 'there is no capture timeout anywhere in src/', which reads as an unbounded hang. Measured against an unroutable host and against a server that completes the handshake and then never speaks, both identical: bare -> not opened after 30.08s; with OPEN/READ at 2000ms -> not opened after 2.02s. The stall is BOUNDED; the defect is that 30s is FFmpeg's choice and no caller could change it. Authoring to the milestone's prose would have produced a node solving a problem that does not exist. THE BOX ASKS FOR SOMETHING IMPOSSIBLE, AND THIS TIME IT WAS CAUGHT AT DIRECTION. m2's box wants a gate on 'any bare cap.read() without a timeout argument'. cv2.VideoCapture.read is read([, image]) -> retval, image; there is no such argument, and a read timeout is a CONSTRUCTION property. Its other two clauses, .get() and .join(), are already satisfied -- no bare call exists in io/, pipeline/ or _streaming.py. Authored to the intent by human decision: gate on construction rather than on the call. The wording amendment is owed, with this evidence, exactly as box 6's was -- but found one beat earlier than last time, before any code was written against a clause no code can satisfy. REFUTED BY MUTATION, six executed: never passing params reddens 4; bounding _open_cap but leaving reconnect bare reddens 1; defaulting to 30000 instead of None reddens 1; treating 0 as absent reddens 1; accepting the parameters on a non-network source reddens 4; dropping the negative validation reddens 2. ONE OF MY CHECKS SURVIVED ITS OWN MUTATION, AND ONLY REFUTATION FOUND IT. test_a_non_network_source_does_not_silently_ignore_them asserted 'network' or 'rtsp' appears in the refusal. With the refusal DELETED, the /var/media/ case still passed -- open_source's unrelated fallback lists 'RTSP URLs (rtsp://)' among supported forms, so the check matched prose that had nothing to do with it. A check passing under the exact change it exists to catch, which is Q4. It now asserts the message NAMES the parameter, which no other error in that factory does, and all four parametrisations redden. My own -k filter hid it first: 'not silent' also deselected 'does_not_silently_ignore', so the mutation looked clean. That is Q5 in a pytest expression, in the same hour I filed Q5. THE GATE REFUSED AND WAS RIGHT. A8, E5, E6 and R:MOCKONLY read as unbound: the first three because two parametrised checks were cited by bare name -- method M12, the third instance this session -- and R:MOCKONLY because I declared the Reject and never bound any check to it, while the two real-socket checks were its answer all along. Citations are one line per parametrisation now and the real-socket checks carry R:MOCKONLY. Bound items went 9 -> 15. FOUR REGRESSIONS IN GATED NODES, ALL FOUND BY THEIR OWN CHECKS. One was MY bug: the negative-timeout raise named neither the source nor the redacted form, and test_every_raise_in_open_source_reads_the_redacted_form caught it -- the invariant is that every raise in that factory reads safe_source so a branch added later inherits redaction. Fixed in my code. The other three were correct pins firing on a contract change the human had already approved: open_source's exact signature list, an assert_called_once_with on RTSPStreamSource, and the sink node's literal mutation string, which refused to run and said so -- 'the refutation is stale, re-derive it before trusting any check here'. All three were moved deliberately and annotated with why, not weakened; I also shortened my own raise to keep it on one line so that mutation stays a simple swap rather than becoming a multi-line rewrite. Unlike the licensing-provenance case, none of these checks was wrong -- they were doing their job, and there was no option to route around them. SCOPE HELD. Retry, backoff and the reconnect deadline are untouched and a check pins that. The guard now + wait > deadline still cannot fire while wait = min(2**n, 10) and the default is 30.0, and deadline is still reset on every disconnect -- a real defect, measured, owned by rtsp-reconnect-correctness. The reconnect orphan I measured earlier -- every periodic reconnect leaks one VideoCapture, because __iter__ holds a local while reconnect rebinds the attribute -- is likewise untouched; that is reader-shutdown's box and it already names the line." }
advised_by: edge-reliability-operator
---
## CARD
goal: How long yowo waits on a wedged camera is a number yowo states, not one FFmpeg picked, and a structural check keeps it that way for captures added later.
why: Measured 2026-09-11 against an unroutable host and against a server that completes the TCP handshake and then never speaks — both the same:
  `cv2.VideoCapture(url, cv2.CAP_FFMPEG)`                      -> not opened after **30.08s**
  `cv2.VideoCapture(url, cv2.CAP_FFMPEG, [OPEN/READ = 2000ms])` -> not opened after **2.02s**
  So the stall is NOT unbounded, and the milestone's "there is no capture timeout anywhere in src/" is literally true but misleading: FFmpeg supplies a ~30s default. The defect is that 30s is FFmpeg's choice, not yowo's, and no caller can change it. Every one of the four `cap.read()` calls at `_source.py:232, 237, 338, 457` sits on a capture built with no timeout parameters at all.
  m2's box also asks for a gate on "any bare `cap.read()` ... without a timeout argument". `cv2.VideoCapture.read` is `read([, image]) -> retval, image` — there is no such argument, and a read timeout is a CONSTRUCTION property. The clause is unsatisfiable as worded, by any code. Its other two clauses are already met: no bare `.get()` or `.join()` exists in `io/`, `pipeline/` or `_streaming.py`. Authored to the intent; the wording amendment goes to the human at the gate with this evidence, the way box 6's did.
next: the constants exist (`CAP_PROP_OPEN_TIMEOUT_MSEC` = 53, `CAP_PROP_READ_TIMEOUT_MSEC` = 54, cv2 5.0.0) and the params-list constructor honours them. What is missing is a caller who passes them and a check that notices when one does not.

## RULES
<must>
- M1 Every RTSP capture is constructed with BOTH an open and a read timeout. All three construction sites — `_open_cap`, and the two inside `reconnect()` — not just the first, because the one that is missed is the one that wedges.
- M2 The timeouts are the caller's to set, carried from `open_source` through to the capture. A hardcoded constant would repeat the defect one layer up: the number would still not be the deployer's.
- M3 A wedged source RAISES within the stated bound rather than blocking past it. Measured against a real socket that accepts and never speaks, not a mock — a mock proves the argument was passed, not that FFmpeg honours it.
- M4 A structural check fails on a network capture constructed without both timeouts, so a site added later cannot quietly reintroduce the bare form. This is the clause m2's box is reaching for.
- M5 The default preserves today's behaviour unless a human chooses otherwise. Shortening it silently changes how long every existing deployment tolerates a camera blip, and that is a deployment decision, not a bugfix.
- M6 Nothing here changes retry, backoff or the reconnect deadline. `rtsp-reconnect-correctness` owns those, including the guard `now + wait > deadline` that cannot fire while `wait <= 10` and the default is `30.0`. Bounding the call is this node; deciding how many times to make it is not.
</must>
<reject>
- R:HARDCODE A timeout the caller cannot set. -> "HARDCODE"
- R:SILENTSHORTEN Changing the effective default wait without a human deciding it. -> "SILENTSHORTEN"
- R:MOCKONLY Proving the bound with a mocked `cv2` alone — that proves the argument was passed, not that it is honoured. -> "MOCKONLY"
- R:PARTIAL Bounding one construction site and leaving another bare. -> "PARTIAL"
- R:SCOPECREEP Touching retry, backoff or the reconnect deadline here. -> "SCOPECREEP"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say whose number this is; taking the DEPLOYER's, set per source at construction, because the right wait for a camera on a LAN and one over a cellular backhaul differ by an order of magnitude and no library default can serve both -> if wrong and it is a global setting, the per-source parameter is redundant but harmless.
- A2 [which] covers: S1 · the request does not say what the defaults should BE; taking 30000ms for both, which is what FFmpeg already applies, so behaviour is unchanged and only controllability is added -> if wrong, every existing deployment's tolerance for a camera blip changes on upgrade, which is R:SILENTSHORTEN. · probe: an unconfigured `RTSPStreamSource` behaves as it does today.
- A3 [when] covers: S1 · the request does not say whether the bound is per attempt or per stream lifetime; taking PER ATTEMPT, matching what the property means to FFmpeg -> if wrong, a caller reading it as a lifetime budget gets a stream that outlives the number they set.
- A4 [absent] covers: S1 · the request does not say what a `None` timeout means; taking None = "leave FFmpeg's default in place", distinct from 0 which FFmpeg reads as no-timeout, and documenting the difference -> if wrong, a caller passing None to mean "never time out" gets 30s instead. · probe: None constructs without the property; 0 is rejected or documented.
- A5 [order] covers: S1 · the request does not say whether the two properties interact; taking them as independent key/value pairs in one params list, order irrelevant, because that is how `VideoCapture`'s params argument is specified -> if wrong, one silently overrides the other.
- A6 [experience] covers: S1 · the request does not say who reads the failure; taking an operator whose camera is down at 3am — the difficulty is that a 30-second stall with no output is indistinguishable from a hung process, so they kill it and lose the diagnosis -> the raise must name the source (redacted) and the bound that was exceeded.
- A7 [who] covers: S2 · the request does not say who calls `open_source` with these; taking the same deployer, via whatever calls the factory — CLI or library — with the parameter simply passed through and not re-decided at each layer -> if wrong, the CLI needs its own flag, which is additive and does not change this contract.
- A8 [which] covers: S2 · the request does not say whether file, directory, image and webcam sources take these too; taking NETWORK SOURCES ONLY. `cv2.VideoCapture(str(path))` and `cv2.VideoCapture(device_index)` use a non-FFMPEG backend where these properties are not honoured, so accepting them there would be a parameter that silently does nothing -> if wrong, a wedged NFS-mounted file still stalls, and that needs a different mechanism than this one. · probe: passing the parameters to a file source is refused or documented as unsupported, never silently ignored.
- A9 [when] covers: S2 · the request does not say whether the factory validates the values; taking validation at the factory boundary — a negative timeout is rejected where the caller can still see their own call, not deep inside a capture -> if wrong, a typo surfaces as a hang rather than as an error.
- A10 [absent] covers: S2 · the request does not say what `open_source` does when the caller omits them entirely; taking the same default as S1, so the factory and the constructor cannot disagree -> if wrong, the same source built two ways waits two different lengths, which is the worst outcome for a thing whose whole subject is predictability.
- A11 [order] covers: S2 · n/a · `open_source` dispatches on source TYPE, and these parameters do not participate in that dispatch, so no ordering or tie-break exists for them to get wrong.
- A12 [experience] covers: S2 · the request does not say who reads the factory's signature; taking someone choosing a timeout for the first time — the difficulty is that milliseconds and seconds already coexist here (`reconnect_timeout_s` is seconds) -> so the unit is in the parameter NAME, and the docstring states the measured default rather than describing it vaguely.
- A13 [who] covers: S3 · n/a · a structural check asserts; it authorises nothing and has no actor whose permission it could misjudge.
- A14 [which] covers: S3 · the request does not say which constructions the gate covers; taking network captures in `src/yowo/io/` only, the sites where the property is honoured, so the gate cannot demand a parameter that would do nothing -> if wrong, the gate is narrower than "every capture" and says so explicitly rather than implying it. · probe: the gate names the sites it scans.
- A15 [when] covers: S3 · the request does not say whether the gate reads source text or behaviour; taking BEHAVIOUR where it can — what the capture was constructed with — because a source-text scan passes on a correctly-spelled call that is never reached, which is method M9 -> if wrong and only text is feasible, the check says which it is rather than implying the stronger one.
- A16 [absent] covers: S3 · the request does not say what the gate does about a construction it cannot classify; taking FAIL, because a gate that skips what it does not understand is a gate that the next unusual call slips past -> if wrong, an odd-but-safe construction needs an explicit, named exemption.
- A17 [order] covers: S3 · n/a · the gate inspects construction sites independently; no ordering between them changes any verdict.
- A18 [experience] covers: S3 · the request does not say who reads the gate's failure; taking whoever just added a capture — the difficulty is that "add a timeout" without the reason invites the smallest edit that silences it -> so the message carries the measured 30.08s-versus-2.02s contrast that motivates the rule.

## PLAN
contract: `RTSPStreamSource.__init__` and `open_source` gain keyword-only `open_timeout_ms` and `read_timeout_ms`. `_open_cap` becomes the single place a capture is built, and `reconnect()`'s two constructions route through it, so there is one site to keep correct instead of three. The params-list form `cv2.VideoCapture(url, cv2.CAP_FFMPEG, [CAP_PROP_OPEN_TIMEOUT_MSEC, ms, CAP_PROP_READ_TIMEOUT_MSEC, ms])` is used, because `cap.set()` after construction is too late to bound the open. A structural check asserts every network capture in `src/yowo/io/` is built through that one path.
strategy: prove the bound against a REAL socket first — a listener that accepts and never speaks — since that is the only evidence FFmpeg honours the property. Mock-based checks come second and cover the wiring, not the guarantee.
regression floor: the full unit suite, plus the existing RTSP and redaction checks that already drive `_open_cap` and `reconnect()`.

## EDGES
- E1 an unroutable host (TEST-NET-1 `192.0.2.1`) — raises within the stated bound, not at 30s.
- E2 a server that completes the TCP handshake and then never speaks — the case a connectivity check would call healthy.
- E3 `reconnect()`'s two construction sites — both bounded, or R:PARTIAL.
- E4 an unconfigured source — behaves exactly as today (A2, R:SILENTSHORTEN).
- E5 a file, directory, image or webcam source — the parameters are not silently accepted and ignored (A8).
- E6 a negative or nonsensical timeout — rejected at the factory, where the caller can still see their own call.
- E7 the reconnect deadline and backoff — untouched by this node (M6, R:SCOPECREEP).

## CHECKS
The real-socket checks carry the guarantee; the mocked ones carry the wiring. Both are cited, and
which is which is stated, so a reader cannot mistake the second kind for the first.

- tests.unit.test_capture_timeouts::test_a_wedged_host_raises_within_the_stated_bound · covers: M3, E1, R:MOCKONLY · a real socket, not a mock — the only evidence FFmpeg honours the property.
- tests.unit.test_capture_timeouts::test_a_silent_server_raises_within_the_stated_bound · covers: M3, E2, R:MOCKONLY · accepts the connection then says nothing; a port check would call this healthy.
- tests.unit.test_capture_timeouts::test_the_open_timeout_is_actually_shorter_than_the_default · covers: M3, A2 · 2s configured against ~30s unconfigured, measured in the check rather than asserted.
- tests.unit.test_capture_timeouts::test_every_network_capture_is_built_with_both_timeouts · covers: M1, M4, S3, A14 · the structural gate; the clause m2's box is reaching for.
- tests.unit.test_capture_timeouts::test_reconnect_builds_its_capture_the_same_way · covers: M1, E3, R:PARTIAL · the site most likely to be missed, because it is not the obvious one.
- tests.unit.test_capture_timeouts::test_the_timeouts_reach_the_capture_from_open_source · covers: M2, S2, R:HARDCODE · carried from the factory, not decided inside.
- tests.unit.test_capture_timeouts::test_an_unconfigured_source_is_unchanged · covers: M5, A2, E4, R:SILENTSHORTEN · upgrading must not shorten anyone's tolerance silently.
- tests.unit.test_capture_timeouts::test_none_leaves_the_backend_default_in_place · covers: A4 · None and 0 mean different things and the difference is documented.
- tests.unit.test_capture_timeouts::test_a_non_network_source_does_not_silently_ignore_them[/var/media/clip.mp4] · covers: A8, E5 · a parameter that does nothing is worse than one that is absent.
- tests.unit.test_capture_timeouts::test_a_non_network_source_does_not_silently_ignore_them[/var/media/] · covers: A8, E5 · a parameter that does nothing is worse than one that is absent.
- tests.unit.test_capture_timeouts::test_a_non_network_source_does_not_silently_ignore_them[0] · covers: A8, E5 · a parameter that does nothing is worse than one that is absent.
- tests.unit.test_capture_timeouts::test_a_non_network_source_does_not_silently_ignore_them[photo.jpg] · covers: A8, E5 · a parameter that does nothing is worse than one that is absent.
- tests.unit.test_capture_timeouts::test_a_negative_timeout_is_rejected_at_the_factory[-1] · covers: A9, E6 · a typo must not surface as a hang.
- tests.unit.test_capture_timeouts::test_a_negative_timeout_is_rejected_at_the_factory[-5000] · covers: A9, E6 · a typo must not surface as a hang.
- tests.unit.test_capture_timeouts::test_the_raise_names_the_source_and_the_bound · covers: A6 · a redacted identifier and the number that was exceeded, or the operator cannot act at 3am.
- tests.unit.test_capture_timeouts::test_the_gate_failure_explains_why · covers: A18 · the message carries the 30.08s-versus-2.02s contrast, so the fix is not "silence it".
- tests.unit.test_capture_timeouts::test_the_gate_fails_a_construction_it_cannot_classify · covers: A16 · a gate that skips the unfamiliar is a gate the next unusual call walks past.
- tests.unit.test_capture_timeouts::test_retry_and_backoff_are_untouched · covers: M6, E7, R:SCOPECREEP · the reconnect deadline stays exactly as wrong as it was; another node owns it.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
