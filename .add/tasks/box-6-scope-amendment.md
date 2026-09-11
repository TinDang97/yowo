---
type: Task
title: Box 6 says what the code guarantees, and path-borne secrets are a named operator constraint
status: direction
depth: deep
sensitivity: security
milestone: m1-trust-the-ship
scope:
  - .add/milestones/
  - SECURITY.md
  - tests/unit/
gives:
  - S1 m1 box 6 — the amended guarantee and its residual-risk clause
  - S2 SECURITY.md § credentials in a source URL — the path-borne operator constraint
  - S3 the box-6 witness pair — what still fails once the box no longer says "no credential"
generated: { by: add/3.5.0, at: 2026-09-11 }
verified:
  - { by: "Tin Dang", at: 2026-09-11, act: interview, authority: human, interview: "sha256:14006a929714996c", receipt: /tasks/box-6-scope-amendment.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A12=confirm|A14=confirm|A15=confirm|A16=confirm|A18=confirm|A19=confirm|R:FICTION=confirm|R:PATHREDACT=confirm|R:UNWITNESSED=confirm|R:SILENTWIDEN=confirm" }
  - { by: "Tin Dang", at: 2026-09-11, act: freeze, authority: human, direction: "sha256:ec1f87636b3b4643", binding: "sha256:d2b42533a0429356" }
advised_by: security-reviewer
---
## CARD
goal: m1 box 6 states the guarantee the code actually makes — userinfo, query and fragment — with path-borne secrets named in the box as an operator constraint and documented in SECURITY.md, so the box can be ticked without asserting something a one-line probe refutes.
why: Box 6 as worded is unsatisfiable by any code change. Measured at f9784fc, `rtsp://h/live/S3CR3T-signed/stream` comes back unchanged, and redaction cannot fix that one — the path IS the camera's identity, so redacting it collapses every camera on a host onto one id, which is the silent cross-delivery bug `credential-sinks-are-bound` fixed at the cause. Three tasks behind this box are done and gated; there is no build work left. Leaving the box open holds m1 hostage against a threat model code cannot address — `milestone-done` refuses an unchecked box — and ticking it as worded makes the milestone assert a guarantee its own suite disproves. Box 2 was amended this way on 2026-09-08, RESIDUAL RISK named inline; this follows that precedent rather than inventing an act.

## RULES
<must>
- M1 Box 6 names the three URL components redaction covers — userinfo, query, fragment — and claims no others. It enumerates; it does not generalise to "URL credentials", because a general phrase re-opens the same unsatisfiability.
- M2 The path-borne exception is stated IN the box, not only in a receipt or a linked task. A reader deciding whether yowo is safe for their cameras sees the limit without opening anything.
- M3 The residual-risk clause says WHY it cannot be fixed — the path is the camera's identity — so a future maintainer does not read it as an open TODO and "fix" it into an identity collapse.
- M4 SECURITY.md documents the operator constraint, at the same commit that amends the box. The box claims the documentation exists; that claim must be true when it is made, not eventually.
- M5 Both box-6 witnesses are re-derived WHOLE — assertion and prose. `test_box_6_is_not_ticked`'s docstring still states that `redact_url` preserves the query string and shows two transformations that f9784fc disproved; a witness whose prose asserts a fiction is precisely the defect this node exists to stop repeating.
- M6 A witness survives the amendment. Something still fails if the path-borne form stops being the named exception — because path redaction was added, or because the residual-risk clause was deleted from the box. The witness does NOT assert the tick state: the tick is an `add check` act after this node's gate, so a check demanding it would be red for the whole build and green only by accident of ordering.
- M7 No redaction behaviour changes. `redact_url` and `safe_stream_id` are not touched. This node moves a contract line and a document, and proves the code already matches what the line now says.
</must>
<reject>
- R:FICTION Box 6, or any witness prose, asserting something a probe refutes. -> "FICTION"
- R:PATHREDACT Redacting the URL path — here, or as the follow-up an open-sounding box invites. -> "PATHREDACT"
- R:UNWITNESSED Amending the box while nothing would fail if the residual-risk clause were deleted. -> "UNWITNESSED"
- R:SILENTWIDEN Amending the box to claim more than the checks establish. -> "SILENTWIDEN"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who may amend a milestone box; taking the human maintainer, via `add interview` then `add check --by`, since the box is a milestone contract and box 2's precedent is stamped "by human decision" -> if wrong and process authority sufficed, this node costs one interview round, which is nothing.
- A2 [which] covers: S1 · the request does not say which credential-bearing components the amended box claims; taking exactly the three that redaction covers — userinfo, query, fragment — and no wider phrase -> if wrong the box under-claims relative to the code. · probe: each of the three forms, driven through all four sinks, comes back clean.
- A3 [when] covers: S1 · the request does not say when the box may be ticked; taking at this node's gate, once SECURITY.md has landed in the same commit, because the box asserts that documentation exists -> if wrong and the tick belongs at m1's close, the box sits ticked against a true statement for a few days, which costs nothing.
- A4 [absent] covers: S1 · the request does not say what the box claims about a credential form nobody has named yet — a future scheme's userinfo-equivalent, a header-borne token; taking silence as NOT CLAIMED, the box enumerating rather than generalising -> if wrong, a reader takes the enumeration as exhaustive of all credential risk in yowo.
- A5 [order] covers: S1 · n/a · a checklist box is a single line with no ordered elements a reader could get wrong; the AMENDED-then-RESIDUAL-RISK clause order copies box 2 verbatim and carries no semantics of its own.
- A6 [experience] covers: S1 · the request does not say who reads the box; taking an operator or auditor reading m1 to decide whether yowo is safe to deploy with signed-URL cameras — the difficulty is that "no credential" reads as total and a ticked box ENDS their reading -> so the limit must sit in the box body, where the tick cannot outrun it.
- A7 [who] covers: S2 · the request does not say whose action the constraint governs; taking the deploying operator who composes the camera URL, not the yowo developer, since nothing a developer writes can fix it -> if wrong and it is a developer note, it belongs in a docstring and SECURITY.md is the wrong file.
- A8 [which] covers: S2 · the request does not say which URL forms the doc must show; taking the leaking form AND a safe rewrite of it, because a constraint stated without a remedy is a complaint -> if wrong the section is longer than it needs to be. · probe: the section contains a path-borne example and a query-borne alternative.
- A9 [when] covers: S2 · the request does not say whether the constraint reaches non-URL sources; taking URL sources only — a file path, a directory and a webcam index are returned byte-identical and carry no credential yowo could redact -> if wrong, an operator believes a local media path is implicated and hunts a risk that is not there.
- A10 [absent] covers: S2 · the request does not say what an operator does when the path-borne token is unavoidable, as it is on cameras that hard-code it; taking an explicit fallback — the stream id IS then the secret, so logs and the feature cache must be treated as secret-bearing — rather than silence -> if wrong, the operator with no choice reads the section, finds no case for themselves, and assumes they are safe.
- A11 [order] covers: S2 · n/a · a prose section has no ordering semantics; nothing in it is a sequence a reader could apply out of order.
- A12 [experience] covers: S2 · the request does not say who reads SECURITY.md; taking someone who arrived from box 6's wording looking for exactly this constraint — the difficulty is landing in a file about vulnerability reporting and supported versions and not finding it -> so it needs its own heading, findable by the words the box uses.
- A13 [who] covers: S3 · n/a · a check asserts and authorises nothing; there is no actor whose permission it could get wrong.
- A14 [which] covers: S3 · the request does not say which of the two witnesses carries the re-derivation; taking BOTH, each re-deriving its reason from the running code, because the drift this node fixes was caused by exactly one copy being updated and the other left standing -> if wrong, one witness is redundant, at the cost of a few milliseconds of suite time.
- A15 [when] covers: S3 · the request does not say what the witnesses assert once the box no longer says "no credential"; taking a flip from "the box is unticked" to "the box carries the residual-risk clause AND the path form is still the exception", which is true the moment the text is amended and does not depend on the separate `add check` act -> if wrong and they should simply be deleted, the amendment becomes unwitnessed, which is R:UNWITNESSED.
- A16 [absent] covers: S3 · the request does not say what happens if someone later adds path redaction; taking the witness goes RED, because that change collapses two cameras on one host onto one id -> if wrong, the collapse ships silently and detections cross-deliver. · probe: a stub that redacts the path turns a witness red.
- A17 [order] covers: S3 · n/a · the two witnesses are independent; neither reads state the other writes, so no execution order can change either outcome.
- A18 [experience] covers: S3 · the request does not say who reads a witness failure; taking a future maintainer who has just "fixed" path redaction and needs to learn from the failure message that identity collapse is the reason -> so the message states the cross-delivery consequence, not merely that a string changed.
- A19 [which] covers: S3 · the request does not say whether the witnesses keep names that read `is_not_ticked` once the box is ticked; taking KEPT, because `credential-sinks-are-bound` and `query-string-credentials` are both closed and their CHECKS cite those exact names, and a rename dangles a closed node's citation (method M12) -> if wrong, two names read historically and a reader must open them to see what they now guard.

## PLAN
contract: `.add/milestones/m1-trust-the-ship.md` box 6 is rewritten to name userinfo, query and fragment, carrying an `AMENDED 2026-09-11 ... by human decision` clause and a `RESIDUAL RISK, named rather than hidden:` clause in box 2's format, and citing all three tasks that fed it. `SECURITY.md` gains a section on credentials in a source URL: what redaction covers, the path-borne form it cannot cover, why, a safe rewrite, and the fallback when the token is unavoidable. The two existing box-6 witnesses flip from asserting the box is unticked to asserting the exception is still the exception, prose re-derived with them. A new `tests/unit/test_box_6_scope.py` owns the claims this node makes that neither older node can witness. `src/` is untouched.
strategy: amend the box first so the witnesses have something true to read, then SECURITY.md, then the witnesses last — each re-derived against the running code rather than against the box text, so a wrong box cannot make a witness agree with it.
regression floor: the full unit suite, which already contains every sink check the three closed nodes bound.

## EDGES
- E1 `rtsp://h/live/S3CR3T-signed/stream` — unchanged through both boundaries. The named exception, re-derived, never asserted from memory.
- E2 the three covered forms — userinfo, query, fragment — clean at all four sinks. The box's positive claim, proved rather than restated.
- E3 the residual-risk clause deleted from the box — a witness reddens (R:UNWITNESSED).
- E4 path redaction added to `redact_url` — a witness reddens (R:PATHREDACT), because two cameras on one host collapse onto one id.
- E5 SECURITY.md missing the constraint while the box claims it is documented — a check reddens (M4).
- E6 `test_box_6_is_not_ticked`'s docstring — no sentence left in it that a probe refutes (M5, R:FICTION).

## CHECKS
The three claims older nodes cannot witness live in a new file this node owns. The two flips land in
the files that already hold them, under their existing names (A19), so no closed node's citation dangles.

- tests.unit.test_box_6_scope::test_the_box_names_the_three_covered_components · covers: M1, A2 · enumerated, not generalised — a wider phrase re-opens the unsatisfiability.
- tests.unit.test_box_6_scope::test_each_named_component_is_actually_clean_at_every_sink[userinfo] · covers: M1, A2, E2 · the box's positive claim, driven rather than restated.
- tests.unit.test_box_6_scope::test_each_named_component_is_actually_clean_at_every_sink[query] · covers: M1, A2, E2 · the box's positive claim, driven rather than restated.
- tests.unit.test_box_6_scope::test_each_named_component_is_actually_clean_at_every_sink[fragment] · covers: M1, A2, E2 · the box's positive claim, driven rather than restated.
- tests.unit.test_box_6_scope::test_the_box_claims_no_component_the_checks_do_not_establish · covers: R:SILENTWIDEN · the box may not out-run its own evidence.
- tests.unit.test_box_6_scope::test_the_residual_risk_is_in_the_box_body · covers: M2, A6 · a ticked box ends the reading, so the limit cannot live in a footnote.
- tests.unit.test_box_6_scope::test_the_residual_risk_says_why_it_cannot_be_fixed · covers: M3 · without the reason it reads as an open TODO and invites R:PATHREDACT.
- tests.unit.test_box_6_scope::test_the_box_claims_nothing_a_probe_refutes · covers: R:FICTION, E6 · every transformation the box or either witness states is executed and compared.
- tests.unit.test_box_6_scope::test_security_md_documents_the_operator_constraint · covers: M4, E5 · the box says "documented"; this is that word being true.
- tests.unit.test_box_6_scope::test_security_md_has_its_own_findable_heading · covers: A12 · arrived from box 6, must not have to read a file about vulnerability reporting to find it.
- tests.unit.test_box_6_scope::test_security_md_shows_a_leaking_form_and_a_safe_rewrite · covers: A8 · a constraint without a remedy is a complaint.
- tests.unit.test_box_6_scope::test_security_md_says_what_to_do_when_the_path_token_is_unavoidable · covers: A10 · the operator with no choice must not read past their own case.
- tests.unit.test_box_6_scope::test_security_md_does_not_implicate_non_url_sources · covers: A9 · a file, a directory and a webcam index carry nothing yowo could redact.
- tests.unit.test_box_6_scope::test_no_redaction_behaviour_changed · covers: M7 · this node moves a contract line and a document; `redact_url` and `safe_stream_id` are byte-identical to f9784fc.
- tests.unit.test_credential_sinks::test_box_6_is_not_ticked · covers: M5, M6, R:UNWITNESSED, E1, E3, E6 · prose and assertion re-derived together; reddens if the clause goes or the exception stops being the exception.
- tests.unit.test_query_credentials::test_box_6_is_not_ticked_until_this_is_green · covers: M5, M6, R:UNWITNESSED, E1, E3 · the second witness, re-derived independently against the running code.
- tests.unit.test_box_6_scope::test_adding_path_redaction_collapses_two_cameras · covers: R:PATHREDACT, A16, E4 · the reason the exception exists, executed — two hosts, one id.
- tests.unit.test_box_6_scope::test_a_witness_failure_names_the_cross_delivery_consequence · covers: A18 · the maintainer who just "fixed" it learns why from the message.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
