---
type: Task
title: Weight provenance, AGPL disclosure, SECURITY.md, project URLs
status: direction
depth: standard
milestone: m1-trust-the-ship
scope:
  - README.md
  - CONTRIBUTING.md
  - SECURITY.md
  - pyproject.toml
gives:
  - S1 the README's licence and weight-provenance section — what a user is told before first download
  - S2 SECURITY.md — how a vulnerability is reported and what is in scope
  - S3 pyproject.toml's declared licence metadata and [project.urls] — what PyPI shows
depends_on:
  - /tasks/sdist-manifest.md
  - /tasks/weight-integrity.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: docs-writer
---
## CARD
goal: Nobody acquires an AGPL-licensed artifact through yowo without being told, and a vulnerability has somewhere to be reported.
why: verified facts, not inference —
  - `pyproject.toml:6` declares `license = "Apache-2.0"`.
  - Every weight URL points into `ultralytics/assets`, which the GitHub licence API reports as
    **AGPL-3.0** (`ultralytics/ultralytics` likewise).
  - README mentions a licence twice and **AGPL zero times**. `README.md:137` reads only "Weights are
    downloaded automatically to `~/.cache/yowo/weights/` on first use."
  So `yowo detect image.jpg` silently places an AGPL-3.0 artifact on a user's disk, from a package whose
  metadata says Apache-2.0, with no disclosure anywhere.
  - There is no SECURITY.md, no NOTICE, no THIRD_PARTY.md, and no `[project.urls]` at all — PyPI shows
    no homepage, repository, issue tracker or changelog link.
  `sdist-manifest` fixed the shipped bytes: the tarball now contains only yowo's own Apache-2.0 code.
  This task fixes what is *told*, which is the part that survived.

BOUNDARY — this task documents facts; it does not give legal advice, and must not pretend to.
  Two questions are genuinely legal and are for the human (and, if they choose, counsel) to answer:
  (a) whether downloading AGPL weights at runtime creates obligations for a user deploying yowo,
      particularly AGPL §13's network-use clause;
  (b) whether `src/yowo/arch/`, re-implemented natively but validated to tensor-equivalence against
      ultralytics' implementation (`tmp/compare_arch.py`), is a derivative work.
  I can state what is verifiable and make the disclosure prominent. I cannot resolve (a) or (b), and a
  README that implied I had would be worse than the present silence.
beat: scaffold · next: author licensing-provenance's RULES, ASSUMPTIONS and CHECKS, then add freeze licensing-provenance

## RULES
<must>
- M1 A user is told, before the first weight download can surprise them, that the default weights are
  AGPL-3.0 and that yowo's own code is Apache-2.0.
- M2 The disclosure states verifiable facts and asserts no legal conclusion in either direction.
- M3 SECURITY.md exists and says where to report, what is in scope, and what response to expect.
- M4 `[project.urls]` exists so PyPI shows a homepage, repository, issue tracker and changelog.
- M5 A check fails the build if the declared licence and the documented weight provenance drift apart.
</must>
<reject>
- R:IMPLIEDADVICE No document may state or imply a conclusion about the reader's own AGPL obligations
  -> "IMPLIEDADVICE"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request does not say who the disclosure is for; taking: someone deciding
  whether to deploy yowo commercially, who needs the facts BEFORE integrating rather than after
  -> a disclosure written for a casual reader buries the one fact that changes a deployment decision.
- A2 [which] covers: S1 · the request does not say which facts appear; taking: yowo's licence, the
  weights' licence, the URL they come from, and that downloading is the USER's acquisition — no more
  · probe: the section names AGPL-3.0 and ultralytics/assets explicitly
  -> naming fewer facts leaves the reader unable to check the claim themselves.
- A3 [when] covers: S1 · the request does not say whether this applies to `--weights` local files;
  taking: no — a user supplying their own weights acquires nothing from us, and the text says so
  -> a blanket warning trains people to ignore it, including when it matters.
- A4 [absent] covers: S1 · the request does not say what happens for a user-registered custom model;
  taking: the disclosure covers the BUILT-IN defaults only, and says so
  -> claiming to know the licence of someone's private bucket is a false statement.
- A5 [order] covers: S1 · the request does not say where the section sits; taking: near the install and
  first-run instructions, not at the bottom -> a licence note below the API reference is not disclosure.
- A6 [experience] covers: S1 · the request does not say what the reader does next; taking: they are
  pointed at the upstream licence and told to seek their own advice, because the alternative is us
  answering a legal question we are not competent to answer (R:IMPLIEDADVICE)
  -> false confidence in a specific reading is worse than the present silence.
- A7 [who] covers: S2 · the request does not say who receives reports; taking: a private channel to the
  maintainer (GitHub private vulnerability reporting), not a public issue
  -> "open an issue" for a vulnerability is a disclosure policy that publishes the vulnerability.
- A8 [which] covers: S2 · the request does not say what is in scope; taking: the library and its release
  pipeline — explicitly NOT the AGPL weights, which are upstream's
  -> reports about ultralytics' models arriving here and going nowhere.
- A9 [when] covers: S2 · the request does not say what response time is promised; taking: a stated
  best-effort acknowledgement window and no fixed remediation SLA, because this is an unfunded project
  · probe: no SLA is promised that cannot be met -> a missed published SLA is worse than none.
- A10 [absent] covers: S2 · the request does not say what happens to an out-of-scope report; taking: the
  document says it will be redirected, not silently closed -> a reporter who is ignored discloses publicly.
- A11 [order] n/a · a policy document; no ordering is constrained.
- A12 [experience] covers: S2 · the request does not say who reads it; taking: a security researcher
  scanning for a contact in under a minute, so the contact route is the first thing on the page
  -> a policy whose contact is buried gets bypassed for a public issue.
- A13 [who] n/a · package metadata has no actor surface.
- A14 [which] covers: S3 · the request does not say which URLs; taking: Homepage, Repository, Issues and
  Changelog — the four PyPI renders as links -> a PyPI page with no repository link reads as abandoned.
- A15 [when] n/a · static metadata.
- A16 [absent] covers: S3 · the request does not say what if a URL 404s; taking: a check asserts each is
  well-formed and points at this repository, not that it resolves — a unit test must not require the
  network -> a link check that needs the network is a flaky check that gets deleted.
- A17 [order] n/a · a mapping.
- A18 [experience] covers: S3 · the request does not say who reads PyPI metadata; taking: someone
  evaluating whether the project is maintained, for whom a changelog link is the strongest signal
  -> evaluation stalls on "is this alive" and the answer is one click away.

## PLAN
contract: a README section near first-run; SECURITY.md; `[project.urls]`; and a check binding the
  declared licence to the documented provenance so they cannot drift.
strategy: state facts, cite where each is checkable, assert no conclusion. The legal questions recorded
  in the CARD stay open and are NOT answered in any published document.
regression floor: `test_sdist_manifest.py` stays green — the allowlist must still ship LICENSE and README.

## EDGES
- E1 A user supplying `--weights` acquires nothing from yowo, and the text must not imply otherwise.
- E2 The disclosure must not become stale if the default weight source changes — the check binds it to
  the registry's actual URLs, not to a hardcoded sentence.

## CHECKS
- test_readme_discloses_the_weight_licence · covers: M1, A2 · names AGPL-3.0 and ultralytics/assets.
- test_disclosure_asserts_no_legal_conclusion · covers: M2, R:IMPLIEDADVICE, A6 · no "you must"/"you are
  required to"/"is permitted" phrasing about the reader's obligations.
- test_disclosure_scopes_itself_to_builtin_defaults · covers: A3, A4, E1 · local and custom weights excluded.
- test_security_md_exists_with_a_private_report_route · covers: M3, A7, A12 · contact route present and first.
- test_security_md_scopes_itself_and_promises_no_unmeetable_sla · covers: A8, A9, A10 · scope stated, no SLA.
- test_project_urls_are_declared_and_well_formed · covers: M4, A14, A16 · four keys, all this repository.
- test_disclosure_matches_the_registry_urls · covers: M5, E2 · bound to real URLs, not a fixed sentence.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
