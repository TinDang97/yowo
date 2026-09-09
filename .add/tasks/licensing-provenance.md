---
type: Task
title: Weight provenance, AGPL disclosure, SECURITY.md, project URLs
status: direction
depth: standard
sensitivity: legal
milestone: m1-trust-the-ship
scope:
  - LICENSE
  - NOTICE
  - README.md
  - CONTRIBUTING.md
  - SECURITY.md
  - pyproject.toml
  - scripts/verify_sdist_contents.py
gives:
  - S1 `LICENSE` and `NOTICE` — the copyright grant and the third-party weight provenance
  - S2 `README.md` §Model weights and licensing — the AGPL statement where an adopter looks
  - S3 `SECURITY.md` — the reporting channel and the trust boundary m1 built
  - S4 `pyproject.toml` `[project.urls]` — the metadata PyPI renders
depends_on:
  - /tasks/sdist-manifest.md
  - /tasks/weight-integrity.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-09, act: interview, authority: human, interview: "sha256:fe68556923c0b6d1", receipt: /tasks/licensing-provenance.d/interviews/1.md, answers: "A1=confirm|A2=confirm|A3=confirm|A4=confirm|A5=confirm|A6=confirm|A7=confirm|A8=confirm|A9=confirm|A10=confirm|A11=confirm|A12=confirm|A13=confirm|A14=confirm|A15=confirm|A16=confirm|A17=confirm|A18=confirm|A19=confirm|R:OVERCLAIM=confirm|R:UNSHIPPED=confirm" }
  - { by: "Tin Dang", at: 2026-09-09, act: freeze, authority: human, direction: "sha256:1c57db0a27fbb11e", binding: "sha256:e9a79d98e3503d91" }
advised_by: docs-writer
---
## CARD
goal: Every shipped byte declares its licence, the AGPL-3.0 provenance of the default weights is
  documented where an adopter will actually find it, and SECURITY.md exists.
why: yowo 2.4.0 and 2.4.1 shipped a 5.6 MB AGPL-3.0 `yolo11n.pt` to PyPI inside an sdist declaring
  `License-Expression: Apache-2.0`. Both are yanked and `sdist-manifest` closed the mechanism, but the
  DECLARATIONS around it were never corrected. Verified against the tree on 2026-09-09:
  - `LICENSE:189` still reads `Copyright [yyyy] [name of copyright owner]` — the Apache-2.0 template
    placeholder, unfilled. Apache-2.0 4(c)/(d) presume attribution; a licensor naming nobody weakens
    the grant and fails most corporate licence scanners.
  - There is no `NOTICE` file, so third-party weight provenance is recorded nowhere.
  - There is no `SECURITY.md`, which the m1 goal box requires.
  - `README.md:801-803` is the entire licence statement: "Apache-2.0 — see LICENSE". No mention of
    weights, of ultralytics, or of AGPL anywhere in the file.
  - `pyproject.toml` has no `[project.urls]` at all, so PyPI shows no homepage, source or issues link.
  - `CONTRIBUTING.md:28` claims yowo is "intentionally ultralytics-free (Apache-2.0 clean)". True of
    the code; the default weights are ultralytics AGPL-3.0 artifacts fetched at runtime, so the
    sentence reads broader than it is.
  - `.gitignore` carried no `*.pt` rule, so the stray 5.6 MB AGPL `yolo11n.pt` at the repo root — the
    same artifact that shipped — was one `git add -A` from re-entering an Apache-2.0 tree. Ignored
    2026-09-09 ahead of this node, as immediate hazard reduction; the rest of the scope is untouched.

  The code IS Apache-2.0 clean and the package now redistributes no weights. The work here is making
  the paperwork say what is already true, in the places a legal review looks.

BLOCKED on one human decision, deliberately not guessed: the copyright holder for `LICENSE:189` and
  `NOTICE`. Git authorship says `Tin Dang <tindang.ht97@gmail.com>`; the account email is
  `vinacapital@trustifytechnology.com`, which raises whether an employer holds the copyright. That is
  a legal identity, it is the one item here that is hard to unwind once published, and no evidence in
  the repository settles it. Three further readings were put to a human at the same time and are also
  unanswered: how far the AGPL boundary goes (document / also warn at first download / also gate
  behind explicit consent), how much of the m1 trust boundary SECURITY.md documents, and whether the
  misleading v2.5.0 GitHub release note and the missing project URLs fall inside this node.
beat: scaffold · next: answer the interview, then author licensing-provenance's RULES, ASSUMPTIONS and CHECKS

## RULES
<must>
- M1 `LICENSE` names a real copyright holder and year. No Apache-2.0 template placeholder survives.
- M2 `NOTICE` exists, ships in every published artifact, and records the third-party provenance of the
  default weights: ultralytics, AGPL-3.0, fetched at runtime, never redistributed by this package.
- M3 `README.md` states four things plainly: the code is Apache-2.0; the default weights download at
  runtime from ultralytics and are AGPL-3.0; this package redistributes no weights; an adopter who
  cannot accept AGPL supplies their own.
- M4 `SECURITY.md` exists, names the supported versions and a private reporting channel, and documents
  the checkpoint trust boundary m1 built.
- M5 No document in the repository states or implies that the weights are Apache-2.0, or that yowo is
  "Apache-2.0 clean" without the runtime-weight caveat.
- M6 `pyproject.toml` declares `[project.urls]`.
- M7 The sdist still passes `scripts/verify_sdist_contents.py`, with `NOTICE` admitted by naming it —
  never by relaxing the allowlist.
</must>
<reject>
- R:OVERCLAIM No document may state a licence position broader than what the runtime actually does -> "OVERCLAIM"
- R:UNSHIPPED No file the grant requires a redistribution to carry may be absent from a published artifact -> "UNSHIPPED"
</reject>

## ASSUMPTIONS
- A1 [who] covers: S1 · the request did not say who holds the copyright; human decision 2026-09-09:
  `Tin Dang`, a personal project with no employer claim, year 2026 -> naming an employer that has no
  claim, or nobody at all, is the one item here that cannot be quietly unwound after publication.
- A2 [which] covers: S1 · the request does not say which third parties `NOTICE` records; taking: only
  the default model weights. Every Python dependency carries its own licence and none is redistributed
  by this package -> enumerating every transitive dependency produces a file nobody maintains, nobody
  reads, and that is wrong the first time a lockfile moves.
- A3 [when] covers: S1, S2 · the request does not say where the code/weight boundary falls; taking: the
  code is Apache-2.0, and the weights are never redistributed — they are fetched at runtime into
  `~/.cache/yowo/weights/`, so AGPL's network-use copyleft attaches to the ADOPTER's use of the
  weights, not to this package -> telling an adopter their deployment is AGPL-free is legal advice we
  are not entitled to give, and R:OVERCLAIM exists for exactly that sentence.
- A4 [absent] covers: S2 · the request does not say what an adopter who cannot accept AGPL does;
  taking: they supply their own weights via `--weights` / `weights_path`, which the loader already
  supports and `resolve_weights` already treats as unpinned -> a licensing note that states a problem
  without naming the escape hatch reads as a warning to leave.
- A5 [order] covers: S2 · the request does not say where the section sits; taking: its own top-level
  `## Model weights and licensing`, not a footnote under the two-line `## License` at line 801
  -> a caveat buried at the bottom is one a legal review finds only after adopting.
- A6 [experience] covers: S2, S3 · the request does not say who reads these; taking: an adopter's legal
  or security review arriving from PyPI with no prior context, so each states its position in the
  first paragraph rather than building to it -> a document that must be read in full to learn its
  conclusion is one that gets skimmed to the wrong conclusion.
- A7 [who] covers: S3 · the request does not say who receives a vulnerability report; taking: private
  reporting through this repository's GitHub Security Advisories, not an email address -> an email
  address on a personal project is a channel that silently rots, and a rotted channel is worse than
  none because it looks live.
- A8 [which] covers: S3 · the request does not say how much of the boundary `SECURITY.md` documents;
  taking: what m1 actually shipped and can be pointed at — checkpoints are untrusted input read
  through a restricted unpickler, weights are digest-pinned and re-verified before conversion, a
  credentialed URL is redacted on emit — and NOT an enumerated threat model · probe: every claim in
  the file names code that exists -> a threat model commits to maintenance as the boundary moves, and
  two of its items (`export-digest-threading`, `storage-suffix-enumeration`) are still open.
- A9 [when] covers: S3 · the request does not say which versions are supported; taking: the current
  minor line only, with no LTS promise -> promising support nobody will backport to is a claim that
  fails the first time it is tested.
- A10 [absent] covers: S1, S4 · the request does not say what a missing `NOTICE` in an artifact means;
  taking: the build FAILS `verify_sdist_contents.py` rather than shipping — `NOTICE` joins
  `REQUIRED_ENTRIES`, not merely the allowed set -> Apache-2.0 4(d) requires a redistribution to carry
  the NOTICE, so an absent one is a defect in the grant, not a cosmetic omission (R:UNSHIPPED).
- A11 [order] covers: S1 · the request does not say what happens to `sdist-manifest`'s frozen
  allowlist; taking: `NOTICE` is added to `only-include` AND to the script's required set in the same
  change, with the reason recorded — the script's own comment anticipates this ("adding it back is a
  one-line change to `only-include`") -> relaxing an allowlist to admit a new file, rather than naming
  it, is how an allowlist stops meaning anything.
- A12 [experience] covers: S1 · the request does not say who reads `NOTICE`; taking: a licence scanner
  and a human doing third-party review, so it names the upstream project, the licence, the URL and the
  fact of non-redistribution in four lines rather than prose -> a NOTICE that argues a position instead
  of stating facts is one a scanner cannot parse and a reviewer does not trust.
- A13 [which] covers: S4 · the request does not say which URLs; taking: Homepage, Source, Issues and
  Changelog — the four PyPI renders -> a package page with no source link is itself the licence-review
  smell this node exists to remove.
- A14 [who] covers: S2, S4 n/a · README text and package metadata have no actor and no authorization
  surface.
- A15 [which] covers: S2 · the request does not say which weights the disclosure covers; taking: every
  built-in registry entry, detection, `-cls` and `-obb` alike — all resolve to ultralytics assets
  -> disclosing only the detection weights would be true of the pinned ten and false of the other
  fifteen.
- A16 [when] covers: S4 n/a · project metadata has no boundary in time; it is read at publication.
- A17 [order] covers: S3, S4 n/a · neither a reporting policy nor a URL table has an ordering the
  reader depends on.
- A18 [absent] covers: S3 · the request does not say what an unsupported-version report gets; taking:
  it is still received and triaged, but the fix lands on the current line -> a policy that refuses to
  hear a report about an old version teaches people to post it publicly instead.
- A19 [experience] covers: S4 · the request does not say who reads project URLs; taking: someone on
  the PyPI page deciding whether the project is real, so Source points at the repository rather than a
  docs mirror -> the question a URL table answers is "can I see the code", and any other first link
  answers a question nobody asked.

## PLAN
contract:
  - S1 `LICENSE` line 189 becomes `Copyright 2026 Tin Dang`; new `NOTICE` records the ultralytics
    AGPL-3.0 weight provenance and the fact of non-redistribution; both ship — `NOTICE` added to
    `only-include` and to `verify_sdist_contents.py`'s required set (A10, A11).
  - S2 `README.md` gains `## Model weights and licensing` carrying M3's four statements, and the
    existing `## License` points at it. `CONTRIBUTING.md:28`'s "Apache-2.0 clean" is qualified to say
    what it means — the code carries no ultralytics dependency — rather than being deleted (M5).
  - S3 `SECURITY.md` — supported versions, GitHub Security Advisories as the private channel, and the
    trust boundary m1 built, every claim pointing at code that exists (A8).
  - S4 `pyproject.toml` `[project.urls]` with Homepage, Source, Issues, Changelog.
strategy: LICENSE and NOTICE first — they are the grant, and the sdist assertion depends on them —
  then the prose, then the metadata.
regression floor: `scripts/verify_sdist_contents.py` passes, `test_sdist_manifest.py` and the m1
  release checks stay green.

## EDGES
- E1 `NOTICE` must be present in a BUILT sdist, not merely in the repo — the allowlist is enforced in
  both directions, so a file in the tree that is not named still fails (A10, A11).
- E2 The README statement must survive a reader who reads only that section: it names the escape hatch
  (A4) as well as the constraint.
- E3 No overclaim anywhere — a repository-wide check, not a check of the files this node writes (M5).
- E4 A `-cls` or `-obb` weight is covered by the disclosure too, not only the pinned ten (A15).

## CHECKS
- test_license_names_a_real_copyright_holder · covers: M1, A1 · no `[yyyy]` and no
  `[name of copyright owner]` survives anywhere in LICENSE.
- test_notice_records_the_weight_provenance · covers: M2, A2, A12 · names ultralytics, AGPL-3.0, the
  licence URL, and that this package redistributes no weights.
- test_notice_ships_in_the_sdist · covers: M2, M7, R:UNSHIPPED, A10, A11, E1 · built sdist carries
  NOTICE, and the allowlist REQUIRES it rather than merely tolerating it.
- test_readme_states_all_four_licence_claims · covers: M3, A3, E2 · code Apache-2.0, weights AGPL and
  runtime-fetched, nothing redistributed, and how to supply your own.
- test_readme_licence_section_is_top_level · covers: A5 · reachable, not a footnote.
- test_no_document_claims_the_weights_are_apache · covers: M5, R:OVERCLAIM, E3 · repository-wide.
- test_contributing_clean_claim_is_qualified · covers: M5, R:OVERCLAIM · the sentence says what it
  means rather than more than it means.
- test_disclosure_covers_every_builtin_variant · covers: A15, E4 · cls and obb included, not only the
  pinned ten.
- test_security_md_names_a_private_reporting_channel · covers: M4, A7, A18 · advisories, not an email.
- test_security_md_documents_the_shipped_trust_boundary · covers: M4, A8, A6 · every claim names code
  that exists.
- test_security_md_names_supported_versions · covers: M4, A9 · the current line, no LTS promise.
- test_pyproject_declares_project_urls · covers: M6, A13, A19 · Homepage, Source, Issues, Changelog.
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
