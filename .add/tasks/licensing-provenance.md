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
  - S1 <the surface this publishes — an endpoint, function, or section>
depends_on:
  - /tasks/sdist-manifest.md
  - /tasks/weight-integrity.md
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
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
- M1 <the rule that must hold>
</must>
<reject>
- R:<NAME> <what must never happen> -> "<NAME>"
</reject>

## ASSUMPTIONS
- A1 [who] covers: <S ids> · the request does not say <who may act / whose data>; taking <reading> -> <cost if wrong>
- A2 [which] covers: <S ids> · the request does not say <which rows/cases are in>; taking <reading> -> <cost if wrong>
- A3 [when] covers: <S ids> · the request does not say <where the boundary falls>; taking <reading> -> <cost if wrong>
- A4 [absent] covers: <S ids> · the request does not say <what a missing value means>; taking <reading> -> <cost if wrong>
- A5 [order] covers: <S ids> · the request does not say <what orders / breaks a tie>; taking <reading> -> <cost if wrong>
- A6 [experience] covers: <S ids> · the request does not say <who receives this and what would make it hard for them>; taking <reading> -> <cost if wrong>
every `gives:` surface is swept on every dimension; `[<dim>] n/a · <why>` retires one. one line, one silence — split, never bundle. `· probe: <what shipped behavior must show>` declares a reading checkable: cite its A id from CHECKS and the gate holds the PASS to it.

## PLAN
contract: <the shape this publishes>

## EDGES
- E1 <a boundary or failure case a check must cover — optional>

## CHECKS
- <test_name> · covers: M1 · <what it proves>
red-first: every check MUST fail first.

## EVIDENCE
receipt: <runs/<n>.md>
gate: <PASS | RISK-ACCEPTED | HARD-STOP>

## LESSONS
- <lesson> -> add learn <lens>
