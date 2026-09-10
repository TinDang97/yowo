---
type: Milestone
title: Nothing yowo publishes is wrong, unverified, or unreproducible
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified:
  - { by: "Tin Dang", at: 2026-09-10, act: check, authority: process, via: process, boxes: "EXIT:8" }
  - { by: "Tin Dang", at: 2026-09-10, act: check, authority: process, via: process, boxes: "EXIT:3,4,7" }
  - { by: "Tin Dang", at: 2026-09-10, act: check, authority: process, via: process, boxes: "EXIT:1" }
advised_by: milestone-planner
---
## CARD
goal: Every byte yowo publishes is intentional, verified, reproducible, and correctly attributed.
why: The live harms are here. `yowo 2.4.1` is on PyPI right now with an AGPL-3.0 weight inside an Apache-2.0 declaration; the release pipeline has been red on every push since v2.5.0 and has never had a publish step; a live PyPI token sat world-readable on the maintainer's disk; and the package reports a version it is not. Nothing else on the roadmap matters while the thing that ships is wrong.
next: add new task <slug>

## SCOPE
In:  sdist/wheel contents and reproducibility · the publish path (OIDC trusted publishing, pinned tooling) · PR-triggered CI · version identity single-sourcing · model-weight integrity (SHA-256, `weights_only`) · RTSP credential redaction · licensing and provenance docs (SECURITY.md, weight AGPL note, project URLs)
Out: test *content* beyond what these tasks need (→ m3-prove-it) · runtime timeout/reconnect behaviour (→ m2-survive-week-two) · export artifact correctness (→ m4-honest-deployment) · the CI *matrix* across Pythons and OSes (→ m3-prove-it; this milestone only makes CI trigger on PRs at all)

## GROUND
touches: pyproject.toml · .github/workflows/ · src/yowo/models/_weights.py · src/yowo/arch/_weights.py · src/yowo/models/_registry.py · src/yowo/io/_source.py · src/yowo/types.py · src/yowo/__init__.py · src/yowo/export/_exporter.py · CONTRIBUTING.md · README.md · SECURITY.md
risks:
  - Adding SHA-256 verification to `ModelMeta` breaks every existing cached weight and every offline user on first upgrade. The migration path (verify-on-hit, warn-and-refetch vs hard-fail) is a human decision, not a default.
  - Redacting the RTSP URL from `Frame.source_id` changes a field that appears in result JSON — anyone keying on it downstream breaks. This is a contract change, not a bugfix.
  - Fixing the sdist include list changes what is published; the first clean build must be diffed against 2.4.1 before it is pushed anywhere.

## EXIT
- [x] CI builds the sdist+wheel twice under a pinned `SOURCE_DATE_EPOCH` and asserts identical digests; the digest is recorded in EVIDENCE so a local build can be diffed against it, and the sdist contains only files on an explicit include list   (← sdist-manifest, reproducible-sdist)
      SPLIT 2026-09-08: `sdist-manifest` shipped the last clause only — the explicit include list — and closed at gate PASS having satisfied its own RULES, which were authored narrower than this box. The reproducibility clauses were never built. `reproducible-sdist` owns them. This box stays unchecked until both land. Recorded because a green gate proves the declared checks ran, never that they were sufficient.
- [ ] (i) the release workflow declares `id-token: write` and greps clean of `password:` / `PYPI_API_TOKEN`; (ii) a TestPyPI dry-run publishes from a tag; (iii) every release-path tool is pinned to an exact version — an exact tag for each action (`@v4.2.2`, not `@v4`) and `==` for python-semantic-release. AMENDED 2026-09-08 from "pinned to a SHA" by human decision: SHA pinning is strictly stronger, but this repo has no automation to refresh SHA pins and a stale pin nobody updates is its own risk. RESIDUAL RISK, named rather than hidden: a tag is mutable, so a compromised upstream could republish the same version pointing at different code. Revisit if Dependabot for actions lands; (iv) PyPI shows zero active project API tokens — recorded as a dated human attestation in EVIDENCE   (← pypi-trusted-publish)
- [x] `yowo.__version__`, `yowo --version`, distribution metadata and every export sidecar report the same version, asserted by a check that fails the build on drift   (← version-single-source)
- [x] `ci.yml` exists, runs on `pull_request`, and its job names are frozen as the contract later tasks append to; the GitHub branch-protection API reports the gate as a required status check — or that leg is recorded as a human attestation   (← pr-ci-gate)
- [ ] No downloaded weight is deserialized before its pinned SHA-256 is verified — on first download and on every cache hit — and `torch.load` runs with `weights_only=True`. The cache-migration policy for already-cached weights is decided by a human and recorded   (← weight-integrity)
- [ ] No credential reaches a log, an exception message, a result payload, or a cache key, asserted by a check using a credentialed RTSP URL   (← rtsp-credential-redaction)
- [x] CI can obtain a real model weight whose SHA-256 is verified before use, without a network fetch per job, and without committing the weight to the repository   (← ci-weight-fixture)
- [x] The licence of every shipped byte is declared, weight provenance and its AGPL implications are documented, and SECURITY.md exists. Checkable only once the sdist include list exists — depends on box 1   (← licensing-provenance)
## CLOSE
evidence: <one row per task>
