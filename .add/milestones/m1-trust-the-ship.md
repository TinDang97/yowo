---
type: Milestone
title: Nothing yowo publishes is wrong, unverified, or unreproducible
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
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
- [ ] `python -m build` produces byte-identical sdist+wheel in CI and locally for the same commit, and the sdist contains only files on an explicit include list   (← sdist-manifest)
- [ ] A tagged push publishes to PyPI automatically via OIDC trusted publishing, with every release-path tool pinned, and no long-lived token exists anywhere   (← pypi-trusted-publish)
- [ ] `yowo.__version__`, `yowo --version`, distribution metadata and every export sidecar report the same version, asserted by a check that fails the build on drift   (← version-single-source)
- [ ] Every pull request runs the quality gate before merge   (← pr-ci-gate)
- [ ] No downloaded weight is deserialized before its pinned SHA-256 is verified — on first download and on every cache hit — and `torch.load` runs with `weights_only=True`   (← weight-integrity)
- [ ] No credential reaches a log, an exception message, a result payload, or a cache key, asserted by a check using a credentialed RTSP URL   (← rtsp-credential-redaction)
- [ ] The licence of every shipped byte is declared, weight provenance and its AGPL implications are documented, and SECURITY.md exists   (← licensing-provenance)

## CLOSE
evidence: <one row per task>
