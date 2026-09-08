---
type: Milestone
title: A stability contract worth pinning
status: direction
generated: { by: add/3.5.0, at: 2026-09-08 }
verified: []
advised_by: milestone-planner
---
## CARD
goal: A public surface a team can pin for three years, and a documented promise about what pinning means.
why: The package is at 2.5.0 — semver already promises stability — with 191 public names across 19 modules, zero `DeprecationWarning` anywhere in `src/`, an `__all__` omitting 15 explicitly re-exported names, a config field that can be passed to bypass its own validation and prevents the config from round-tripping through YAML, and `Development Status :: 3 - Alpha` still in the classifiers. The promise has been made; the contract behind it has not been written.
next: add new task <slug>

## SCOPE
In:  public surface audit and `__all__` correctness · the `_VALID_LOG_LEVELS` ClassVar defect and config round-tripping · a deprecation mechanism and a written stability policy · the `backend_instance=` extension contract, documented and tested · GA metadata (Development Status, project URLs) · deployment and operations guide
Out: new features · anything m1–m4 already closed

## GROUND
touches: src/yowo/__init__.py · src/yowo/config.py · src/yowo/types.py · src/yowo/errors.py · src/yowo/backends/__init__.py · pyproject.toml · docs/
risks:
  - Correcting `__all__` and removing accidental exports is a breaking change for anyone importing a name that was never meant to be public. It needs a deprecation cycle, which is itself one of this milestone's deliverables — so the order inside this milestone matters.
  - Declaring GA is a commitment that constrains every future change. It should not be taken until m1–m4 have actually closed.

## EXIT
- [ ] A deprecation mechanism exists and fires: at least one `DeprecationWarning` in `src/`, with a test asserting it fires. The written stability policy accompanies it as a review item — the mechanism is the gate   (← deprecation-policy)
- [ ] Every public name is deliberate, in `__all__`, and returns no underscore-prefixed type — across the top-level surface AND `yowo.arch`'s 16 architecture exports, which freeze model internals. If the arch surface is deliberately excluded, that exclusion is written down   (← public-surface-audit)
- [ ] Config objects round-trip through `asdict`/`yaml.safe_dump` and no field can bypass its own validation   (← config-contract-repair)
- [ ] The `backend_instance=` extension point is documented and covered by a test that injects a third-party backend, which is admitted by passing the m3 conformance suite   (← backend-extension-contract)
- [ ] Package metadata states the real maturity, `CHANGELOG.md` has its missing 2.5.0 section, and a deployment/operations guide exists   (← ga-metadata)
## CLOSE
evidence: <one row per task>
