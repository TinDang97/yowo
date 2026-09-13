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
- [ ] Config objects round-trip through `asdict`/`yaml.safe_dump` and no field can bypass its own validation   (← config-contract-repair) · AMENDED 2026-09-13: both halves are broken far beyond the single field the milestone `why` blames, so the node must not be scoped to that field. Measured — (a) BYPASS works on all three configs: `_VALID_LOG_LEVELS` is a real dataclass field, so `InferenceConfig(log_level='PWNED', _VALID_LOG_LEVELS=frozenset({'PWNED'}))` is ACCEPTED. (b) `yaml.safe_dump(asdict(cfg))` raises `RepresenterError` on the frozenset, and with private fields stripped it still raises on `PosixPath` — every `Path` and all 13 yowo `StrEnum`s are unrepresentable. (c) The loader SILENTLY DROPS four fields on the way back: `num_classes` 7→None, `class_names` ['a','b']→None, `log_level` 'DEBUG'→'WARNING', `structured_logging` True→False. Unknown keys are silently ignored too. (d) `YOWO_STRUCTURED_LOGGING` is the ONLY one of six bool env vars parsed without `.lower()` (`config.py:466`, `:601`), so `YOWO_STRUCTURED_LOGGING=True` silently does nothing while `YOWO_CACHE=True` works. A round-trip test written against one field would go green over all of this.
- [ ] The `backend_instance=` extension point is documented and covered by a test that injects a third-party backend **against a conformance suite that does not require a `BackendType` enum member**   (← backend-extension-contract) · AMENDED 2026-09-13: UNSATISFIABLE as originally written. The m3 conformance suite parametrises over the `BackendType` enum, and a genuine third-party backend has no enum member — so "admitted by passing the m3 conformance suite" could only be satisfied by the backend LYING about its identity (returning `BackendType.ONNX`), which is the opposite of what this box wants. Measured: a duck-typed backend implementing all 9 `InferenceBackend` protocol members injects successfully and lands a bare `str` ('acme-npu') in `BackendSelection.backend`, a public frozen dataclass field annotated `BackendType`. So the extension point works but the type contract is a lie at runtime. Closing this box requires either splitting the conformance suite into an enum-free core, or deciding `BackendType` is open — the latter is a public-API decision this box cannot make alone.
- [ ] Package metadata states the real maturity **and a deployment/operations guide exists**   (← ga-metadata) · AMENDED 2026-09-13: the `CHANGELOG.md` 2.5.0 clause is deleted because it is ALREADY TRUE — the section exists, so ticking the box as written would credit work nobody did. What remains is real: `pyproject.toml` still declares `Development Status :: 3 - Alpha` at version 2.5.0, and no deployment/operations guide exists. RESIDUAL affecting the whole milestone: the latest INSTALLABLE release on PyPI is 2.2.1 — 2.4.0 and 2.4.1 are yanked and 2.5.0 was never published — so whether "declare GA" requires a working 2.5.x on PyPI first, and whether it changes the version number, are open decisions this box inherits.
## CLOSE
evidence: <one row per task>
