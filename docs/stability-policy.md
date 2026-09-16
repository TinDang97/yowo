# Stability policy

What this package promises about its public surface, and how a name is retired
without breaking you.

Written 2026-09-16, after the mechanism it describes. Two source comments had
deferred decisions to "the deprecation policy" for months while no such policy
existed, so this document describes something that runs rather than something
intended.

## What is public

**A name is public when it appears in a module's `__all__`.** Nothing else is,
regardless of whether it lacks a leading underscore.

That is a narrower promise than "anything importable", and it is deliberate: it
is the only definition already checkable by machine here, and
`scripts/check_public_surface.py` enforces it against a consumer install on
every CI run. A name that is merely reachable — `yowo.engine._resolve_model_meta`,
say — may change or disappear in any release.

If you depend on something that is not in an `__all__`, open an issue asking for
it to be published rather than relying on it where it sits.

## How long a deprecated name survives

A deprecated name warns for **at least one minor release** and is removed **no
sooner than the next major**.

Concretely: a name marked in 2.6.0 may not be removed in 2.7.0, 2.8.0 or any
other 2.x. The earliest it can go is 3.0.0.

This promises nothing new. The package is at 2.5.0 and semver already implies
it; the policy makes it explicit and gives you a whole major cycle to react.

## What a deprecation does, and does not, do

Marking a name deprecated **changes only one thing: it warns.** The name stays
importable, keeps every field and every behaviour, and passes the same tests it
passed before. A deprecation that also changed semantics would be two breaks
announced as one.

The warning is a `DeprecationWarning`. CPython hides that category by default
outside `__main__`, which is intended — it is the category the ecosystem
filters on, and a library author does not get to decide your noise level by
reaching for something louder. To see them:

```bash
python -W always::DeprecationWarning -m yourapp
# or, in pytest
pytest -W error::DeprecationWarning
```

## The removal version is a promise, not an enforcement

Each deprecation records a `removed_in` version. When the package reaches that
version, **the name keeps warning**. Removal remains a human decision in a
reviewed commit.

A library that starts raising because a version number passed is worse than one
that keeps warning: it breaks people who upgraded for entirely unrelated
reasons, at a moment they did not choose.

## Where deprecations are recorded

One place: `DEPRECATIONS` in `src/yowo/_deprecation.py`. Each entry carries the
name, the version it was marked in, the earliest version it may be removed in,
and what to use instead. The warning message is built from that record, so a
warning and its recorded promise cannot drift apart.

## Currently deprecated

| name | since | may be removed in | use instead |
|---|---|---|---|
| `ExportResult` | 2.6.0 | 3.0.0 | `yowo.export.ExportMetadata` |

`ExportResult` is a 7-field dataclass that **nothing in this package ever
returned** — `export_model` has always returned the 24-field `ExportMetadata`.
It was published, documented three incompatible ways, and constructed by
nobody. See issue #84.
