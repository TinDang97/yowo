"""The mechanism for retiring a public name without breaking anyone.

Before this module, ``src/`` contained zero ``DeprecationWarning`` while the
package sat at 2.5.0 on PyPI with 191 public names across 19 modules. Two
source comments already deferred decisions to "the deprecation policy" -- a
policy that did not exist -- and the m4 milestone was constrained to
additive-only as a direct consequence: it could add names and fix behaviour,
but never remove or rename one, because there was no channel to announce a
removal through.

The rules this implements are written in ``docs/stability-policy.md`` and were
decided by the author on 2026-09-16:

* **Public** means a name in a module's ``__all__``. That is the only
  definition already mechanically checkable here, and
  ``scripts/check_public_surface.py`` enforces it.
* A deprecated name warns for **at least one minor release** and is removed no
  sooner than the **next major**. This promises nothing new; it makes explicit
  what semver already implies for a published 2.5.0.
* A recorded ``removed_in`` is a **promise, not an enforcement**. When that
  version arrives the name keeps warning and removal stays a human decision. A
  library that starts raising because a version passed is worse than one that
  keeps warning: it breaks people who upgraded for unrelated reasons.

Marking a name is reversible. Deleting its :data:`DEPRECATIONS` entry and the
call in its ``__post_init__`` restores it untouched, because a deprecation
never changes what the name does -- that would be two breaks announced as one.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

__all__ = ["DEPRECATIONS", "Deprecation", "warn_deprecated"]


@dataclass(frozen=True)
class Deprecation:
    """One deprecated public name, and everything a caller needs to act.

    Attributes:
        name: The public name being retired.
        since: Version in which it was first marked.
        removed_in: Earliest version it may be removed in. A promise, not an
            enforcement -- see the module docstring.
        instead: The dotted path a caller should use in its place.
    """

    name: str
    since: str
    removed_in: str
    instead: str


#: Every deprecated public name, in one place a check can enumerate. A second
#: copy of this list anywhere would be a list inside the thing meant to notice
#: it changing -- the defect class this project keeps finding.
DEPRECATIONS: dict[str, Deprecation] = {
    "ExportResult": Deprecation(
        name="ExportResult",
        since="2.6.0",
        removed_in="3.0.0",
        # `export_model` has always returned ExportMetadata; ExportResult is a
        # 7-field public name that nothing in this package ever constructed.
        instead="yowo.export.ExportMetadata",
    ),
}


def warn_deprecated(record: Deprecation, stacklevel: int = 4) -> None:
    """Emit the deprecation warning for *record*.

    The message is built FROM the record rather than written out beside it, so
    a warning and its recorded promise cannot drift apart.

    Args:
        record: The deprecation to announce.
        stacklevel: Frames to skip so the warning names the CALLER's line.
            The default of 4 is what a dataclass ``__post_init__`` needs, and
            the count is not obvious: this function, then ``__post_init__``,
            then the ``__init__`` the dataclass decorator GENERATES (which
            lives in ``<string>``), then the caller. Measured rather than
            reasoned -- 3 blames ``types.py``, which tells a user nothing
            about their own code.

    Note:
        The category is ``DeprecationWarning``, which CPython hides by default
        outside ``__main__``. That is deliberate: it is the category the
        ecosystem filters on, and a library author does not get to decide a
        consumer's noise level by reaching for something louder.
    """
    warnings.warn(
        f"{record.name} is deprecated since {record.since} and may be removed in "
        f"{record.removed_in}. Use {record.instead} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
