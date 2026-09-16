"""The inputs a cached artifact depends on, named in one place.

Three caches key artifacts on a machine: the tune profile, the INT8
calibration table and the TensorRT engine cache. Measured 2026-09-16, each
key omitted inputs that change what its artifact IS -- the tune fingerprint
takes only a ``HardwareProfile``, so backend and precision could not be in it
by construction; the calibration table is named from the engine path, so a
different image set reuses the previous table; the TensorRT engine cache has
no prefix, so the filename is whatever the execution provider chooses.

:class:`InvalidationSet` names those inputs once. It is a dataclass rather
than a dict so a check can enumerate its fields MECHANICALLY: a field added
later is perturbed by
``tests/unit/test_cache_keys.py::test_every_element_of_the_invalidation_set_changes_the_key``
without anyone remembering to add it to a list.

An element that cannot be probed on this machine is rendered as an explicit
``<absent>`` marker, never as an empty string: "the driver did not report a
version" and "the driver reported an empty version" are different machines
and must not share a key.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ABSENT",
    "InvalidationSet",
    "calibration_digest",
    "invalidation_for_hardware",
]

#: Rendered in place of an element this machine could not probe. Distinct from
#: an empty string, which means "probed, and the answer was empty".
ABSENT = "<absent>"

#: Hex characters kept from the digest. 16 gives 64 bits -- far past the point
#: where a collision is likelier than a disk error, and short enough to read
#: in a filename.
_KEY_CHARS = 16

#: Read in 1 MiB blocks so a large calibration set never loads whole.
_DIGEST_BLOCK = 1024 * 1024


def _render_value(value: object) -> str:
    """Render one element unambiguously.

    Newlines and backslashes are escaped because the rendering separates
    fields by newline: without this, a value containing ``"\\nbackend=onnx"``
    could forge another field's line and two different sets could collide.
    """
    if value is None:
        return ABSENT
    if isinstance(value, tuple):
        return ",".join(_render_value(item) for item in value)
    return str(value).replace("\\", "\\\\").replace("\n", "\\n")


@dataclass(frozen=True)
class InvalidationSet:
    """Every input a cached artifact depends on.

    Attributes:
        model: Model identifier the artifact was built for.
        backend: Backend that built or will consume the artifact.
        precision: Numerical precision string ("fp32", "fp16", "int8").
        runtime_versions: Versions of the runtimes involved, as
            ``"name=version"`` strings.
        gpu_name: Marketing name of the GPU, or None when there is none.
        gpu_arch: Compute capability as reported, or None.
        driver_version: NVIDIA driver version, or None when unprobeable.
        tensorrt_version: Installed TensorRT version, or None.
        shape_profile: The input shape the artifact was built for.
        calibration_digest: Digest of the INT8 calibration image SET, or None.
        cpu_count: CPUs this process may use -- not the host's core count.
        platform: Platform string.
    """

    model: str
    backend: str
    precision: str
    runtime_versions: tuple[str, ...] = ()
    gpu_name: str | None = None
    gpu_arch: str | None = None
    driver_version: str | None = None
    tensorrt_version: str | None = None
    shape_profile: str | None = None
    calibration_digest: str | None = None
    cpu_count: int | None = None
    platform: str | None = None

    def render(self) -> str:
        """Render the set canonically, one ``name=value`` per line.

        Sorted by field name, so reordering the declaration does not
        invalidate every cache already on disk.
        """
        return "\n".join(
            f"{field.name}={_render_value(getattr(self, field.name))}"
            for field in sorted(dataclasses.fields(self), key=lambda f: f.name)
        )

    def key(self) -> str:
        """The cache key for this set -- a short, stable hex digest."""
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:_KEY_CHARS]


def calibration_digest(images: list[Path]) -> str:
    """Digest the CONTENT of a calibration image set.

    Follows the images, not the directory holding them: renaming the directory
    leaves the digest alone, and changing one image inside it does not. The
    file names are folded in too, so reordering or renaming within the set is
    a different calibration -- which it is, since the order feeds the batches.

    Args:
        images: The calibration images, in the order they will be read.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.name.encode("utf-8"))
        digest.update(b"\0")
        with image.open("rb") as handle:
            while block := handle.read(_DIGEST_BLOCK):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()[:_KEY_CHARS]


def invalidation_for_hardware(
    hw: object,
    *,
    model: str,
    backend: str,
    precision: str,
    shape_profile: str | None = None,
    calibration: str | None = None,
) -> InvalidationSet:
    """Build an :class:`InvalidationSet` from a live ``HardwareProfile``.

    Every element this machine cannot probe stays ``None`` and renders as
    :data:`ABSENT`, so an artifact built where the driver was unknown is not
    reused where it is known.

    Args:
        hw: A ``HardwareProfile``. Typed loosely to keep ``yowo.cache`` free
            of a hardware import at module scope.
        model: Model identifier.
        backend: Backend name.
        precision: Precision string.
        shape_profile: Input shape the artifact is built for, if fixed.
        calibration: Digest from :func:`calibration_digest`, for INT8.

    Returns:
        The populated set.
    """
    import platform as _platform

    from yowo.hardware import effective_cpu_count

    gpu = getattr(hw, "primary_gpu", None)
    libraries = getattr(hw, "libraries", None)

    versions: list[str] = []
    for name in ("torch_version", "cuda_version", "onnxruntime_version"):
        value = getattr(libraries, name, None)
        if value:
            versions.append(f"{name}={value}")

    arch = getattr(gpu, "arch", None)
    return InvalidationSet(
        model=model,
        backend=backend,
        precision=precision,
        runtime_versions=tuple(versions),
        gpu_name=getattr(gpu, "name", None),
        gpu_arch=getattr(arch, "value", None) if arch is not None else None,
        driver_version=getattr(gpu, "driver_version", None),
        tensorrt_version=getattr(libraries, "tensorrt_version", None),
        shape_profile=shape_profile,
        calibration_digest=calibration,
        # The CPUs this process may use, not the host's core count -- two
        # containers with different quotas must not share a cache entry.
        cpu_count=effective_cpu_count(),
        platform=_platform.platform(),
    )
