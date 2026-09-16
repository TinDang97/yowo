"""Read facts back off a produced artifact, instead of copying the request.

Every field the sidecar records about a graph used to be taken from what the
caller asked for. Measured 2026-09-16 on yolo11n/fp32: ``opset_version=17``
was requested and ``ai.onnx 18`` was produced -- the downconversion raises
(``No Adapter To Version $17 for Resize``) and the failure is swallowed -- and
a ``yolo11n-cls`` export recorded ``input_shape: [1, 3, 640, 640]`` for a
``[batch, 3, 224, 224]`` graph, 8.16x the pixels.

The functions here have no opinion about the export. They read what is on
disk, or say they could not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from yowo.errors import ExportError

logger = logging.getLogger(__name__)

#: A protobuf message cannot exceed 2 GiB, so a model above it CANNOT be
#: written with its weights inline. External data is then correct rather than
#: a defect -- and the sidecar has to name the companion file instead of
#: pretending the ``.onnx`` is the whole artifact.
PROTOBUF_CEILING_BYTES = 2 * 1024**3

#: A dimension the graph leaves symbolic (``"batch"``). Callers substitute the
#: concrete value they exported with; ``None`` would be ambiguous against a
#: shape that could not be read at all.
DYNAMIC_DIM = -1


@dataclass(frozen=True)
class GraphFacts:
    """What could actually be read off an artifact. ``None`` means unknown.

    ``None`` is load-bearing: a tensorrt engine or an openvino directory has
    no opset this module can read, and recording an integer there would be a
    claim nothing measured.
    """

    opset: int | None = None
    input_shape: list[int] | None = None


def read_onnx_graph_facts(path: Path) -> GraphFacts:
    """The opset and input shape of an ONNX graph, or unknowns.

    Never raises. A path that is not an ONNX graph -- a ``.engine``, an
    openvino directory, a coreml package -- is not an error here; it is a
    thing whose opset this module does not know.
    """
    try:
        import onnx  # type: ignore[import-untyped]

        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # unreadable is a fact here, not a failure
        logger.debug("no ONNX graph facts from %s: %s", path, exc)
        return GraphFacts()

    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        None,
    )

    shape: list[int] | None = None
    if model.graph.input:
        shape = [
            d.dim_value if d.HasField("dim_value") else DYNAMIC_DIM
            for d in model.graph.input[0].type.tensor_type.shape.dim
        ]

    return GraphFacts(opset=opset, input_shape=shape)


def external_companions(path: Path) -> list[str]:
    """Names of the files an ONNX graph needs beside it, plus the sibling.

    The sibling ``<name>.onnx.data`` is included even when no initializer
    references it, because ``torch.onnx.export`` writes one and onnxslim then
    inlines the weights without removing it -- measured 2026-09-16 as a
    10,616,832 B file that nothing points at and nothing deletes. Unreferenced
    or not, it is on disk, so either the sidecar names it or it goes.
    """
    names: set[str] = set()
    try:
        import onnx  # type: ignore[import-untyped]

        model = onnx.load(str(path), load_external_data=False)
        for init in model.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                names.update(kv.value for kv in init.external_data if kv.key == "location")
    except Exception as exc:
        logger.debug("could not read external data locations from %s: %s", path, exc)

    sibling = path.parent / f"{path.name}.data"
    if sibling.exists():
        names.add(sibling.name)

    # `location` is a string inside the graph, and `internalize_external_data`
    # UNLINKS what this returns. A location of `../../something` would resolve
    # outside `path.parent` and delete a file the export never wrote. Today
    # the only graph reaching here is one this package produced seconds ago,
    # so this is defence for a caller that does not exist yet -- but the cost
    # is one comparison and the failure mode is deleting someone else's file.
    safe: set[str] = set()
    root = path.parent.resolve()
    for name in names:
        candidate = (path.parent / name).resolve()
        if candidate.parent != root:
            logger.warning("ignoring external-data location %r: it resolves outside %s", name, root)
            continue
        safe.add(name)

    return sorted(n for n in safe if (path.parent / n).exists())


def internalize_external_data(
    path: Path,
    *,
    size_ceiling_bytes: int = PROTOBUF_CEILING_BYTES,
) -> list[str]:
    """Fold weights into the ``.onnx`` and drop the companion; name what stays.

    This is `_export_onnx_kv`'s existing behaviour applied to the ordinary
    export path. That one internalizes because the CoreML EP cannot follow an
    external-data reference; this one does it because a user on a
    ``yowo[pytorch]`` install -- the install ``_exporter.py``'s own
    DependencyError names -- otherwise gets a 612,255 B stub that needs 130 of
    its 201 initializers from a file the sidecar never mentions.

    Returns the companion files that REMAIN, which is empty below the ceiling
    and the surviving names above it. Above the ceiling the weights cannot go
    inline at all, so the companion file is correct and must be named.
    """
    companions = external_companions(path)
    if not companions:
        # Nothing lives outside the file, so it is already self-contained.
        # Returning here rather than round-tripping it through onnx keeps this
        # function from being the thing that decides whether a produced
        # artifact parses -- that is the exporter's business, and a caller
        # that mocked `torch.onnx.export` away should not meet a DecodeError
        # raised by a step whose whole job is a no-op for them.
        return []

    total = path.stat().st_size + sum(
        (path.parent / n).stat().st_size for n in companions if (path.parent / n).exists()
    )

    if total >= size_ceiling_bytes:
        logger.info(
            "%s is %.1f MB with its external data, at or above the %.1f MB protobuf "
            "ceiling -- keeping %d companion file(s), which the sidecar names",
            path.name,
            total / 1_048_576,
            size_ceiling_bytes / 1_048_576,
            len(companions),
        )
        return companions

    try:
        import onnx  # type: ignore[import-untyped]

        model = onnx.load(str(path), load_external_data=True)
        tmp = path.with_suffix(".onnx.tmp")
        onnx.save(model, str(tmp))
        tmp.replace(path)
    except Exception as exc:
        raise ExportError(
            f"could not make {path.name} self-contained: {type(exc).__name__}: {exc}"
        ) from exc

    for name in companions:
        (path.parent / name).unlink(missing_ok=True)

    return []


def produced_files(output_dir: Path, before: frozenset[str], entry: Path) -> list[str]:
    """Names of the files this export added, entry file first.

    ``before`` is the directory's contents at the start. ``output_dir`` is a
    user directory that may hold anything, and claiming a file the export did
    not write is the mirror of the defect this fixes.
    """
    now = {p.name for p in output_dir.iterdir() if not p.name.endswith(".yowo.json")}
    added = sorted(now - before)
    if entry.name in added:
        added.remove(entry.name)
    return [entry.name, *added]


def verify_sidecar_describes_directory(
    sidecar_path: Path,
    output_dir: Path,
    *,
    ignore: frozenset[str] = frozenset(),
) -> None:
    """Load the sidecar back and refuse one that does not match the directory.

    ``ExportMetadata.load()`` had zero callers in ``src/``. That is why a
    sidecar could name one file while the directory held two and nothing
    noticed: nothing ever read it back. This gives it its first caller, at
    export time, so a mismatch surfaces to whoever ran the export rather than
    to whoever deploys it.
    """
    from yowo.export._metadata import ExportMetadata

    meta = ExportMetadata.load(sidecar_path)
    named = set(meta.artifact_files)

    missing = sorted(n for n in named if not (output_dir / n).exists())
    if missing:
        raise ExportError(
            f"sidecar {sidecar_path.name} names {missing}, which the export did not leave in "
            f"{output_dir}"
        )

    present = {p.name for p in output_dir.iterdir() if not p.name.endswith(".yowo.json")} - ignore
    orphans = sorted(present - named)
    if orphans:
        raise ExportError(
            f"the export left {orphans} in {output_dir} and sidecar {sidecar_path.name} names "
            f"none of them -- a file a deploying reader can neither safely ship nor safely delete"
        )
