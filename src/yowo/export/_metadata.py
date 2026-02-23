"""Export metadata sidecar (.yowo.json files)."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExportMetadata:
    """Serialisable record of a completed model export.

    Written as a .yowo.json sidecar alongside every exported model artifact.
    """

    model_name: str
    format: str
    precision: str
    imgsz: int
    batch_size: int
    dynamic: bool
    input_shape: list[int]
    file_path: str
    file_size_bytes: int
    created_at: str
    export_duration_sec: float
    source_weights: str
    yowo_version: str
    python_version: str = field(default_factory=platform.python_version)
    platform_system: str = field(default_factory=platform.system)
    platform_machine: str = field(default_factory=platform.machine)
    gpu_name: str | None = None
    calibration_data: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        """Write .yowo.json alongside model. Atomic write."""
        out = path or Path(self.file_path).with_suffix("").with_suffix(".yowo.json")
        tmp = out.with_suffix(".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(out)
        return out

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def load(cls, path: Path) -> ExportMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Backward compat: remap old sidecar key from pre-0.2 exports
        if "ultralytics_version" in data and "yowo_version" not in data:
            data["yowo_version"] = data.pop("ultralytics_version")
        return cls(**data)


__all__ = ["ExportMetadata"]
