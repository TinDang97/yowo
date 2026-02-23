"""Exception hierarchy for yowo.

All exceptions inherit from YowoError for catch-all handling.
Each module raises only its own error subtype — never a sibling's type.

Hierarchy::

    YowoError
    ├── DependencyError         # Missing optional package
    ├── BackendError            # Backend init/inference failure
    │   ├── BackendLoadError    # Failed to load model into backend
    │   └── InferenceError      # Runtime inference failure
    ├── DeviceError             # Device not found / OOM
    ├── ModelError
    │   ├── ModelNotFoundError  # .pt or exported file not found
    │   └── ModelLoadError      # File found but corrupt/incompatible
    ├── ExportError             # Export operation failed
    │   └── ExportUnsupportedError  # Format not supported on this platform
    ├── SourceError             # Cannot open input source
    │   └── SourceTimeoutError  # Stream timed out
    └── ConfigError             # Invalid configuration
"""

from __future__ import annotations


class YowoError(Exception):
    """Base class for all yowo exceptions.

    Catch this to handle any library error uniformly.
    """


# ---------------------------------------------------------------------------
# Dependency errors
# ---------------------------------------------------------------------------


class DependencyError(YowoError):
    """A required optional package is not installed.

    Args:
        package: Import name of the missing package (e.g. ``"tensorrt"``).
        install_cmd: Command the user should run to install it.
            Falls back to ``pip install <package>`` when empty.
        message: Additional context. Combined with the standard message
            when provided.

    Example::

        raise DependencyError(
            "onnxruntime",
            install_cmd="pip install yowo[onnx]",
        )
    """

    def __init__(
        self,
        package: str,
        install_cmd: str = "",
        message: str = "",
    ) -> None:
        self.package = package
        self.install_cmd = install_cmd or f"pip install {package}"

        parts = [f"Missing optional dependency: '{package}'."]
        parts.append(f"Install it with: {self.install_cmd}")
        if message:
            parts.append(message)

        super().__init__(" ".join(parts))


# ---------------------------------------------------------------------------
# Backend errors
# ---------------------------------------------------------------------------


class BackendError(YowoError):
    """Base class for all inference backend failures."""


class BackendLoadError(BackendError):
    """Backend could not load the model artifact.

    Raised when a serialised model (engine, onnx, …) cannot be deserialised
    into the backend runtime — e.g. version mismatch or corrupt file.
    """


class InferenceError(BackendError):
    """Runtime inference failure after the model was loaded successfully.

    Raised on shape mismatches, OOM during a forward pass, or any other
    error that occurs *during* an inference call.
    """


# ---------------------------------------------------------------------------
# Device errors
# ---------------------------------------------------------------------------


class DeviceError(YowoError):
    """The requested compute device is unavailable or out of resources.

    Raised when a CUDA device index is out of range, a GPU OOM occurs
    outside of inference, or a driver version is incompatible.
    """


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------


class ModelError(YowoError):
    """Base class for model file / registry failures."""


class ModelNotFoundError(ModelError):
    """The requested .pt weights or exported artifact does not exist.

    Raised when a path does not exist on disk and cannot be downloaded,
    or when the model is not registered in the model registry.
    """


class ModelLoadError(ModelError):
    """The model file exists but cannot be loaded.

    Raised when a .pt file is found but is corrupt, truncated, or was
    saved with an incompatible version of PyTorch / ultralytics.
    """


# ---------------------------------------------------------------------------
# Export errors
# ---------------------------------------------------------------------------


class ExportError(YowoError):
    """Base class for model export failures."""


class ExportUnsupportedError(ExportError):
    """The requested export format is not supported on this platform.

    Raised when, for example, TensorRT export is attempted on macOS or
    when an INT8 export is requested without calibration data.
    """


# ---------------------------------------------------------------------------
# Source errors
# ---------------------------------------------------------------------------


class SourceError(YowoError):
    """The input source cannot be opened or read.

    Raised when a file path does not exist, an RTSP URL is unreachable,
    or a webcam device index is invalid.
    """


class SourceTimeoutError(SourceError):
    """An open stream stopped producing frames within the expected interval.

    Raised by the stream reader after the reconnect timeout expires without
    receiving a new frame.
    """


# ---------------------------------------------------------------------------
# Config errors
# ---------------------------------------------------------------------------


class ConfigError(YowoError):
    """Invalid or inconsistent configuration values.

    Raised during dataclass ``__post_init__`` validation or when a YAML
    config file contains unrecognised keys.
    """


__all__ = [
    "BackendError",
    "BackendLoadError",
    "ConfigError",
    "DependencyError",
    "DeviceError",
    "ExportError",
    "ExportUnsupportedError",
    "InferenceError",
    "ModelError",
    "ModelLoadError",
    "ModelNotFoundError",
    "SourceError",
    "SourceTimeoutError",
    "YowoError",
]
