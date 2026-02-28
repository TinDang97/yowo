"""Shared OrtValue inference utility for ONNX and TensorRT backends."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from yowo.types import PreprocessedTensor


def infer_standard_ortvalue(
    ort_module: Any,
    session: Any,
    input_name: str,
    output_names: list[str],
    tensor: PreprocessedTensor,
) -> NDArray[np.float32]:
    """Run inference using OrtValue for zero-copy input wrapping.

    Wraps the input numpy array as an OrtValue to avoid an extra copy
    on GPU execution providers (CUDA / TensorRT).

    Args:
        ort_module: The ``onnxruntime`` module (provides ``OrtValue``).
        session: An ``ort.InferenceSession`` that supports ``run_with_ort_values``.
        input_name: Name of the model's input node.
        output_names: Names of the model's output nodes.
        tensor: Preprocessed input tensor.

    Returns:
        Model output as a float32 numpy array.
    """
    input_ort = ort_module.OrtValue.ortvalue_from_numpy(tensor.data)
    ort_outputs = session.run_with_ort_values(output_names, {input_name: input_ort})
    out = ort_outputs[0].numpy()
    return out if out.dtype == np.float32 else out.astype(np.float32)
