"""Export-only wrapper that externalizes Attention KV tensors as ONNX I/O.

``YOLOKVWrapper`` wraps a ``YOLOModel`` and converts KV cache state from
internal Python attributes into explicit ONNX model inputs/outputs.

This enables KV-cached inference on stateless runtimes (ONNX Runtime,
TensorRT, OpenVINO) by passing ``past_k_i`` / ``past_v_i`` tensors across
frames at the session boundary rather than storing them in Python state.

Usage::

    from yowo.export._kv_wrapper import YOLOKVWrapper
    wrapper = YOLOKVWrapper(model)
    # Pass to torch.onnx.export — see _exporter.py
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from yowo.arch._attention import Attention
from yowo.arch._yolo import YOLOModel

__all__ = ["YOLOKVWrapper"]


class YOLOKVWrapper(nn.Module):
    """Export-only wrapper: externalizes Attention KV cache as ONNX I/O.

    On construction, all ``Attention`` modules in the wrapped model are set
    to ``_export_mode = True``.  ``forward()`` distributes ``past_k/v``
    tensors to each module before running the model, then collects the
    freshly computed ``present_k/v`` tensors for the next frame.

    This class is NOT intended for direct inference — use it only inside
    ``torch.onnx.export``.
    """

    def __init__(self, model: YOLOModel) -> None:
        super().__init__()
        self.model = model

        # Collect Attention modules in registration order (= execution order)
        self._attention_modules: list[Attention] = [
            m for m in model.modules() if isinstance(m, Attention)
        ]

        # Enable export path in every Attention module
        for attn in self._attention_modules:
            attn.enable_export_mode()

    # ------------------------------------------------------------------
    # Metadata properties (used by _exporter.py to build ONNX I/O lists)
    # ------------------------------------------------------------------

    @property
    def num_attn(self) -> int:
        """Number of Attention modules in the wrapped model."""
        return len(self._attention_modules)

    @property
    def kv_input_names(self) -> list[str]:
        """Ordered ONNX input names for KV cache (after 'images').

        Format: ``["use_cache", "past_k_0", "past_v_0", "past_k_1", ...]``
        """
        names: list[str] = ["use_cache"]
        for i in range(self.num_attn):
            names.append(f"past_k_{i}")
            names.append(f"past_v_{i}")
        return names

    @property
    def kv_output_names(self) -> list[str]:
        """Ordered ONNX output names for present KV (after 'output0').

        Format: ``["present_k_0", "present_v_0", "present_k_1", ...]``
        """
        names: list[str] = []
        for i in range(self.num_attn):
            names.append(f"present_k_{i}")
            names.append(f"present_v_{i}")
        return names

    def build_dummy_kv_inputs(self, imgsz: int, dtype: object = None) -> list[Tensor]:
        """Build zero-filled dummy K,V tensors for ONNX export tracing.

        Args:
            imgsz: Square input image size (e.g. 640). K,V spatial dim is
                ``(imgsz // 32) ** 2`` (stride-32 feature map).
            dtype: Tensor dtype (e.g. ``torch.float32``). Defaults to fp32.

        Returns:
            Flat list ``[past_k_0, past_v_0, past_k_1, ...]`` with shape
            ``(1, num_heads, N, key_dim)`` and ``(1, num_heads, N, head_dim)``.
        """
        import torch

        N = (imgsz // 32) ** 2
        tensors: list[Tensor] = []
        for attn in self._attention_modules:
            k_shape = (1, attn.num_heads, N, attn.key_dim)
            v_shape = (1, attn.num_heads, N, attn.head_dim)
            kw = {"dtype": dtype} if dtype is not None else {}
            tensors.append(torch.zeros(k_shape, **kw))  # type: ignore[arg-type]
            tensors.append(torch.zeros(v_shape, **kw))  # type: ignore[arg-type]
        return tensors

    # ------------------------------------------------------------------
    # Forward — called during torch.onnx.export tracing
    # ------------------------------------------------------------------

    def forward(
        self,
        images: Tensor,
        use_cache: Tensor,
        *past_kvs: Tensor,
    ) -> tuple[Tensor, ...]:
        """Run the model with externalized KV I/O.

        Args:
            images: ``(B, 3, H, W)`` input tensor.
            use_cache: Scalar float tensor (0.0 = cold, 1.0 = warm).
            *past_kvs: Interleaved ``(past_k_0, past_v_0, past_k_1, ...)``
                tensors from the previous frame.  Must supply exactly
                ``2 * num_attn`` tensors.

        Returns:
            ``(detections, present_k_0, present_v_0, ...)`` — detection
            output followed by freshly computed K,V for each Attention.
        """
        # Distribute past K,V and use_cache to each Attention module
        for i, attn in enumerate(self._attention_modules):
            attn.set_export_inputs(past_kvs[2 * i], past_kvs[2 * i + 1], use_cache)

        # Run the full model (Attention.forward dispatches to _forward_kv_export)
        detections = self.model(images)

        # Collect freshly computed present K,V from each Attention module
        present_kvs: list[Tensor] = []
        for attn in self._attention_modules:
            present_k, present_v = attn.get_export_outputs()
            present_kvs.append(present_k)
            present_kvs.append(present_v)

        return (detections, *present_kvs)
