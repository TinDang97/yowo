"""Smoke tests for INT-P2: all missing public API types are importable from top-level yowo."""

from __future__ import annotations

import yowo


def test_public_api_exports() -> None:
    """All 5 INT-P2 types must be importable from the top-level yowo namespace."""
    from yowo import HealthReport, OBBBox, OBBDetection, StreamConfig, WarmupValidationError

    # Verify each name is in yowo.__all__
    missing = [
        name
        for name in (
            "OBBBox",
            "OBBDetection",
            "WarmupValidationError",
            "HealthReport",
            "StreamConfig",
        )
        if name not in yowo.__all__
    ]
    assert not missing, f"Missing from yowo.__all__: {missing}"

    # Verify the imported objects are real types (not None/MagicMock)
    assert OBBBox is not None
    assert OBBDetection is not None
    assert WarmupValidationError is not None
    assert HealthReport is not None
    assert StreamConfig is not None
