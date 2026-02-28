"""Object counting for yowo — zone-based and line-cross detection counting.

Provides composable post-processing over Detection / TrackedDetection:

* **Per-class counts** — live (current frame) and cumulative (since reset)
* **Zone-based counting** — count detections whose center falls inside a polygon
* **Line-cross counting** — detect IN/OUT crossings of a directed line (requires tracking)

Example::

    from yowo import InferenceEngine, open_source
    from yowo.counter import CountZone, CountLine, ObjectCounter

    zone = CountZone("entrance", vertices=((100,100),(400,100),(400,400),(100,400)))
    counter = ObjectCounter(zones=[zone])

    with InferenceEngine() as engine:
        for det in engine.stream(open_source("video.mp4")):
            result = counter.update(det)
            print(result.cumulative_counts)
"""

from __future__ import annotations

from yowo.counter._counter import ObjectCounter
from yowo.counter._types import (
    CountLine,
    CountResult,
    CountZone,
    CrossDirection,
    LineCrossEvent,
)

__all__ = [
    "CountLine",
    "CountResult",
    "CountZone",
    "CrossDirection",
    "LineCrossEvent",
    "ObjectCounter",
]
