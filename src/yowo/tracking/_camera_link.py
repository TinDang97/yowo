"""Camera link model for spatial-temporal pruning of cross-camera matches."""

from __future__ import annotations

from dataclasses import dataclass

from yowo.tracking._gallery import GalleryMatch

__all__ = ["CameraLink", "CameraLinkModel"]


@dataclass(frozen=True)
class CameraLink:
    """Defines a spatial-temporal transition between two cameras.

    Links are directional: (A→B, 10-30s) does NOT imply (B→A, 10-30s).

    Args:
        src_camera: Source camera identifier.
        dst_camera: Destination camera identifier.
        min_transit_sec: Minimum travel time between cameras.
        max_transit_sec: Maximum travel time between cameras.
    """

    src_camera: str
    dst_camera: str
    min_transit_sec: float
    max_transit_sec: float


class CameraLinkModel:
    """Prunes cross-camera matches using spatial-temporal constraints.

    If a vehicle exits Camera A at time T, it can only appear in Camera B
    between T + min_transit and T + max_transit. Transitions outside this
    window are rejected, eliminating 90%+ of false cross-camera matches.

    When no link is configured for a camera pair, a permissive default
    window is used (graceful degradation).
    """

    __slots__ = ("_default_window", "_links")

    def __init__(
        self,
        links: list[CameraLink] | None = None,
        default_window: tuple[float, float] = (5.0, 120.0),
    ) -> None:
        """Initialize camera link model.

        Args:
            links: List of camera-to-camera transition constraints.
            default_window: (min_sec, max_sec) for unconfigured camera pairs.
        """
        self._links: dict[tuple[str, str], CameraLink] = {}
        self._default_window = default_window
        if links:
            for link in links:
                key = (link.src_camera, link.dst_camera)
                self._links[key] = link

    def is_feasible(
        self,
        src_camera: str,
        dst_camera: str,
        src_exit_time: float,
        dst_entry_time: float,
    ) -> bool:
        """Check if a transition is temporally feasible.

        Args:
            src_camera: Camera where vehicle was last seen.
            dst_camera: Camera where vehicle is now detected.
            src_exit_time: Timestamp when vehicle exited src_camera.
            dst_entry_time: Timestamp when vehicle appeared in dst_camera.

        Returns:
            True if the transit time falls within the allowed window.
        """
        if src_camera == dst_camera:
            return False

        transit = dst_entry_time - src_exit_time
        if transit < 0:
            return False

        key = (src_camera, dst_camera)
        link = self._links.get(key)
        if link is not None:
            return link.min_transit_sec <= transit <= link.max_transit_sec

        min_sec, max_sec = self._default_window
        return min_sec <= transit <= max_sec

    def filter_matches(
        self,
        query_camera: str,
        query_time: float,
        matches: list[GalleryMatch],
    ) -> list[GalleryMatch]:
        """Remove infeasible matches based on camera link constraints.

        Args:
            query_camera: Camera where the query track is active.
            query_time: Current timestamp of the query.
            matches: Candidate gallery matches to filter.

        Returns:
            Filtered list with only temporally feasible matches.
        """
        return [
            m
            for m in matches
            if self.is_feasible(
                src_camera=m.camera_id,
                dst_camera=query_camera,
                src_exit_time=m.timestamp,
                dst_entry_time=query_time,
            )
        ]

    @property
    def link_count(self) -> int:
        """Number of configured camera links."""
        return len(self._links)
