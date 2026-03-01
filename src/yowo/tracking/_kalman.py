"""Kalman filter for bounding box tracking. State: [cx, cy, a, h, vx, vy, va, vh].

Canonical implementation from ifzhang/ByteTrack/yolox/tracker/kalman_filter.py.
State vector uses center-x, center-y, aspect-ratio, height, and their velocities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_STD_WEIGHT_POSITION = 1.0 / 20
_STD_WEIGHT_VELOCITY = 1.0 / 160


class KalmanFilterXYAH:
    """Kalman filter for axis-aligned bounding boxes.

    State: [cx, cy, a, h, vx, vy, va, vh]
    Measurement: [cx, cy, a, h]

    where a = aspect ratio (w/h), h = height.
    """

    _motion_mat: NDArray[np.float64]  # 8x8 F matrix
    _update_mat: NDArray[np.float64]  # 4x8 H matrix

    def __init__(self) -> None:
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float64)

    def initiate(
        self, measurement: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Create track from an unassociated measurement.

        Args:
            measurement: (4,) [cx, cy, a, h] bounding box.

        Returns:
            (mean, covariance) initial state estimate.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        h = measurement[3]
        std = [
            2 * _STD_WEIGHT_POSITION * h,
            2 * _STD_WEIGHT_POSITION * h,
            1e-1,  # aspect ratio — loose for edge-entering objects
            2 * _STD_WEIGHT_POSITION * h,
            10 * _STD_WEIGHT_VELOCITY * h,
            10 * _STD_WEIGHT_VELOCITY * h,
            1e-3,  # aspect ratio velocity — allow rapid convergence
            10 * _STD_WEIGHT_VELOCITY * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run Kalman filter prediction step.

        Args:
            mean: Current state mean (8,).
            covariance: Current state covariance (8, 8).

        Returns:
            (predicted_mean, predicted_covariance).
        """
        h = mean[3]
        std_pos = [
            _STD_WEIGHT_POSITION * h,
            _STD_WEIGHT_POSITION * h,
            1e-2,
            _STD_WEIGHT_POSITION * h,
        ]
        std_vel = [
            _STD_WEIGHT_VELOCITY * h,
            _STD_WEIGHT_VELOCITY * h,
            1e-5,
            _STD_WEIGHT_VELOCITY * h,
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(
        self,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project state distribution to measurement space.

        Args:
            mean: State mean (8,).
            covariance: State covariance (8, 8).

        Returns:
            (projected_mean, projected_covariance) in measurement space.
        """
        h = mean[3]
        std = [
            _STD_WEIGHT_POSITION * h,
            _STD_WEIGHT_POSITION * h,
            1e-1,
            _STD_WEIGHT_POSITION * h,
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return projected_mean, projected_cov

    def update(
        self,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
        measurement: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run Kalman filter correction step.

        Uses Cholesky decomposition for numerically stable gain computation.

        Args:
            mean: Predicted state mean (8,).
            covariance: Predicted state covariance (8, 8).
            measurement: Observed measurement (4,).

        Returns:
            (updated_mean, updated_covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain via Cholesky solve: K = P H^T (H P H^T + R)^{-1}
        # = (covariance @ _update_mat.T) @ projected_cov^{-1}
        # Implemented via Cholesky factorisation for numerical stability.
        # Fall back to direct inversion when Cholesky fails (near-degenerate covariance
        # from zero-area boxes or floating-point drift over many frames).
        rhs = self._update_mat @ covariance.T
        try:
            chol = np.linalg.cholesky(projected_cov)
            kalman_gain = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs)).T
        except np.linalg.LinAlgError:
            kalman_gain = (np.linalg.inv(projected_cov + 1e-6 * np.eye(4)) @ rhs).T

        innovation = measurement - projected_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    @staticmethod
    def xyxy_to_xyah(xyxy: tuple[float, float, float, float]) -> NDArray[np.float64]:
        """Convert xyxy bounding box to [cx, cy, a, h] representation.

        Args:
            xyxy: (x1, y1, x2, y2) bounding box.

        Returns:
            (4,) array [cx, cy, aspect_ratio, height].
        """
        x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        a = w / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float64)

    @staticmethod
    def xyah_to_xyxy(xyah: NDArray[np.float64]) -> tuple[float, float, float, float]:
        """Convert [cx, cy, a, h] representation back to xyxy bounding box.

        Args:
            xyah: (4,) array [cx, cy, aspect_ratio, height].

        Returns:
            (x1, y1, x2, y2) bounding box.
        """
        cx, cy, a, h = float(xyah[0]), float(xyah[1]), float(xyah[2]), float(xyah[3])
        w = a * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


__all__ = ["KalmanFilterXYAH"]
