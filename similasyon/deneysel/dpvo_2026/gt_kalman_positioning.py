"""GT-gated position Kalman experiment for DPVO.

This class intentionally treats health=0 GT as unavailable.  The static
DPVO->NED scale is still learned by ``PositioningDPVO`` from health=1 pairs;
the Kalman layer only smooths the resulting NED measurement and re-anchors
when a legitimate later health=1 observation arrives.  It cannot manufacture
an absolute correction while no external observation exists.
"""

from __future__ import annotations

import numpy as np

from src.models.positioning_dpvo import PositioningDPVO


class ExperimentalPositioningDPVO(PositioningDPVO):
    """Constant-velocity Kalman filter with GT updates only at health=1."""

    def __init__(self, config: dict):
        super().__init__(config)
        kalman = config.get("experimental_kalman", {})
        self.kalman_dt = max(1e-4, float(kalman.get("dt", 1.0 / 7.5)))
        self.kalman_process_noise = max(0.0, float(kalman.get("process_noise", 0.1)))
        self.kalman_vo_measurement_noise = max(
            1e-8, float(kalman.get("vo_measurement_noise", 1.0))
        )
        self.kalman_gt_measurement_noise = max(
            1e-10, float(kalman.get("gt_measurement_noise", 1e-4))
        )
        self.kalman_enabled = bool(kalman.get("enabled", True))
        self._kf_initialized = False
        self._kf_x = np.zeros(6, dtype=np.float64)  # [position(3), velocity(3)]
        self._kf_P = np.eye(6, dtype=np.float64) * 100.0
        self.kalman_updates_gt = 0
        self.kalman_updates_vo = 0

        dt = self.kalman_dt
        self._F = np.block(
            [[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
        )
        self._H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self._I = np.eye(6)
        self._Q = self.kalman_process_noise * np.block(
            [
                [(dt**4 / 4.0) * np.eye(3), (dt**3 / 2.0) * np.eye(3)],
                [(dt**3 / 2.0) * np.eye(3), (dt**2) * np.eye(3)],
            ]
        )

    def _initialize_filter(self, position: np.ndarray) -> None:
        self._kf_x[:] = 0.0
        self._kf_x[:3] = position
        self._kf_P = np.eye(6, dtype=np.float64) * 100.0
        self._kf_initialized = True

    def _predict(self) -> None:
        self._kf_x = self._F @ self._kf_x
        self._kf_P = self._F @ self._kf_P @ self._F.T + self._Q

    def _update(self, measurement: np.ndarray, variance: float) -> None:
        R = np.eye(3, dtype=np.float64) * variance
        innovation = measurement - self._H @ self._kf_x
        innovation_covariance = self._H @ self._kf_P @ self._H.T + R
        gain = self._kf_P @ self._H.T @ np.linalg.inv(innovation_covariance)
        self._kf_x = self._kf_x + gain @ innovation
        # Joseph form preserves positive semidefiniteness under finite
        # precision and makes telemetry covariance meaningful.
        residual = self._I - gain @ self._H
        self._kf_P = residual @ self._kf_P @ residual.T + gain @ R @ gain.T

    def process_frame(self, *args, **kwargs) -> tuple:
        health_status = kwargs.get("health_status", args[2] if len(args) > 2 else None)
        status = None if health_status is None else str(health_status)
        result = super().process_frame(*args, **kwargs)
        measurement = np.asarray(result, dtype=np.float64)
        if not self.kalman_enabled or not np.all(np.isfinite(measurement)):
            return result

        if status == "1":
            # Base class returns exactly the legitimate GT payload here. Do not
            # smooth or alter it; use it only to update/re-anchor the filter.
            if not self._kf_initialized:
                self._initialize_filter(measurement)
            else:
                self._predict()
                self._update(measurement, self.kalman_gt_measurement_noise)
            self._kf_x[:3] = measurement
            self.last_known_position = measurement.copy()
            self.kalman_updates_gt += 1
            return result

        if status == "0":
            # ``measurement`` is a DPVO-derived NED estimate. The evaluator
            # masks health=0 GT before it reaches this method.
            if not self._kf_initialized:
                self._initialize_filter(measurement)
            else:
                self._predict()
                self._update(measurement, self.kalman_vo_measurement_noise)
            filtered = self._kf_x[:3].copy()
            self.last_velocity = self._kf_x[3:].copy() * self.kalman_dt
            self.last_known_position = filtered.copy()
            self.kalman_updates_vo += 1
            return tuple(float(value) for value in filtered)
        return result

    @property
    def kalman_telemetry(self) -> dict:
        return {
            "enabled": self.kalman_enabled,
            "gt_updates": self.kalman_updates_gt,
            "vo_updates": self.kalman_updates_vo,
            "covariance_trace": float(np.trace(self._kf_P)),
            "dt": self.kalman_dt,
            "process_noise": self.kalman_process_noise,
            "vo_measurement_noise": self.kalman_vo_measurement_noise,
            "gt_measurement_noise": self.kalman_gt_measurement_noise,
        }
