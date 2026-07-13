"""Experimental repair for a DPVO gauge change caused by global BA.

DPVO global bundle adjustment can normalize/re-scale its active trajectory.
The production alignment is learned in the pre-BA raw coordinate system, so a
closure can otherwise become a kilometre-scale NED jump. This subclass consumes
an *exact* pre/post snapshot emitted by DPVO at the BA boundary, fits the
post-gauge -> pre-gauge Sim3, then composes it with the frozen NED alignment.

It is deliberately loaded only with ``--positioner-path`` from the experiment
runner; it is not part of the live positioning path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.positioning_dpvo import PositioningDPVO


class ExperimentalPositioningDPVO(PositioningDPVO):
    """Fail-closed gauge repair for the loop-closure experiment only."""

    supports_loop_gauge_repair = True
    MIN_PAIRS = 8
    MIN_RELATIVE_SPAN = 1e-3
    MIN_RAW_SPAN = 1e-2
    MAX_GAUGE_SCALE = 20.0
    MAX_RELATIVE_FIT_RMSE = 0.03
    MAX_RELATIVE_MAX_RESIDUAL = 0.10
    MAX_CONTINUITY_OFFSET_M = 50.0

    def __init__(self, config: dict):
        super().__init__(config)
        if self.fit_method not in {"linear", "ridge", "sim3"}:
            raise ValueError(
                "closure-aware deney yalnız absolute linear/ridge/sim3 "
                f"hizalamasını destekler; gelen={self.fit_method!r}"
            )
        self.closure_events: list[dict[str, Any]] = []
        self._closure_blocked = False

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        """Convert NumPy scalars used by telemetry to JSON-safe primitives."""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _pop_gauge_events(self) -> list[dict]:
        """Consume every completed gauge event without touching the DPVO graph."""
        if self.slam is None:
            return []
        pop_event = getattr(self.slam, "pop_gauge_event", None)
        if pop_event is None:
            pop_event = getattr(self.slam, "pop_global_ba_event", None)
        if pop_event is None:
            return []
        events: list[dict] = []
        # DPVO's queue is bounded to eight events. This loop is intentionally
        # bounded as well so a broken wrapper cannot hold a frame forever.
        for _ in range(8):
            try:
                event = pop_event()
            except Exception as exc:
                self.logger.warning("Global BA olayı okunamadı: %s", exc)
                break
            if event is None:
                break
            events.append(event)
        return events

    def _fit_event_gauge(self, event: dict, *, frame_idx: int) -> tuple[dict | None, dict]:
        """Fit ``post_c2w -> pre_c2w`` Sim3 and return its safe telemetry."""
        summary: dict[str, Any] = {
            "event_id": self._json_scalar(event.get("event_id")),
            "kind": str(event.get("kind", "global_ba")),
            "event_reason": self._json_scalar(event.get("reason")),
            "frame_idx": int(frame_idx),
            "trigger_input_timestamp": self._json_scalar(
                event.get("trigger_input_timestamp")
            ),
            "n_active": self._json_scalar(event.get("n_active")),
            "n_full_edges": self._json_scalar(event.get("n_full_edges")),
            "status": "rejected",
        }
        try:
            old_raw = np.asarray(event["before_c2w_xyz"], dtype=np.float64)
            new_raw = np.asarray(event["after_c2w_xyz"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            summary["reason"] = f"invalid_ba_snapshot: {exc}"
            return None, summary

        if (
            old_raw.shape != new_raw.shape
            or old_raw.ndim != 2
            or old_raw.shape[1:] != (3,)
            or len(new_raw) < self.MIN_PAIRS
            or not np.all(np.isfinite(old_raw))
            or not np.all(np.isfinite(new_raw))
        ):
            summary.update(
                {"pair_count": int(len(new_raw)) if new_raw.ndim else 0,
                 "reason": "invalid_or_insufficient_ba_pairs"}
            )
            return None, summary

        centered_new = new_raw - new_raw.mean(axis=0)
        try:
            singular_values = np.linalg.svd(centered_new, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            summary["reason"] = f"gauge_geometry_svd_failed: {exc}"
            return None, summary
        raw_span = float(np.linalg.norm(old_raw - old_raw.mean(axis=0), axis=1).mean())
        relative_span = float(
            singular_values[-1] / max(float(singular_values[0]), 1e-12)
        )
        summary.update(
            {
                "pair_count": int(len(new_raw)),
                "raw_span": raw_span,
                "relative_3d_span": relative_span,
            }
        )
        if raw_span < self.MIN_RAW_SPAN or relative_span < self.MIN_RELATIVE_SPAN:
            summary["reason"] = "gauge_pairs_geometrically_degenerate"
            return None, summary

        try:
            rotation, translation, scale = self._sim3_umeyama(new_raw, old_raw)
        except Exception as exc:
            summary["reason"] = f"sim3_fit_failed: {exc}"
            return None, summary

        reconstructed_old = (scale * (rotation @ new_raw.T)).T + translation
        residuals = np.linalg.norm(reconstructed_old - old_raw, axis=1)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        relative_rmse = float(rmse / max(raw_span, 1e-12))
        relative_max_residual = float(np.max(residuals) / max(raw_span, 1e-12))
        summary.update(
            {
                "gauge_scale_new_to_old": float(scale),
                "raw_fit_rmse": rmse,
                "raw_fit_relative_rmse": relative_rmse,
                "raw_fit_relative_max_residual": relative_max_residual,
            }
        )
        if not (1.0 / self.MAX_GAUGE_SCALE <= scale <= self.MAX_GAUGE_SCALE):
            summary["reason"] = "gauge_scale_out_of_bounds"
            return None, summary
        if relative_rmse > self.MAX_RELATIVE_FIT_RMSE:
            summary["reason"] = "gauge_sim3_fit_too_noisy"
            return None, summary
        if relative_max_residual > self.MAX_RELATIVE_MAX_RESIDUAL:
            summary["reason"] = "gauge_sim3_outlier_residual"
            return None, summary

        return {
            "rotation": rotation,
            "translation": translation,
            "scale": float(scale),
        }, summary

    def _compose_alignment(self, gauge: dict) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the old NED mapping rewritten for the new DPVO gauge."""
        rotation = np.asarray(gauge["rotation"], dtype=np.float64)
        translation = np.asarray(gauge["translation"], dtype=np.float64)
        scale = float(gauge["scale"])
        old_rotation = np.asarray(self.sim3_R, dtype=np.float64)
        old_translation = np.asarray(self.sim3_t, dtype=np.float64)
        old_scale = float(self.sim3_s)
        if (
            old_rotation.shape != (3, 3)
            or old_translation.shape != (3,)
            or not np.all(np.isfinite(old_rotation))
            or not np.all(np.isfinite(old_translation))
            or not np.isfinite(old_scale)
        ):
            raise ValueError("existing alignment is invalid")

        # old_raw = scale * rotation * new_raw + translation
        # NED = old_scale * old_rotation * old_raw + old_translation + anchor
        new_translation = old_scale * (old_rotation @ translation) + old_translation
        if self.fit_method == "sim3":
            # Preserve the explicit Sim3 state when the original calibration
            # genuinely was Sim3.
            new_rotation = old_rotation @ rotation
            new_scale = old_scale * scale
        else:
            # ``linear``/``ridge`` already use sim3_R as a general affine
            # matrix. Store the composed affine map without pretending it is
            # an orthonormal rotation.
            new_rotation = old_scale * old_rotation @ (scale * rotation)
            new_scale = 1.0
        return new_rotation, new_translation, new_scale

    def _commit_alignment(
        self, rotation: np.ndarray, translation: np.ndarray, scale: float
    ) -> None:
        self.sim3_R = np.asarray(rotation, dtype=np.float64).copy()
        self.sim3_t = np.asarray(translation, dtype=np.float64).copy()
        self.sim3_s = float(scale)

    def _repair_output_gauge(
        self, gauge: dict, summary: dict, *, before_ned: np.ndarray
    ) -> tuple[bool, dict]:
        """Compose a gauge map and preserve the current NED output continuity."""
        if self.last_dpvo_raw is None:
            summary["reason"] = "missing_current_raw_pose"
            return False, summary
        try:
            new_rotation, new_translation, new_scale = self._compose_alignment(gauge)
            raw = np.asarray(self.last_dpvo_raw, dtype=np.float64)
            candidate = (
                new_scale * (new_rotation @ raw)
                + new_translation
                + self.alignment_anchor_offset
            )
        except Exception as exc:
            summary["reason"] = f"alignment_composition_failed: {exc}"
            return False, summary

        continuity_offset = np.asarray(before_ned, dtype=np.float64) - candidate
        continuity_norm = float(np.linalg.norm(continuity_offset))
        summary["continuity_offset_m"] = continuity_offset.astype(float).tolist()
        summary["continuity_offset_norm_m"] = continuity_norm
        if not np.all(np.isfinite(candidate)) or continuity_norm > self.MAX_CONTINUITY_OFFSET_M:
            summary["reason"] = "continuity_correction_out_of_bounds"
            return False, summary

        self._commit_alignment(new_rotation, new_translation, new_scale)
        self.alignment_anchor_offset += continuity_offset
        repaired = self._align_dpvo_to_gt(raw)
        self.last_known_position = repaired.copy()
        self.last_velocity = repaired - before_ned
        # The raw delta spanning pre/post gauges is invalid. Subsequent raw
        # deltas are all in the new gauge and can accumulate normally.
        self.raw_delta_history = []
        summary.update(
            {
                "status": "repaired",
                "repaired_ned": repaired.astype(float).tolist(),
            }
        )
        return True, summary

    def _rebase_calibration_buffer(self, gauge: dict, event: dict) -> int:
        """Move pre-BA calibration raw points into the post-BA gauge.

        This path matters only if a BA occurs while health=1. The main output
        is GT in that phase, but without rebasing the next online fit would mix
        coordinate gauges.
        """
        trigger = event.get("trigger_input_timestamp")
        try:
            trigger_value = float(trigger)
        except (TypeError, ValueError):
            raise ValueError("global BA event has no numeric input timestamp")

        rotation = np.asarray(gauge["rotation"], dtype=np.float64)
        translation = np.asarray(gauge["translation"], dtype=np.float64)
        scale = float(gauge["scale"])
        changed = 0
        rebased: list[list[float]] = []
        for index, raw in zip(self.frame_indices, self.dpvo_buffer):
            raw_array = np.asarray(raw, dtype=np.float64)
            if float(index) < trigger_value:
                # Inverse of old_raw = s * R * new_raw + t.
                raw_array = (rotation.T @ (raw_array - translation)) / scale
                changed += 1
            rebased.append(raw_array.astype(float).tolist())
        self.dpvo_buffer = rebased
        self.raw_delta_history = []
        self._calibrated_sample_count = -1
        return changed

    def _hold_after_rejected_closure(
        self, before_ned: np.ndarray, summary: dict
    ) -> tuple[float, float, float]:
        """Fail closed instead of emitting a stale-gauge NED coordinate."""
        self._closure_blocked = True
        self.last_known_position = np.asarray(before_ned, dtype=np.float64).copy()
        self.last_velocity = np.zeros(3, dtype=np.float64)
        self.raw_delta_history = []
        summary["status"] = "blocked"
        summary["held_ned"] = self.last_known_position.astype(float).tolist()
        return tuple(float(value) for value in self.last_known_position)

    def process_frame(self, *args, **kwargs) -> tuple:
        frame_idx = int(kwargs.get("frame_idx", args[0] if args else -1))
        health_status = kwargs.get("health_status", args[2] if len(args) > 2 else None)
        status = None if health_status is None else str(health_status)
        before_ned = self.last_known_position.copy()
        result = super().process_frame(*args, **kwargs)
        events = self._pop_gauge_events()

        if status == "1":
            # Output is GT during calibration, so safely rewrite the raw
            # buffer then force a clean fit in the new gauge.
            for event in events:
                gauge, summary = self._fit_event_gauge(event, frame_idx=frame_idx)
                if gauge is None:
                    self.closure_events.append(summary)
                    continue
                try:
                    rotation, translation, scale = self._compose_alignment(gauge)
                    self._commit_alignment(rotation, translation, scale)
                    summary["rebased_calibration_points"] = self._rebase_calibration_buffer(
                        gauge, event
                    )
                    self._update_calibration()
                    summary["status"] = "calibration_rebased"
                    self._closure_blocked = False
                except Exception as exc:
                    summary["reason"] = f"calibration_rebase_failed: {exc}"
                self.closure_events.append(summary)
            return result

        if status == "0":
            for event in events:
                gauge, summary = self._fit_event_gauge(event, frame_idx=frame_idx)
                if not self.is_calibrated:
                    summary.update({"status": "skipped_uncalibrated", "reason": "alignment_not_ready"})
                    self.closure_events.append(summary)
                    continue
                if gauge is None:
                    result = self._hold_after_rejected_closure(before_ned, summary)
                    self.closure_events.append(summary)
                    continue
                repaired, summary = self._repair_output_gauge(
                    gauge, summary, before_ned=before_ned
                )
                if repaired:
                    result = tuple(float(value) for value in self.last_known_position)
                    self.logger.info(
                        "Closure gauge onarıldı: event=%s frame=%d pairs=%d "
                        "scale=%.6f fit_rel=%.6f continuity=%.3fm",
                        summary.get("event_id"),
                        frame_idx,
                        summary.get("pair_count", 0),
                        summary.get("gauge_scale_new_to_old", float("nan")),
                        summary.get("raw_fit_relative_rmse", float("nan")),
                        summary.get("continuity_offset_norm_m", float("nan")),
                    )
                else:
                    result = self._hold_after_rejected_closure(before_ned, summary)
                    self.logger.warning(
                        "Closure gauge onarımı reddedildi; güvenli hold etkin: "
                        "event=%s frame=%d reason=%s",
                        summary.get("event_id"),
                        frame_idx,
                        summary.get("reason"),
                    )
                self.closure_events.append(summary)

            if self._closure_blocked:
                # ``super`` has already updated its stale mapping for this
                # frame. Restore the last trusted NED point and keep holding
                # until a future health=1 calibration phase can re-establish
                # a known coordinate gauge.
                result = self._hold_after_rejected_closure(
                    before_ned,
                    {
                        "frame_idx": frame_idx,
                        "status": "blocked",
                        "reason": "awaiting_gt_recalibration",
                    },
                )
        return result
