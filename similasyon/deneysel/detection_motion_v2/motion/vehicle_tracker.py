"""Arac tracker ve moving_status v2 (Faz M0-M3).

M0 P0 kusur düzeltmeleri:
1. Kamera transform track.center (son gozlem) uzerine uygulanir
2. Kalman predict() sonucu association icin kullanilir
3. IoU, warped predicted bbox ile hesaplanir
4. Missed track her kare current frame koordinatina tasinir
5. Birden fazla missed frame icin kamera transform zinciri korunur
6. Predict/update kare basina tam birer kez cagirilir

M2 Kalman ve association v2:
- Box state: [cx, cy, w, h, vx, vy, vw, vh]
- Mahalanobis gating
- ByteTrack tarzi iki asamali confidence association
- Dynamic gate: bbox diagonal/uncertainty tabanli

M3 Normalize hareket kaniti:
- residual_px / max(bbox_diagonal, min_diagonal) / delta_time
- Local optical flow kaniti (opsiyonel)
- Kalite agirlikli birlesim
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

from .camera_motion import CameraTransform, CameraModel

logger = logging.getLogger(__name__)


def _linear_assignment(cost_matrix: np.ndarray) -> list:
    """Hungarian assignment; scipy yoksa greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        matches = []
        used_rows = set()
        used_cols = set()
        flat = [
            (float(cost_matrix[r, c]), r, c)
            for r in range(cost_matrix.shape[0])
            for c in range(cost_matrix.shape[1])
        ]
        for _, r, c in sorted(flat, key=lambda x: x[0]):
            if r in used_rows or c in used_cols:
                continue
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
        return matches


def _box_iou(box1, box2) -> float:
    """Iki kutu arasi IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return float(inter / (area1 + area2 - inter))


def _bbox_diagonal(bbox: tuple) -> float:
    """Bbox diagonal uzunlugu."""
    w = max(1e-6, bbox[2] - bbox[0])
    h = max(1e-6, bbox[3] - bbox[1])
    return float(np.sqrt(w * w + h * h))


def _center_from_bbox(bbox: tuple) -> np.ndarray:
    """Bbox merkez noktasi."""
    return np.array([
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ], dtype=np.float32)


def _wh_from_bbox(bbox: tuple) -> tuple:
    """Bbox width ve height."""
    return (max(1e-6, float(bbox[2] - bbox[0])),
            max(1e-6, float(bbox[3] - bbox[1])))


def _bbox_from_cxywh(cx: float, cy: float, w: float, h: float) -> tuple:
    """cx, cy, w, h -> (x1, y1, x2, y2)."""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class VehicleTrackV2:
    """Tek arac track'inin durumu.

    8D Kalman box state: [cx, cy, w, h, vx, vy, vw, vh]
    Sabit hiz modeli: her boyut icin pozisyon + hiz.
    """

    def __init__(self, track_id: int, bbox: tuple, frame_idx: int,
                 config: dict):
        self.id = track_id
        self.frame_idx = frame_idx

        cx, cy = _center_from_bbox(bbox)
        w, h = _wh_from_bbox(bbox)

        self.bbox = bbox
        self.center = np.array([cx, cy], dtype=np.float32)
        self.wh = np.array([w, h], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_status = "0"
        self.last_frame = frame_idx

        self.motion_scores = []
        self.normalized_scores = []

        motion_cfg = config.get("motion", {})
        self._max_age = int(motion_cfg.get("max_track_age", 15))

        self.kalman_enabled = bool(motion_cfg.get("kalman_enabled", True))
        pn = float(motion_cfg.get("kalman_process_noise", 1.0))
        mn = float(motion_cfg.get("kalman_measurement_noise", 10.0))
        self.motion_window = int(motion_cfg.get("motion_window", 6))
        self.decision_window = int(motion_cfg.get("decision_window", 5))
        self.moving_votes_required = int(motion_cfg.get("moving_votes_required", 3))
        self.min_track_len = int(motion_cfg.get("min_track_len", 4))
        self.moving_start_threshold = float(motion_cfg.get("moving_start_threshold", 0.15))
        self.moving_stop_threshold = float(motion_cfg.get("moving_stop_threshold", 0.08))
        self.normalize_by_bbox_diagonal = bool(motion_cfg.get("normalize_by_bbox_diagonal", True))
        self.min_diagonal = float(motion_cfg.get("min_diagonal", 20.0))

        self._init_kalman(cx, cy, w, h, pn, mn)

    def _init_kalman(self, cx, cy, w, h, process_noise, meas_noise):
        """8D Kalman filter baslat: [cx, cy, w, h, vx, vy, vw, vh]."""
        if not self.kalman_enabled:
            self.kf = None
            return

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.F[i, i + 4] = dt

        self.kf.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.H[i, i] = 1.0

        self.kf.P = np.eye(8, dtype=np.float32) * 100.0
        self.kf.P[4:, 4:] *= 10.0

        self.kf.R = np.eye(4, dtype=np.float32) * meas_noise

        q = process_noise
        self.kf.Q = np.eye(8, dtype=np.float32) * q
        for i in range(4):
            self.kf.Q[i, i + 4] = q * 0.5
            self.kf.Q[i + 4, i] = q * 0.5
            self.kf.Q[i + 4, i + 4] = q

        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

    def predict(self, camera_transform: Optional[CameraTransform] = None) -> dict:
        """Kalman predict + kamera kompanzasyonu.

        Iki ayri center dondurur:
        - camera_expected_center: self.center uzerine kamera transformu (hareketsiz varsayimi)
        - association_center: Kalman predicted center uzerine kamera transformu (hiz ile)

        Returns:
            {"camera_expected_center", "association_center", "bbox", "wh"}
        """
        if self.kf is not None:
            self.kf.predict()

        if self.kf is not None:
            kalman_center = self.kf.x[:2].copy()
            pred_wh = self.kf.x[2:4].copy()
        else:
            kalman_center = self.center + self.velocity
            pred_wh = self.wh.copy()

        if camera_transform is not None and camera_transform.is_reliable:
            camera_expected_center = camera_transform.apply_to_point(self.center)
            association_predicted_center = camera_transform.apply_to_point(kalman_center)
            warped_bbox = camera_transform.apply_to_bbox(self.bbox)
            pred_bbox = warped_bbox
            warped_wh = np.array([
                warped_bbox[2] - warped_bbox[0],
                warped_bbox[3] - warped_bbox[1],
            ], dtype=np.float32)
            pred_wh = warped_wh
        else:
            camera_expected_center = self.center.copy()
            association_predicted_center = kalman_center
            cx, cy = float(association_predicted_center[0]), float(association_predicted_center[1])
            w, h = float(pred_wh[0]), float(pred_wh[1])
            pred_bbox = _bbox_from_cxywh(cx, cy, w, h)

        self.age += 1
        self.time_since_update += 1

        return {
            "camera_expected_center": camera_expected_center,
            "association_center": association_predicted_center,
            "bbox": pred_bbox,
            "wh": pred_wh,
        }

    def update(self, bbox: tuple, frame_idx: int,
               camera_transform: Optional[CameraTransform] = None,
               det_center: Optional[np.ndarray] = None):
        """Track'i yeni gozlemle guncelle.

        M0 düzeltme: predict/update kare basina tam birer kez cagirilir.
        """
        if det_center is None:
            det_center = _center_from_bbox(bbox)
        w, h = _wh_from_bbox(bbox)

        self.bbox = bbox
        self.wh = np.array([w, h], dtype=np.float32)

        if self.kf is not None:
            z = np.array([det_center[0], det_center[1], w, h], dtype=np.float32)
            self.kf.update(z)
            self.velocity = self.kf.x[4:6].copy()
        else:
            old_center = self.center.copy()
            new_v = det_center - old_center
            self.velocity = 0.7 * self.velocity + 0.3 * new_v

        self.center = det_center.copy()

        self.hits += 1
        self.age = 0
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.frame_idx = frame_idx

    def compute_motion_score(
        self,
        det_center: np.ndarray,
        pred_center: np.ndarray,
        camera_transform: Optional[CameraTransform] = None,
    ) -> float:
        """Normalize hareket skoru hesapla (Faz M3).

        residual_px = norm(det_center - camera_expected_center)
        normalized_residual = residual_px / max(bbox_diagonal, min_diagonal) / delta_time

        delta_time = 1 frame (simdilik)
        """
        residual_px = float(np.linalg.norm(det_center - pred_center))

        if self.normalize_by_bbox_diagonal:
            diag = max(_bbox_diagonal(self.bbox), self.min_diagonal)
            normalized = residual_px / diag
        else:
            normalized = residual_px

        self.motion_scores.append(residual_px)
        self.normalized_scores.append(normalized)

        if len(self.motion_scores) > self.motion_window:
            self.motion_scores.pop(0)
            self.normalized_scores.pop(0)

        if len(self.normalized_scores) >= self.decision_window:
            votes = sum(1 for s in self.normalized_scores[-self.decision_window:]
                        if s > self.moving_start_threshold)
            if votes >= self.moving_votes_required:
                return float(np.median(self.normalized_scores[-self.decision_window:]))
            votes_stop = sum(1 for s in self.normalized_scores[-self.decision_window:]
                             if s <= self.moving_stop_threshold)
            if votes_stop >= self.moving_votes_required:
                return float(np.median(self.normalized_scores[-self.decision_window:]))

        return float(np.median(self.normalized_scores)) if self.normalized_scores else normalized

    def decide_moving_status(self, score: float) -> str:
        """Histerzisli durum karari (normalize skor uzerinden)."""
        if self.hits < self.min_track_len:
            return "0"

        votes_moving = sum(
            1 for s in self.normalized_scores[-self.decision_window:]
            if s > self.moving_start_threshold
        )
        votes_stationary = sum(
            1 for s in self.normalized_scores[-self.decision_window:]
            if s <= self.moving_stop_threshold
        )

        if self.last_status == "1":
            if votes_stationary >= self.moving_votes_required:
                return "0"
            return "1"
        else:
            if votes_moving >= self.moving_votes_required:
                return "1"
            return "0"

    def propagate_missed(self, camera_transform: Optional[CameraTransform]):
        """Missed track'i current frame'e tasi (M0 düzeltme #4/#5).

        Kamera transform zinciri korunur.
        Track.center ve track.bbox current frame koordinatina guncellenir.
        """
        if camera_transform is not None and camera_transform.is_reliable:
            self.center = camera_transform.apply_to_point(self.center)
            self.bbox = camera_transform.apply_to_bbox(self.bbox)
            if self.kf is not None:
                cx, cy = _center_from_bbox(self.bbox)
                w, h = _wh_from_bbox(self.bbox)
                self.kf.x[:4] = np.array([cx, cy, w, h], dtype=np.float32)

    def mahalanobis_distance(self, det_center: np.ndarray, pred_center: np.ndarray) -> float:
        """Mahalanobis distance using Kalman position covariance."""
        if self.kf is None:
            return float('inf')
        diff = (det_center - pred_center).reshape(2, 1)
        P_pos = self.kf.P[:2, :2].astype(np.float64)
        try:
            S = P_pos + np.eye(2) * 1e-4
            S_inv = np.linalg.inv(S)
            maha_sq = float((diff.T @ S_inv @ diff)[0, 0])
            return float(np.sqrt(max(0.0, maha_sq)))
        except np.linalg.LinAlgError:
            return float('inf')

    @property
    def is_active(self) -> bool:
        return self.time_since_update <= self._max_age

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.min_track_len


class VehicleTrackerV2:
    """Arac tracker v2 - M0 düzeltmeleri + M2 association + M3 normalize skor.

    Args:
        config: Deneysel config dict (motion section)
    """

    def __init__(self, config: dict):
        self.config = config
        motion_cfg = config.get("motion", {})

        self.max_track_age = int(motion_cfg.get("max_track_age", 15))
        self.track_match_distance = float(motion_cfg.get("track_match_distance", 90.0))
        self.high_conf_threshold = float(motion_cfg.get("high_conf_threshold", 0.40))
        self.low_conf_threshold = float(motion_cfg.get("low_conf_threshold", 0.15))
        self.two_stage_association = bool(motion_cfg.get("two_stage_association", True))
        self.dynamic_gating = bool(motion_cfg.get("dynamic_gating", True))
        self.gate_diagonal_ratio = float(motion_cfg.get("gate_diagonal_ratio", 1.5))
        self.maha_gate_threshold = float(motion_cfg.get("maha_gate_threshold", 9.49))

        self.tracks: OrderedDict[int, VehicleTrackV2] = OrderedDict()
        self.next_id = 0
        self.frame_count = 0
        self.prev_gray = None

        self._max_age = self.max_track_age

        logger.info(
            "VehicleTrackerV2: max_track_age=%d, two_stage=%s, dynamic_gating=%s",
            self.max_track_age, self.two_stage_association, self.dynamic_gating,
        )

    def _compute_gate(self, track: VehicleTrackV2) -> float:
        """Dynamic gate: bbox diagonal/uncertainty tabanli (M2)."""
        if not self.dynamic_gating:
            return self.track_match_distance
        diag = _bbox_diagonal(track.bbox)
        return max(self.track_match_distance, diag * self.gate_diagonal_ratio)

    def _build_cost(
        self,
        det_centers: list,
        det_bboxes: list,
        det_confs: list,
        tracks: list,
        predictions: list,
    ) -> np.ndarray:
        """Cost matrix: normalized center distance + IoU + size ratio."""
        n_dets = len(det_centers)
        n_tracks = len(tracks)
        cost = np.full((n_dets, n_tracks), 1e6, dtype=np.float32)

        for di in range(n_dets):
            det_center = det_centers[di]
            det_bbox = det_bboxes[di]
            det_area = max(1e-6, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
            for ti in range(n_tracks):
                tr = tracks[ti]
                pred = predictions[ti]
                pred_center = pred["association_center"]
                pred_bbox = pred["bbox"]

                dist = float(np.linalg.norm(det_center - pred_center))
                iou = _box_iou(det_bbox, pred_bbox)

                tr_area = max(1e-6, (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1]))
                ratio = max(det_area, tr_area) / min(det_area, tr_area) - 1.0
                size_penalty = min(ratio, 5.0)

                gate = self._compute_gate(tr)
                c = dist + (1.0 - iou) * 30.0 + size_penalty * 10.0

                maha = tr.mahalanobis_distance(det_center, pred_center)
                if maha > self.maha_gate_threshold:
                    c = 1e6
                if dist > gate and iou < 0.05:
                    c = 1e6

                cost[di, ti] = c

        return cost

    def update(
        self,
        detections: list,
        gray: np.ndarray,
        camera_transform: Optional[CameraTransform] = None,
    ) -> list:
        """Bir kare isle.

        M0 düzeltme akisi:
        1. Kare sayacini artir
        2. Arac detection'larini hazirla (cls == 0)
        3. Tum track'ler icin predict() cagir (kamera kompanzasyonu ile)
        4. Predicted center/bbox kullanarak association yap
        5. Eslesen track'leri update() ile guncelle
        6. Eslesmeyen missed track'leri propagate_missed() ile tasi
        7. Eslesmeyen detection'lara yeni track ac
        8. Stale track'leri temizle
        9. moving_status ata

        Args:
            detections: Detection dict listesi
            gray: Mevcut frame grayscale
            camera_transform: Kamera transformu (CameraMotionEstimator'dan)

        Returns:
            Güncellenmis detection listesi (moving_status eklenmis)
        """
        self.frame_count += 1
        self.prev_gray = gray

        # Arac detection'larini hazirla
        vehicle_dets = []
        for i, det in enumerate(detections):
            if det.get("cls") == 0:
                vehicle_dets.append((i, det))

        active_tracks = [tr for tr in self.tracks.values() if tr.time_since_update <= self.max_track_age]

        # M0 #6: Tum track'ler icin predict() tam birer kez cagir
        predictions = []
        for tr in active_tracks:
            pred = tr.predict(camera_transform)
            predictions.append(pred)

        # Association
        det_indices = [vd[0] for vd in vehicle_dets]
        det_centers = [_center_from_bbox(vd[1]["bbox"]) for vd in vehicle_dets]
        det_bboxes = [vd[1]["bbox"] for vd in vehicle_dets]
        det_confs = [vd[1].get("conf", 0.5) for vd in vehicle_dets]

        matched_det_set = set()
        matched_track_set = set()
        det_idx_to_track = {}

        if self.two_stage_association and vehicle_dets:
            # Stage 1: Yuksek confidence
            high_indices = [i for i, c in enumerate(det_confs) if c >= self.high_conf_threshold]
            low_indices = [i for i, c in enumerate(det_confs) if c < self.high_conf_threshold]

            if high_indices and active_tracks:
                high_centers = [det_centers[i] for i in high_indices]
                high_bboxes = [det_bboxes[i] for i in high_indices]
                high_confs = [det_confs[i] for i in high_indices]

                cost_h = self._build_cost(high_centers, high_bboxes, high_confs,
                                          active_tracks, predictions)
                for di, ti in _linear_assignment(cost_h):
                    if cost_h[di, ti] >= 1e6:
                        continue
                    real_di = high_indices[di]
                    det_idx = det_indices[real_di]
                    tr = active_tracks[ti]
                    prev_center = tr.center.copy()

                    tr.update(det_bboxes[real_di], self.frame_count,
                              camera_transform, det_centers[real_di])

                    camera_expected = predictions[ti]["camera_expected_center"]
                    score = tr.compute_motion_score(
                        det_centers[real_di], camera_expected, camera_transform,
                    )
                    status = tr.decide_moving_status(score)
                    tr.last_status = status

                    detections[det_idx]["_track_id"] = tr.id
                    detections[det_idx]["_motion_score"] = score
                    detections[det_idx]["moving_status"] = status

                    matched_det_set.add(real_di)
                    matched_track_set.add(ti)
                    det_idx_to_track[det_idx] = tr.id

            # Stage 2: Dusuk confidence (yalnizca mevcut track kurtarma)
            remaining_tracks = [active_tracks[ti] for ti in range(len(active_tracks))
                               if ti not in matched_track_set]
            remaining_preds = [predictions[ti] for ti in range(len(active_tracks))
                              if ti not in matched_track_set]

            if low_indices and remaining_tracks:
                low_centers = [det_centers[i] for i in low_indices if i not in matched_det_set]
                low_bboxes = [det_bboxes[i] for i in low_indices if i not in matched_det_set]
                low_confs = [det_confs[i] for i in low_indices if i not in matched_det_set]
                low_map = [i for i in low_indices if i not in matched_det_set]

                if low_centers and remaining_tracks:
                    cost_l = self._build_cost(low_centers, low_bboxes, low_confs,
                                               remaining_tracks, remaining_preds)
                    for di, ti in _linear_assignment(cost_l):
                        if cost_l[di, ti] >= 1e6:
                            continue
                        real_di = low_map[di]
                        det_idx = det_indices[real_di]
                        tr = remaining_tracks[ti]
                        prev_center = tr.center.copy()

                        tr.update(det_bboxes[real_di], self.frame_count,
                                  camera_transform, det_centers[real_di])

                        camera_expected = remaining_preds[ti]["camera_expected_center"]
                        score = tr.compute_motion_score(
                            det_centers[real_di], camera_expected,
                            camera_transform,
                        )
                        status = tr.decide_moving_status(score)
                        tr.last_status = status

                        detections[det_idx]["_track_id"] = tr.id
                        detections[det_idx]["_motion_score"] = score
                        detections[det_idx]["moving_status"] = status

                        matched_det_set.add(real_di)
                        matched_track_set.add(active_tracks.index(tr))
                        det_idx_to_track[det_idx] = tr.id

        elif vehicle_dets and active_tracks:
            # Tek asamali association
            cost = self._build_cost(det_centers, det_bboxes, det_confs,
                                    active_tracks, predictions)
            for di, ti in _linear_assignment(cost):
                if cost[di, ti] >= 1e6:
                    continue
                det_idx = det_indices[di]
                tr = active_tracks[ti]
                prev_center = tr.center.copy()

                tr.update(det_bboxes[di], self.frame_count,
                          camera_transform, det_centers[di])

                camera_expected = predictions[ti]["camera_expected_center"]
                score = tr.compute_motion_score(
                    det_centers[di], camera_expected, camera_transform,
                )
                status = tr.decide_moving_status(score)
                tr.last_status = status

                detections[det_idx]["_track_id"] = tr.id
                detections[det_idx]["_motion_score"] = score
                detections[det_idx]["moving_status"] = status

                matched_det_set.add(di)
                matched_track_set.add(ti)
                det_idx_to_track[det_idx] = tr.id

        # M0 #4/#5: Eslesmeyen missed track'leri current frame'e tasi
        for ti, tr in enumerate(active_tracks):
            if ti not in matched_track_set:
                tr.propagate_missed(camera_transform)

        # Eslesmeyen araclara yeni track ac
        for di in range(len(vehicle_dets)):
            if di in matched_det_set:
                continue
            det_idx = det_indices[di]
            det_bbox = det_bboxes[di]
            tr = VehicleTrackV2(self.next_id, det_bbox, self.frame_count, self.config)
            self.tracks[self.next_id] = tr
            self.next_id += 1

            detections[det_idx]["_track_id"] = tr.id
            detections[det_idx]["_motion_score"] = 0.0
            detections[det_idx]["moving_status"] = "0"
            det_idx_to_track[det_idx] = tr.id

        # Diger siniflar icin moving_status=-1
        for i, det in enumerate(detections):
            if det.get("cls") != 0:
                det["moving_status"] = "-1"
            elif i not in det_idx_to_track:
                det["moving_status"] = "0"

        # Stale track'leri temizle
        stale = [tid for tid, tr in self.tracks.items()
                 if tr.time_since_update > self.max_track_age]
        for tid in stale:
            del self.tracks[tid]

        return detections

    def reset(self):
        """Tracker'i sifirla."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        self.prev_gray = None
