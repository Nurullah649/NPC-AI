"""Insan tracker v2 (Faz P4).

Kalman box state: [cx, cy, w, h, vx, vy, vw, vh]
Kamera kompanzasyonlu ByteTrack/OC-SORT tarzi iki asamali association.

Prototip sorunlarinin düzeltmeleri:
- Ilk detections bastirilmiyor (detector'in eşiği geçen gozlemi ciktiya verilir)
- age yalnizca mark_missed() icinde artar (predict'te degil)
- Tek kaçırmada LOST durumuna gecis yok (max_age config'ten)
- Dusuk/orta guvenli yeni insan tespitleri track baslatabilir
- Config isimleri config_experiment.yaml ile uyumlu
- Kamera kompanzasyonu entegre

Tracker kurallari:
- Ana detector'in esiğini geçen gozlem tracker tarafindan bastirilmaz
- Track-only bbox en fazla max_coast_output_frames kare, azalan confidence ile
- Track-only cikti config ile kapatilabilir
- Track kaybinde ilgili ROI sonraki karede yuksek oncelikle taranir
"""

import logging
from collections import OrderedDict
from enum import IntEnum
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from motion.camera_motion import CameraTransform

logger = logging.getLogger(__name__)


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DEAD = 3


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


def _center_from_bbox(bbox: tuple) -> np.ndarray:
    return np.array([
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ], dtype=np.float32)


def _wh_from_bbox(bbox: tuple) -> tuple:
    return (max(1e-6, float(bbox[2] - bbox[0])),
            max(1e-6, float(bbox[3] - bbox[1])))


def _bbox_from_cxywh(cx, cy, w, h) -> tuple:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class PersonTrackV2:
    """Tek insan track'inin durumu.

    8D Kalman box state: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self, track_id: int, bbox: tuple, frame_idx: int, conf: float,
                 config: dict):
        self.id = track_id
        self.frame_idx = frame_idx
        self.state = TrackState.TENTATIVE

        cx, cy = _center_from_bbox(bbox)
        w, h = _wh_from_bbox(bbox)

        self.bbox = bbox
        self.center = np.array([cx, cy], dtype=np.float32)
        self.wh = np.array([w, h], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.confidence = conf

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.coast_frames = 0

        pt_cfg = config.get("person_tracking", {})
        self.max_age = int(pt_cfg.get("max_age", 4))
        self.max_coast_output_frames = int(pt_cfg.get("max_coast_output_frames", 1))
        self.confirm_hits = int(pt_cfg.get("confirm_hits", 2))
        self.confirm_window = int(pt_cfg.get("confirm_window", 6))
        self.coast_enabled = bool(pt_cfg.get("coast_enabled", True))
        self.coast_confidence_decay = float(pt_cfg.get("coast_confidence_decay", 0.3))

        kalman_cfg = pt_cfg.get("kalman", {})
        self.kalman_enabled = True
        pn = float(kalman_cfg.get("std_weight_position", 0.05))
        mn = float(kalman_cfg.get("std_weight_velocity", 0.00625))

        self._init_kalman(cx, cy, w, h, pn, mn)

        self.image_bounds = None

    def _init_kalman(self, cx, cy, w, h, pos_noise, vel_noise):
        """8D Kalman filter: [cx, cy, w, h, vx, vy, vw, vh]."""
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        dt = 1.0
        self.kf.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.F[i, i + 4] = dt

        self.kf.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.H[i, i] = 1.0

        self.kf.P = np.eye(8, dtype=np.float32) * 10.0
        self.kf.P[4:, 4:] *= 100.0

        self.kf.R = np.eye(4, dtype=np.float32) * 1.0

        q = pos_noise
        qv = vel_noise
        self.kf.Q = np.eye(8, dtype=np.float32) * q
        for i in range(4):
            self.kf.Q[i, i + 4] = q * 0.5
            self.kf.Q[i + 4, i] = q * 0.5
            self.kf.Q[i + 4, i + 4] = qv

        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

    def predict(self, camera_transform: Optional[CameraTransform] = None) -> dict:
        """Kalman predict + kamera kompanzasyonu.

        M0 düzeltme: age burada ARTMAZ (yalnizca mark_missed'de artar).
        """
        self.kf.predict()

        pred_center = self.kf.x[:2].copy()
        pred_wh = self.kf.x[2:4].copy()

        if camera_transform is not None and camera_transform.is_reliable:
            pred_center = camera_transform.apply_to_point(pred_center)
            warped_bbox = camera_transform.apply_to_bbox(self.bbox)
            wcx = (warped_bbox[0] + warped_bbox[2]) / 2.0
            wcy = (warped_bbox[1] + warped_bbox[3]) / 2.0
            pred_center = np.array([wcx, wcy], dtype=np.float32)
            pred_wh = np.array([
                warped_bbox[2] - warped_bbox[0],
                warped_bbox[3] - warped_bbox[1],
            ], dtype=np.float32)
            pred_bbox = warped_bbox
        else:
            cx, cy = float(pred_center[0]), float(pred_center[1])
            w, h = float(pred_wh[0]), float(pred_wh[1])
            pred_bbox = _bbox_from_cxywh(cx, cy, w, h)

        self.time_since_update += 1

        return {
            "center": pred_center,
            "bbox": pred_bbox,
            "wh": pred_wh,
        }

    def update(self, bbox: tuple, frame_idx: int, conf: float,
               camera_transform: Optional[CameraTransform] = None):
        """Track'i yeni gozlemle guncelle."""
        det_center = _center_from_bbox(bbox)
        w, h = _wh_from_bbox(bbox)

        self.bbox = bbox
        self.center = det_center.copy()
        self.wh = np.array([w, h], dtype=np.float32)
        self.confidence = conf

        z = np.array([det_center[0], det_center[1], w, h], dtype=np.float32)
        self.kf.update(z)
        self.velocity = self.kf.x[4:6].copy()
        self.image_bounds = (float(bbox[2]), float(bbox[3]))

        self.hits += 1
        self.age = 0
        self.time_since_update = 0
        self.coast_frames = 0
        self.last_frame = frame_idx
        self.frame_idx = frame_idx

        if self.state == TrackState.TENTATIVE:
            if self.hits >= self.confirm_hits:
                self.state = TrackState.CONFIRMED

    def mark_missed(self, frame_idx: int):
        """Track kaçırmış olarak işaretle.

        M0 düzeltme: age yalnizca burada artar (predict'te degil).
        Tek kaçırmada LOST'a gecis yok (max_age kontrolu).
        """
        self.age += 1
        self.time_since_update += 1
        if self.state == TrackState.TENTATIVE:
            if self.age > 1:
                self.state = TrackState.DEAD
        elif self.state == TrackState.CONFIRMED:
            if self.age > self.max_age:
                self.state = TrackState.LOST
            else:
                self.coast_frames += 1
        elif self.state == TrackState.LOST:
            if self.age > self.max_age * 2:
                self.state = TrackState.DEAD

    def propagate_missed(self, camera_transform: Optional[CameraTransform]):
        """Missed track'i current frame'e tasi."""
        if camera_transform is not None and camera_transform.is_reliable:
            self.center = camera_transform.apply_to_point(self.center)
            self.bbox = camera_transform.apply_to_bbox(self.bbox)
            cx, cy = _center_from_bbox(self.bbox)
            w, h = _wh_from_bbox(self.bbox)
            self.kf.x[:4] = np.array([cx, cy, w, h], dtype=np.float32)

    def get_coast_bbox(self) -> Optional[tuple]:
        """Track-only (coast) bbox dondur.

        En fazla max_coast_output_frames kare, azalan confidence ile.
        Ek kisitlamalar: coast_enabled, goruntu sinirlari, velocity, covariance.
        """
        if self.state != TrackState.CONFIRMED:
            return None
        if not (self.time_since_update > 0 and self.time_since_update <= self.max_coast_output_frames):
            return None
        if not self.coast_enabled:
            return None

        # Coast bbox merkezi goruntu sinirlari icinde mi?
        if self.image_bounds is not None:
            W, H = self.image_bounds
            cx, cy = float(self.center[0]), float(self.center[1])
            if not (0 <= cx <= W and 0 <= cy <= H):
                return None

        # Dusuk hareket belirsizligi: velocity < bbox_diagonal * 0.5
        bbox_diag = float(np.sqrt(
            (self.bbox[2] - self.bbox[0]) ** 2 +
            (self.bbox[3] - self.bbox[1]) ** 2
        ))
        vel_mag = float(np.linalg.norm(self.velocity))
        if vel_mag >= bbox_diag * 0.5:
            return None

        if not self.covariance_low:
            return None

        decay = max(0.0, 1.0 - self.coast_confidence_decay * self.coast_frames)
        self._coast_conf = self.confidence * decay
        return self.bbox

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST)

    @property
    def is_dead(self) -> bool:
        return self.state == TrackState.DEAD

    @property
    def covariance_low(self) -> bool:
        """Kalman pozisyon kovaryansi dusukse True."""
        return (self.kf.P[0, 0] + self.kf.P[1, 1]) < 50.0


class PersonTrackerV2:
    """Insan tracker v2.

    ByteTrack tarzi iki asamali association:
    - Stage 1: Yuksek guvenli detections -> tum track'ler
    - Stage 2: Dusuk guvenli detections -> kalan track'ler (kurtarma)

    Args:
        config: Deneysel config dict (person_tracking section)
    """

    def __init__(self, config: dict):
        self.config = config
        pt_cfg = config.get("person_tracking", {})

        self.enabled = bool(pt_cfg.get("enabled", True))
        self.high_conf_threshold = float(pt_cfg.get("high_conf_threshold", 0.40))
        self.low_conf_threshold = float(pt_cfg.get("low_conf_threshold", 0.10))
        self.new_track_threshold = float(pt_cfg.get("new_track_threshold", 0.30))
        self.use_camera_compensation = bool(pt_cfg.get("use_camera_compensation", True))
        self.max_age = int(pt_cfg.get("max_age", 4))
        self.max_coast_output_frames = int(pt_cfg.get("max_coast_output_frames", 1))

        self.tracks: OrderedDict[int, PersonTrackV2] = OrderedDict()
        self.next_id = 0
        self.frame_count = 0

        self.lost_rois = []

        logger.info(
            "PersonTrackerV2: high_conf=%.2f, low_conf=%.2f, new_track=%.2f, max_age=%d",
            self.high_conf_threshold, self.low_conf_threshold,
            self.new_track_threshold, self.max_age,
        )

    def _build_cost(self, det_centers, det_bboxes, det_confs,
                    tracks, predictions) -> np.ndarray:
        """Cost matrix: (1-IoU) + normalized center distance."""
        n_dets = len(det_centers)
        n_tracks = len(tracks)
        cost = np.full((n_dets, n_tracks), 1e6, dtype=np.float32)

        for di in range(n_dets):
            det_center = det_centers[di]
            det_bbox = det_bboxes[di]
            for ti in range(n_tracks):
                tr = tracks[ti]
                pred = predictions[ti]
                pred_center = pred["center"]
                pred_bbox = pred["bbox"]

                iou = _box_iou(det_bbox, pred_bbox)
                dist = float(np.linalg.norm(det_center - pred_center))

                diag = float(np.sqrt(
                    (det_bbox[2] - det_bbox[0]) ** 2 +
                    (det_bbox[3] - det_bbox[1]) ** 2
                ))
                norm_dist = dist / max(diag, 20.0)

                c = (1.0 - iou) + norm_dist * 0.5

                if iou < 0.05 and dist > diag * 1.5:
                    c = 1e6

                cost[di, ti] = c

        return cost

    def update(
        self,
        detections: list,
        frame_idx: int,
        camera_transform: Optional[CameraTransform] = None,
    ) -> list:
        """Bir kare isle.

        Args:
            detections: Insan detection dict listesi (cls=1)
            frame_idx: Kare indeksi
            camera_transform: Kamera transformu

        Returns:
            Cikti detection listesi:
            - Detector'dan gelen gozlemler (eşiği geçen) bastirilmaz
            - Track-only (coast) bbox'lar azalan confidence ile eklenir
        """
        if not self.enabled:
            return detections

        self.frame_count = frame_idx

        # Tum person detection'larini al (cls=1)
        person_dets = [(i, det) for i, det in enumerate(detections) if det.get("cls") == 1]

        active_tracks = [tr for tr in self.tracks.values() if tr.is_active]

        # Predict tum track'ler
        predictions = []
        for tr in active_tracks:
            pred = tr.predict(camera_transform if self.use_camera_compensation else None)
            predictions.append(pred)

        det_indices = [pd[0] for pd in person_dets]
        det_centers = [_center_from_bbox(pd[1]["bbox"]) for pd in person_dets]
        det_bboxes = [pd[1]["bbox"] for pd in person_dets]
        det_confs = [pd[1].get("conf", 0.5) for pd in person_dets]

        matched_det_set = set()
        matched_track_set = set()

        # Stage 1: Yuksek guvenli detections
        high_indices = [i for i, c in enumerate(det_confs) if c >= self.high_conf_threshold]
        low_indices = [i for i, c in enumerate(det_confs)
                       if self.low_conf_threshold <= c < self.high_conf_threshold]

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
                tr = active_tracks[ti]

                tr.update(det_bboxes[real_di], frame_idx, det_confs[real_di],
                          camera_transform if self.use_camera_compensation else None)

                matched_det_set.add(real_di)
                matched_track_set.add(ti)

        # Stage 2: Dusuk guvenli detections (yalniz track kurtarma)
        remaining_tracks = [(ti, active_tracks[ti]) for ti in range(len(active_tracks))
                           if ti not in matched_track_set]
        remaining_preds = [predictions[ti] for ti in range(len(active_tracks))
                          if ti not in matched_track_set]

        low_available = [i for i in low_indices if i not in matched_det_set]
        if low_available and remaining_tracks:
            low_centers = [det_centers[i] for i in low_available]
            low_bboxes = [det_bboxes[i] for i in low_available]
            low_confs = [det_confs[i] for i in low_available]
            rem_tracks_list = [t for _, t in remaining_tracks]

            cost_l = self._build_cost(low_centers, low_bboxes, low_confs,
                                      rem_tracks_list, remaining_preds)
            for di, ti in _linear_assignment(cost_l):
                if cost_l[di, ti] >= 1e6:
                    continue
                real_di = low_available[di]
                tr = rem_tracks_list[ti]

                tr.update(det_bboxes[real_di], frame_idx, det_confs[real_di],
                          camera_transform if self.use_camera_compensation else None)

                matched_det_set.add(real_di)
                matched_track_set.add(active_tracks.index(tr))

        # Missed track'leri propagate et
        self.lost_rois = []
        for ti, tr in enumerate(active_tracks):
            if ti not in matched_track_set:
                tr.mark_missed(frame_idx)
                if self.use_camera_compensation:
                    tr.propagate_missed(camera_transform)
                if tr.state == TrackState.LOST:
                    self.lost_rois.append(tr.bbox)

        # LOST track re-identification: eslesmeyen detection'lari once LOST track'lerle dene
        for di in range(len(person_dets)):
            if di in matched_det_set:
                continue
            conf = det_confs[di]
            if conf < self.new_track_threshold:
                continue
            det_bbox = det_bboxes[di]
            det_center = det_centers[di]

            best_lost = None
            best_lost_id = None
            best_iou = 0.0

            for tid, tr in self.tracks.items():
                if tr.state != TrackState.LOST:
                    continue
                iou = _box_iou(det_bbox, tr.bbox)
                diag = float(np.sqrt(
                    (det_bbox[2] - det_bbox[0]) ** 2 +
                    (det_bbox[3] - det_bbox[1]) ** 2
                ))
                center_dist = float(np.linalg.norm(det_center - tr.center))
                if iou > 0.3 or (diag > 0 and center_dist < diag * 0.5):
                    if iou > best_iou:
                        best_iou = iou
                        best_lost = tr
                        best_lost_id = tid

            if best_lost is not None:
                # LOST track'i canlandir
                best_lost.state = TrackState.CONFIRMED
                best_lost.update(det_bbox, frame_idx, conf,
                                 camera_transform if self.use_camera_compensation else None)
                matched_det_set.add(di)

        # Eslesmeyen detection'lara yeni track ac (dedup ile)
        new_track_bboxes = []
        for di in range(len(person_dets)):
            if di in matched_det_set:
                continue
            conf = det_confs[di]
            if conf < self.new_track_threshold:
                continue
            det_bbox = det_bboxes[di]
            is_dup = False
            for nb in new_track_bboxes:
                if _box_iou(det_bbox, nb) > 0.5:
                    is_dup = True
                    break
            if is_dup:
                continue
            tr = PersonTrackV2(
                self.next_id, det_bbox, frame_idx, conf, self.config,
            )
            self.tracks[self.next_id] = tr
            self.next_id += 1
            new_track_bboxes.append(det_bbox)

        # Dead track'leri temizle
        dead = [tid for tid, tr in self.tracks.items() if tr.is_dead]
        for tid in dead:
            del self.tracks[tid]

        # Cikti: detector gozlemleri + coast bbox'lar
        output = list(detections)

        # Detector'dan gelen eslesmis gozlemlere track_id ekle
        for di in matched_det_set:
            det_idx = det_indices[di]
            for tr in self.tracks.values():
                if tr.last_frame == frame_idx and tr.time_since_update == 0:
                    cx_det = _center_from_bbox(detections[det_idx]["bbox"])
                    if np.allclose(tr.center, cx_det, atol=1.0):
                        if "_track_id" not in detections[det_idx]:
                            detections[det_idx]["_track_id"] = tr.id
                        break

        # Coast (track-only) bbox'lar ekle
        for tr in self.tracks.values():
            if tr.is_confirmed and tr.time_since_update > 0:
                coast_bbox = tr.get_coast_bbox()
                if coast_bbox is not None:
                    output.append({
                        "cls": 1,
                        "cls_name": "Insan",
                        "conf": float(tr._coast_conf),
                        "bbox": coast_bbox,
                        "_track_id": tr.id,
                        "_source": "tracker_coast",
                    })

        return output

    def get_active_track_bboxes(self) -> list:
        """Aktif track bbox listesi (tile scheduler icin)."""
        return [
            {"bbox": tr.bbox, "id": tr.id}
            for tr in self.tracks.values()
            if tr.is_confirmed and tr.time_since_update <= 2
        ]

    def get_lost_rois(self) -> list:
        """Kaybolmus track ROI listesi (tile scheduler icin)."""
        return list(self.lost_rois)

    def reset(self):
        """Tracker'i sifirla."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0
        self.lost_rois = []
