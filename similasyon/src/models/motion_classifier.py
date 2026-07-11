import logging
from collections import OrderedDict

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


def _linear_assignment(cost_matrix):
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
            (cost_matrix[r, c], r, c)
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


class VehicleTrack:
    """Tek taşıt track'inin durumunu tutan sınıf."""

    def __init__(self, track_id: int, bbox: tuple, center: np.ndarray, frame_count: int,
                 kalman_enabled: bool = True,
                 kalman_process_noise: float = 1.0,
                 kalman_measurement_noise: float = 10.0):
        self.id = track_id
        self.bbox = bbox
        self.center = center.copy()
        self.last_center = center.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.history = [(frame_count, center.copy())]
        self.motion_scores = []
        self.hits = 1
        self.age = 0
        self.last_status = "0"
        self.last_frame = frame_count

        self.kalman_enabled = kalman_enabled
        self.kf = None
        if kalman_enabled:
            self.kf = KalmanFilter(dim_x=4, dim_z=2)
            self.kf.F = np.array([[1, 0, 1, 0],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], dtype=np.float32)
            self.kf.H = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]], dtype=np.float32)
            self.kf.P = np.eye(4, dtype=np.float32) * 100.0
            self.kf.R = np.eye(2, dtype=np.float32) * kalman_measurement_noise
            q = kalman_process_noise
            self.kf.Q = np.array([[1/3, 0,   1/2, 0  ],
                                  [0,   1/3, 0,   1/2],
                                  [1/2, 0,   1,   0  ],
                                  [0,   1/2, 0,   1  ]], dtype=np.float32) * q
            self.kf.x = np.array([center[0], center[1], 0.0, 0.0], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Kalman predict varsa kullan, yoksa last_center + velocity."""
        if self.kf is not None:
            self.kf.predict()
            return self.kf.x[:2].copy()
        return self.last_center + self.velocity

    def update(self, bbox: tuple, center: np.ndarray, frame_count: int):
        """Track'i yeni gözlemle güncelle; hızı yumuşat (Kalman varsa Kalman kullan)."""
        self.bbox = bbox
        self.last_center = self.center.copy()
        self.center = center.copy()
        if self.kf is not None:
            self.kf.update(center)
            self.velocity = self.kf.x[2:4].copy()
        else:
            new_velocity = center - self.last_center
            self.velocity = 0.7 * self.velocity + 0.3 * new_velocity
        self.history.append((frame_count, self.center.copy()))
        if len(self.history) > 20:
            self.history.pop(0)
        self.hits += 1
        self.age = 0
        self.last_frame = frame_count


class MotionClassifier:
    """Taşıt hareket durumu sınıflandırıcısı.

    Tracker tabanlı üretim versiyonu:
    - ORB + affine/shift ile frame-to-frame kamera kaymasını hesaplar.
    - Track'leri Hungarian assignment ile eşleştirir; tahmin merkezleri
      kamera transformuna göre ilerletilir.
    - Hareket skoru residual medyanıyla; histerzisli durum geçişleri.
    - Sadece cls=0 (Taşıt) için çalışır. İnsan/UAP/UAI moving_status=-1.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        motion_cfg = config.get("motion", {})

        self.min_track_len = int(motion_cfg.get("min_track_len", 4))
        self.max_track_age = int(motion_cfg.get("max_track_age", 15))
        self.track_match_distance = float(motion_cfg.get("track_match_distance", 90.0))
        self.moving_start_threshold = float(motion_cfg.get("moving_start_threshold", 18.0))
        self.moving_stop_threshold = float(motion_cfg.get("moving_stop_threshold", 8.0))
        self.motion_window = int(motion_cfg.get("motion_window", 6))
        self.affine_min_matches = int(motion_cfg.get("affine_min_matches", 12))
        self.affine_ransac_threshold = float(motion_cfg.get("affine_ransac_threshold", 5.0))

        self.kalman_enabled = bool(motion_cfg.get("kalman_enabled", True))
        self.kalman_process_noise = float(motion_cfg.get("kalman_process_noise", 1.0))
        self.kalman_measurement_noise = float(motion_cfg.get("kalman_measurement_noise", 10.0))

        self.tracks = OrderedDict()
        self.next_id = 0
        self.frame_count = 0

        self.prev_gray = None

        self.logger.info(
            "MotionClassifier başlatıldı. "
            f"min_track_len={self.min_track_len}, "
            f"max_track_age={self.max_track_age}, "
            f"track_match_distance={self.track_match_distance}px, "
            f"moving_start_threshold={self.moving_start_threshold}px, "
            f"moving_stop_threshold={self.moving_stop_threshold}px, "
            f"motion_window={self.motion_window}, "
            f"affine_min_matches={self.affine_min_matches}, "
            f"affine_ransac_threshold={self.affine_ransac_threshold}, "
            f"kalman_enabled={self.kalman_enabled}, "
            f"kalman_process_noise={self.kalman_process_noise}, "
            f"kalman_measurement_noise={self.kalman_measurement_noise}"
        )

    @staticmethod
    def _center(bbox):
        return np.array(
            [
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                (float(bbox[1]) + float(bbox[3])) / 2.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _iou(box1, box2):
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

    def _estimate_camera_transform(self, prev_gray: np.ndarray, gray: np.ndarray):
        """İki frame arası kamera transformunu affine (ya da shift) olarak hesapla."""
        orb = cv2.ORB.create(nfeatures=800)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 8:
            return None, None

        matches = matches[:160]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        A, inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=self.affine_ransac_threshold
        )
        if A is not None and inliers is not None:
            inlier_count = int(np.sum(inliers))
            if inlier_count >= self.affine_min_matches:
                return A, None

        # Fallback: ortalama kayma
        shift = (np.mean(dst_pts, axis=0) - np.mean(src_pts, axis=0)).flatten().astype(np.float32)
        return None, shift

    def update(self, detections: list, gray: np.ndarray) -> list:
        self.frame_count += 1

        A = None
        shift = None
        if self.prev_gray is not None:
            A, shift = self._estimate_camera_transform(self.prev_gray, gray)
        self.prev_gray = gray

        # Track yaşlarını artır
        for track in self.tracks.values():
            track.age += 1

        # Araç deteksiyonlarını hazırla (cls == 0)
        vehicle_indices = []
        vehicle_centers = []
        vehicle_bboxes = []
        for i, det in enumerate(detections):
            if det.get("cls") == 0:
                bbox = det["bbox"]
                center = self._center(bbox)
                vehicle_indices.append(i)
                vehicle_centers.append(center)
                vehicle_bboxes.append(bbox)

        active_tracks = [tr for tr in self.tracks.values() if tr.age <= self.max_track_age]

        # Tahmin merkezlerini hesapla (prediction-forward)
        # Kalman predict'ini tüm track'ler için çağır (predict-update döngüsü)
        # Not: predict() ile kf state ilerletilir, assignment için hâlâ
        # kamera kompanzasyonlu tahmin (affine/shift) kullanılır.
        for tr in active_tracks:
            tr.predict()  # calls kf.predict() if Kalman enabled

        predicted_centers = []
        for tr in active_tracks:
            if A is not None:
                pt = np.array([tr.last_center[0], tr.last_center[1], 1.0], dtype=np.float32)
                pred = (A @ pt)[:2]
            elif shift is not None:
                pred = tr.last_center + shift
            else:
                pred = tr.last_center + tr.velocity
            predicted_centers.append(pred)

        n_dets = len(vehicle_indices)
        n_tracks = len(active_tracks)
        cost = np.full((n_dets, n_tracks), 1e6, dtype=np.float32)

        for di in range(n_dets):
            det_center = vehicle_centers[di]
            det_bbox = vehicle_bboxes[di]
            det_area = max(1e-6, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
            for ti in range(n_tracks):
                tr = active_tracks[ti]
                pred_center = predicted_centers[ti]
                dist = float(np.linalg.norm(det_center - pred_center))
                iou = self._iou(det_bbox, tr.bbox)
                tr_area = max(1e-6, (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1]))
                ratio = max(det_area, tr_area) / min(det_area, tr_area) - 1.0
                size_ratio_penalty = min(ratio, 5.0)
                c = dist + (1.0 - iou) * 30.0 + size_ratio_penalty * 10.0
                if dist > self.track_match_distance and iou < 0.05:
                    c = 1e6
                cost[di, ti] = c

        matched_det_set = set()
        matched_track_set = set()
        det_idx_to_track = {}

        if n_dets > 0 and n_tracks > 0:
            for di, ti in _linear_assignment(cost):
                if cost[di, ti] >= 1e6:
                    continue
                tr = active_tracks[ti]
                det_idx = vehicle_indices[di]
                det_center = vehicle_centers[di]
                det_bbox = vehicle_bboxes[di]

                # Track güncelle
                tr.update(det_bbox, det_center, self.frame_count)

                # Hareket skoru (residual medyanı)
                pred_center = predicted_centers[ti]
                residual = float(np.linalg.norm(det_center - pred_center))
                tr.motion_scores.append(residual)
                if len(tr.motion_scores) > self.motion_window:
                    tr.motion_scores.pop(0)

                if len(tr.motion_scores) >= self.motion_window:
                    score = float(np.median(tr.motion_scores))
                else:
                    score = residual

                # Histerzisli durum kararı
                if tr.hits < self.min_track_len:
                    status = "0"
                elif tr.last_status == "1":
                    status = "1" if score > self.moving_stop_threshold else "0"
                else:
                    status = "1" if score > self.moving_start_threshold else "0"
                tr.last_status = status

                detections[det_idx]["_track_id"] = tr.id
                detections[det_idx]["_motion_score"] = score
                detections[det_idx]["moving_status"] = status

                matched_det_set.add(di)
                matched_track_set.add(ti)
                det_idx_to_track[det_idx] = tr.id

        # Eşleşmeyen araçlara yeni track aç
        for di in range(n_dets):
            if di in matched_det_set:
                continue
            det_idx = vehicle_indices[di]
            det_bbox = vehicle_bboxes[di]
            det_center = vehicle_centers[di]
            tr = VehicleTrack(self.next_id, det_bbox, det_center, self.frame_count,
                              kalman_enabled=self.kalman_enabled,
                              kalman_process_noise=self.kalman_process_noise,
                              kalman_measurement_noise=self.kalman_measurement_noise)
            self.tracks[self.next_id] = tr
            self.next_id += 1

            detections[det_idx]["_track_id"] = tr.id
            detections[det_idx]["_motion_score"] = 0.0
            detections[det_idx]["moving_status"] = "0"
            det_idx_to_track[det_idx] = tr.id

        # Diğer sınıflar için moving_status=-1
        for i, det in enumerate(detections):
            if det.get("cls") != 0:
                det["moving_status"] = "-1"
            elif i not in det_idx_to_track:
                det["moving_status"] = "0"

        # Eski trackleri temizle
        stale_ids = [tid for tid, tr in self.tracks.items() if tr.age > self.max_track_age]
        for tid in stale_ids:
            del self.tracks[tid]

        return detections

    def get_moving_status(self, cls_id: int, bbox: tuple, gray: np.ndarray) -> str:
        return "0"
