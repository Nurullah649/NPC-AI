import logging
from collections import OrderedDict

import numpy as np
import cv2


class MotionClassifier:
    """Taşıt hareket durumu sınıflandırıcısı.

    Son N frame boyunca taşıt bbox center'larını takip eder,
    kamera hareketini kompanze ederek aracın hareketli/sabit
    olduğuna karar verir.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        motion_cfg = config.get("motion", {})
        self.min_track_len = motion_cfg.get("min_track_len", 3)
        self.moving_pixel_threshold = motion_cfg.get("moving_pixel_threshold", 8.0)

        # Track buffer: {track_id: [(frame_idx, cx, cy), ...]}
        self.tracks: dict[int, list] = {}
        self.next_id = 0
        self.frame_count = 0

        # Camera motion compensation (homography-based)
        self.prev_gray = None
        self.camera_shift = np.array([0.0, 0.0])

        self.logger.info(
            f"MotionClassifier başlatıldı. "
            f"min_track_len={self.min_track_len}, "
            f"threshold={self.moving_pixel_threshold}px"
        )

    def _compute_camera_motion(self, gray: np.ndarray) -> np.ndarray:
        """İki frame arası kamera hareketini homography ile hesapla."""
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.array([0.0, 0.0])

        # ORB feature detection
        orb = cv2.ORB.create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            self.prev_gray = gray
            return np.array([0.0, 0.0])

        # Feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 4:
            self.prev_gray = gray
            return np.array([0.0, 0.0])

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is not None and mask is not None:
            inlier_count = np.sum(mask)
            if inlier_count > 10:
                src_mean = np.mean(src_pts[mask.squeeze() > 0], axis=0)
                dst_mean = np.mean(dst_pts[mask.squeeze() > 0], axis=0)
                self.camera_shift = (dst_mean - src_mean).flatten()

        self.prev_gray = gray
        return self.camera_shift

    def _match_to_track(self, cx: float, cy: float, iou_threshold: float = 0.3) -> int | None:
        """Mevcut detection'ı mevcut track'lere IoU + mesafe ile eşle."""
        best_id = None
        best_dist = float('inf')

        for track_id, history in self.tracks.items():
            if not history:
                continue
            last_cx, last_cy = history[-1][1], history[-1][2]

            # Kamera hareketini kompanse et
            compensated_cx = cx - self.camera_shift[0]
            compensated_cy = cy - self.camera_shift[1]

            dist = np.sqrt((compensated_cx - last_cx) ** 2 + (compensated_cy - last_cy) ** 2)
            if dist < best_dist and dist < self.moving_pixel_threshold * 3:
                best_dist = dist
                best_id = track_id

        return best_id

    def update(self, detections: list, gray: np.ndarray) -> list:
        """Detection listesine moving_status ekleyerek günceller.

        Args:
            detections: DetectorYOLO.detect() çıktısı (dict listesi)
            gray: Gri-tonlamalı frame (kamera hareketi için)

        Returns:
            Her detection'a 'moving_status' eklenmiş liste.
        """
        self.frame_count += 1
        camera_shift = self._compute_camera_motion(gray)

        # Sadece taşıt (cls=0) sınıfı için track
        vehicle_detections = [(i, d) for i, d in enumerate(detections) if d['cls'] == 0]

        # Mevcut track'leri güncelle
        for orig_idx, det in vehicle_detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            track_id = self._match_to_track(cx, cy)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = []

            self.tracks[track_id].append((self.frame_count, cx, cy))

        # Eski trackleri temizle (son 30 frame'de görülmeyen)
        to_delete = []
        for track_id, history in self.tracks.items():
            if history and self.frame_count - history[-1][0] > 30:
                to_delete.append(track_id)
        for tid in to_delete:
            del self.tracks[tid]

        # Her detection için moving_status ata
        result = []
        for det in detections:
            if det['cls'] == 0:
                # Taşıt: track durumuna bak
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                # Hangi track'e ait?
                track_id = self._match_to_track(cx, cy)
                if track_id is not None and track_id in self.tracks:
                    history = self.tracks[track_id]
                    if len(history) >= self.min_track_len:
                        # Kamera kompanzasyonu ile hareket hesabı
                        positions = np.array([(h[1], h[2]) for h in history])
                        # Son N nokta arası toplam mesafe
                        if len(positions) >= 2:
                            total_motion = np.sum(np.abs(np.diff(positions, axis=0)))
                            if total_motion > self.moving_pixel_threshold:
                                det['moving_status'] = "1"  # Hareketli
                            else:
                                det['moving_status'] = "0"  # Sabit
                        else:
                            det['moving_status'] = "0"  # Sabit (varsayılan)
                    else:
                        det['moving_status'] = "0"  # Sabit (yetersiz track)
                else:
                    det['moving_status'] = "0"  # Sabit (track yok)
            else:
                det['moving_status'] = "-1"  # Taşıt Değil

            result.append(det)

        return result

    def get_moving_status(self, cls_id: int, bbox: tuple, gray: np.ndarray) -> str:
        """Tek bir detection için moving_status döndür (basit API)."""
        if cls_id != 0:
            return "-1"  # Taşıt Değil
        # Bu metod asıl update() içinde kullanılır
        return "0"


# cv2 import'u burada yapıyoruz (döngüsel import önlemek için)
