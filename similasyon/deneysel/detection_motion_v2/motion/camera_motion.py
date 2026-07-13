"""Robust kamera hareketi tahmini (Faz M1).

Pipeline:
1. Detection bbox'larini feature alanindan maskele
2. Statik arka planda GFTT/ORB noktalari sec
3. LK optical flow veya descriptor matching ile noktalari takip et
4. RANSAC homography hesapla
5. Homography kalitesi gecmezse affine fallback
6. Affine de gecmezse inlier/robust median translation
7. Hiçbiri guvenilir degilse transform_unreliable durumu

Kalite alanlari:
- match_count: toplam eslesme sayisi
- inlier_count, inlier_ratio: RANSAC inlier
- reprojection_error: median/mean reprojection hatasi
- model_type: homography | affine | shift | none
- confidence: [0, 1] arasi transform guven skoru
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraModel(IntEnum):
    NONE = 0
    SHIFT = 1
    AFFINE = 2
    HOMOGRAPHY = 3


@dataclass
class CameraTransform:
    """Kamera transformu ve kalite bilgisi."""
    model_type: CameraModel = CameraModel.NONE
    matrix: np.ndarray = None  # 3x3 homography veya 2x3 affine
    shift: np.ndarray = None   # (dx, dy) shift
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    confidence: float = 0.0

    @property
    def is_reliable(self) -> bool:
        return self.model_type != CameraModel.NONE and self.confidence > 0.0

    def apply_to_point(self, pt: np.ndarray) -> np.ndarray:
        """Tek bir noktayi current frame'e tasi."""
        if self.model_type == CameraModel.HOMOGRAPHY and self.matrix is not None:
            pt_h = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
            result_h = self.matrix @ pt_h
            result = result_h[:2] / result_h[2]
            return result.astype(np.float32)
        elif self.model_type == CameraModel.AFFINE and self.matrix is not None:
            pt_h = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
            result = (self.matrix @ pt_h)[:2]
            return result.astype(np.float32)
        elif self.model_type == CameraModel.SHIFT and self.shift is not None:
            return (pt + self.shift).astype(np.float32)
        return pt.copy().astype(np.float32)

    def apply_to_bbox(self, bbox: tuple) -> tuple:
        """Bbox koselerini current frame'e warp et."""
        x1, y1, x2, y2 = bbox
        corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)

        if self.model_type == CameraModel.HOMOGRAPHY and self.matrix is not None:
            ones = np.ones((4, 1), dtype=np.float32)
            pts_h = np.hstack([corners, ones])
            warped = (self.matrix @ pts_h.T).T
            warped = warped[:, :2] / warped[:, 2:3]
        elif self.model_type == CameraModel.AFFINE and self.matrix is not None:
            ones = np.ones((4, 1), dtype=np.float32)
            pts_h = np.hstack([corners, ones])
            warped = (self.matrix @ pts_h.T).T[:, :2]
        elif self.model_type == CameraModel.SHIFT and self.shift is not None:
            warped = corners + self.shift
        else:
            warped = corners

        nx1 = float(np.min(warped[:, 0]))
        ny1 = float(np.min(warped[:, 1]))
        nx2 = float(np.max(warped[:, 0]))
        ny2 = float(np.max(warped[:, 1]))
        return (nx1, ny1, nx2, ny2)

    def chain(self, other: "CameraTransform") -> "CameraTransform":
        """Iki transformu zincirle (missed frame'ler icin)."""
        if not other.is_reliable:
            return self
        if not self.is_reliable:
            return other

        result = CameraTransform()
        result.model_type = max(self.model_type, other.model_type)
        result.match_count = max(self.match_count, other.match_count)
        result.inlier_count = max(self.inlier_count, other.inlier_count)
        result.inlier_ratio = (self.inlier_ratio + other.inlier_ratio) / 2.0
        result.reprojection_error = (self.reprojection_error + other.reprojection_error) / 2.0
        result.confidence = (self.confidence + other.confidence) / 2.0

        if self.model_type == CameraModel.HOMOGRAPHY and other.model_type == CameraModel.HOMOGRAPHY:
            result.matrix = other.matrix @ self.matrix
        elif self.model_type in (CameraModel.HOMOGRAPHY, CameraModel.AFFINE) and \
             other.model_type in (CameraModel.HOMOGRAPHY, CameraModel.AFFINE):
            m1 = self.matrix if self.matrix.shape == (3, 3) else np.vstack([self.matrix, [0, 0, 1]])
            m2 = other.matrix if other.matrix.shape == (3, 3) else np.vstack([other.matrix, [0, 0, 1]])
            result.matrix = (m2 @ m1)[:2] if result.model_type == CameraModel.AFFINE else m2 @ m1
        else:
            result.model_type = CameraModel.SHIFT
            s1 = self._get_shift()
            s2 = other._get_shift()
            result.shift = (s1 + s2).astype(np.float32)

        return result

    def _get_shift(self) -> np.ndarray:
        if self.model_type == CameraModel.SHIFT and self.shift is not None:
            return self.shift
        elif self.matrix is not None and self.matrix.shape == (2, 3):
            return np.array([self.matrix[0, 2], self.matrix[1, 2]], dtype=np.float32)
        elif self.matrix is not None and self.matrix.shape == (3, 3):
            return np.array([self.matrix[0, 2], self.matrix[1, 2]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)


class CameraMotionEstimator:
    """Robust kamera hareketi tahmin edici.

    Args:
        config: motion section of config_experiment.yaml
    """

    def __init__(self, config: dict):
        motion_cfg = config.get("motion", {})
        self.camera_model = motion_cfg.get("camera_model", "homography")
        self.affine_fallback = bool(motion_cfg.get("affine_fallback", True))
        self.median_shift_fallback = bool(motion_cfg.get("median_shift_fallback", True))
        self.mask_detection_regions = bool(motion_cfg.get("mask_detection_regions", True))
        self.min_inlier_ratio = float(motion_cfg.get("min_inlier_ratio", 0.35))
        self.max_reprojection_error = float(motion_cfg.get("max_reprojection_error", 5.0))
        self.feature_type = motion_cfg.get("feature_type", "orb")
        self.max_features = int(motion_cfg.get("max_features", 800))
        self.affine_min_matches = int(motion_cfg.get("affine_min_matches", 12))
        self.affine_ransac_threshold = float(motion_cfg.get("affine_ransac_threshold", 5.0))

        self._orb = None
        self._gftt_params = None
        if self.feature_type == "orb":
            self._orb = cv2.ORB.create(nfeatures=self.max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        logger.info(
            "CameraMotionEstimator: model=%s, feature=%s, min_inlier_ratio=%.2f",
            self.camera_model, self.feature_type, self.min_inlier_ratio,
        )

    def _create_mask(self, shape: tuple, detections: list, expand_ratio: float = 1.2) -> np.ndarray:
        """Detection bbox'larini maskele (feature alani disinda birak)."""
        H, W = shape[:2]
        mask = np.ones((H, W), dtype=np.uint8) * 255

        if not self.mask_detection_regions or not detections:
            return mask

        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = (x2 - x1) * expand_ratio
            h = (y2 - y1) * expand_ratio
            mx1 = max(0, int(cx - w / 2))
            my1 = max(0, int(cy - h / 2))
            mx2 = min(W, int(cx + w / 2))
            my2 = min(H, int(cy + h / 2))
            mask[my1:my2, mx1:mx2] = 0

        return mask

    def _extract_features(self, gray: np.ndarray, mask: np.ndarray):
        """ORB veya GFTT ile feature cikar."""
        if self.feature_type == "orb" and self._orb is not None:
            kp, des = self._orb.detectAndCompute(gray, mask)
            return kp, des
        else:
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_features, qualityLevel=0.01,
                minDistance=10, mask=mask,
            )
            if corners is None:
                return [], None
            return [cv2.KeyPoint(p[0][0], p[0][1], 1) for p in corners], None

    def _match_features(self, kp1, des1, kp2, des2):
        """ORB descriptor ile feature eslestir."""
        if des1 is None or des2 is None:
            return [], kp1, kp2
        matches = self._bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches, kp1, kp2

    def estimate(self, prev_gray: np.ndarray, gray: np.ndarray,
                 detections: list = None,
                 prev_detections: list = None) -> CameraTransform:
        """Iki frame arasi kamera transformunu tahmin et.

        Args:
            prev_gray: Onceki frame grayscale
            gray: Mevcut frame grayscale
            detections: Mevcut frame detection listesi (maskeleme icin)
            prev_detections: Onceki frame detection listesi

        Returns:
            CameraTransform nesnesi
        """
        transform = CameraTransform()

        if prev_gray is None or gray is None:
            return transform

        H, W = gray.shape[:2]
        mask = self._create_mask(gray.shape, detections or [])
        prev_mask = self._create_mask(prev_gray.shape, prev_detections or [])

        kp1, des1 = self._extract_features(prev_gray, prev_mask)
        kp2, des2 = self._extract_features(gray, mask)

        # GFTT + LK optical flow dogrudan burada
        if self.feature_type != "orb" and len(kp1) >= 4 and len(kp2) >= 4:
            pts1 = np.float32([k.pt for k in kp1]).reshape(-1, 1, 2)
            pts2, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts1, None)
            class _M:
                def __init__(self, q, t, d):
                    self.queryIdx = q
                    self.trainIdx = t
                    self.distance = d
            matches = []
            for i, s in enumerate(status):
                if s and i < len(kp2):
                    matches.append(_M(i, i, float(err[i]) if err is not None else 0.0))
            mkp1, mkp2 = kp1, kp2
        else:
            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                logger.debug("Kamera transform: yetersiz feature (kp1=%d, kp2=%d)", len(kp1), len(kp2))
                return transform

            matches, mkp1, mkp2 = self._match_features(kp1, des1, kp2, des2)
        if len(matches) < 8:
            logger.debug("Kamera transform: yetersiz match (%d)", len(matches))
            return transform

        matches = matches[:160]
        src_pts = np.float32([mkp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([mkp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        transform.match_count = len(matches)

        # 1. Homography (RANSAC)
        if self.camera_model == "homography":
            H_mat, inliers_h = cv2.findHomography(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=self.affine_ransac_threshold,
            )
            if H_mat is not None and inliers_h is not None:
                inlier_count = int(np.sum(inliers_h))
                inlier_ratio = inlier_count / len(matches)
                rep_err = self._reprojection_error_homography(src_pts, dst_pts, H_mat, inliers_h)
                if inlier_ratio >= self.min_inlier_ratio and rep_err <= self.max_reprojection_error:
                    transform.model_type = CameraModel.HOMOGRAPHY
                    transform.matrix = H_mat.astype(np.float32)
                    transform.inlier_count = inlier_count
                    transform.inlier_ratio = inlier_ratio
                    transform.reprojection_error = rep_err
                    transform.confidence = self._compute_confidence(
                        inlier_ratio, rep_err, len(matches),
                    )
                    return transform

        # 2. Affine fallback
        if self.affine_fallback:
            A_mat, inliers_a = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=self.affine_ransac_threshold,
            )
            if A_mat is not None and inliers_a is not None:
                inlier_count = int(np.sum(inliers_a))
                inlier_ratio = inlier_count / len(matches)
                rep_err = self._reprojection_error_affine(src_pts, dst_pts, A_mat, inliers_a)
                if inlier_count >= self.affine_min_matches and \
                   inlier_ratio >= self.min_inlier_ratio * 0.8 and \
                   rep_err <= self.max_reprojection_error:
                    transform.model_type = CameraModel.AFFINE
                    transform.matrix = A_mat.astype(np.float32)
                    transform.inlier_count = inlier_count
                    transform.inlier_ratio = inlier_ratio
                    transform.reprojection_error = rep_err
                    transform.confidence = self._compute_confidence(
                        inlier_ratio, rep_err, len(matches),
                    ) * 0.9
                    return transform

        # 3. Robust median shift fallback
        if self.median_shift_fallback:
            shifts = (dst_pts.reshape(-1, 2) - src_pts.reshape(-1, 2))
            median_shift = np.median(shifts, axis=0).astype(np.float32)
            deviations = np.linalg.norm(shifts - median_shift, axis=1)
            mad = np.median(deviations) if len(deviations) > 0 else 0.0
            if mad < 1e-6:
                mad = float(np.mean(deviations)) if len(deviations) > 0 else 1.0
            inlier_mask = deviations < 2.0 * mad if len(deviations) > 0 else \
                np.ones(len(deviations), dtype=bool)
            inlier_count = int(np.sum(inlier_mask))
            inlier_ratio = inlier_count / len(matches)

            if inlier_count >= 6:
                transform.model_type = CameraModel.SHIFT
                transform.shift = median_shift
                transform.inlier_count = inlier_count
                transform.inlier_ratio = inlier_ratio
                transform.reprojection_error = float(np.mean(deviations[inlier_mask])) if inlier_count > 0 else 0.0
                transform.confidence = self._compute_confidence(
                    inlier_ratio, transform.reprojection_error, len(matches),
                ) * 0.5
                return transform

        # 4. Guvenilir transform yok
        transform.model_type = CameraModel.NONE
        transform.confidence = 0.0
        logger.debug("Kamera transform: guvenilir model yok")
        return transform

    @staticmethod
    def _reprojection_error_homography(src_pts, dst_pts, H_mat, inliers):
        """Homography icin inlier reprojection hatasi."""
        if inliers is None:
            return 999.0
        inlier_mask = inliers.flatten().astype(bool)
        src_in = src_pts[inlier_mask].reshape(-1, 1, 2)
        dst_in = dst_pts[inlier_mask].reshape(-1, 1, 2)
        if len(src_in) == 0:
            return 999.0
        projected = cv2.perspectiveTransform(src_in, H_mat)
        errors = np.linalg.norm(projected - dst_in, axis=2)
        return float(np.median(errors))

    @staticmethod
    def _reprojection_error_affine(src_pts, dst_pts, A_mat, inliers):
        """Affine icin inlier reprojection hatasi."""
        if inliers is None:
            return 999.0
        inlier_mask = inliers.flatten().astype(bool)
        src_in = src_pts[inlier_mask].reshape(-1, 1, 2)
        dst_in = dst_pts[inlier_mask].reshape(-1, 1, 2)
        if len(src_in) == 0:
            return 999.0
        ones = np.ones((len(src_in), 1), dtype=np.float32)
        pts_h = src_in.reshape(-1, 2)
        pts_h = np.hstack([pts_h, ones])
        projected = (A_mat @ pts_h.T).T[:, :2].reshape(-1, 1, 2)
        errors = np.linalg.norm(projected - dst_in, axis=2)
        return float(np.median(errors))

    @staticmethod
    def _compute_confidence(inlier_ratio: float, rep_error: float, match_count: int) -> float:
        """Transform guven skorunu hesapla [0, 1]."""
        ratio_score = min(1.0, inlier_ratio / 0.6)
        error_score = max(0.0, 1.0 - rep_error / 10.0)
        count_score = min(1.0, match_count / 60.0)
        return float(0.4 * ratio_score + 0.3 * error_score + 0.3 * count_score)
