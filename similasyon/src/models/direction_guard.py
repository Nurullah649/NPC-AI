"""DPVO kalibrasyonu için sanity guard.

Eski Class/Calculate_Direction.py mantığını temiz, güvenli şekilde yeniden uygular.
Sim(3) ana yöntemini korur, sadece kötü kalibrasyonu engeller.
"""
import logging
import numpy as np


class DirectionGuard:
    """DPVO kalibrasyon kalitesini doğrular.

    GT ve DPVO raw trajectory'leri arasında:
    - Toplam yön benzerliği
    - Scale factor sınırları
    - X/Y işaret tutarlılığı
    kontrolleri yapar.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        dpvo_cfg = config.get("dpvo", {})
        self.use_guard = dpvo_cfg.get("use_direction_guard", True)
        self.min_points = dpvo_cfg.get("direction_guard_min_points", 30)
        self.max_allowed_scale = dpvo_cfg.get("max_allowed_scale", 50.0)
        self.min_allowed_scale = dpvo_cfg.get("min_allowed_scale", 0.02)
        self.min_direction_similarity = dpvo_cfg.get("min_direction_similarity", -0.2)

        if self.use_guard:
            self.logger.info(
                f"DirectionGuard aktif: min_scale={self.min_allowed_scale}, "
                f"max_scale={self.max_allowed_scale}, "
                f"min_similarity={self.min_direction_similarity}"
            )

    def evaluate(self, gt_buffer: list, dpvo_buffer: list) -> dict:
        """GT ve DPVO buffer'larını karşılaştır.

        Args:
            gt_buffer: [(x, y, z), ...] GT noktaları.
            dpvo_buffer: [(x, y, z), ...] DPVO raw noktaları.

        Returns:
            {
                "ok": bool,
                "direction_code": int,
                "scale_factor": float,
                "direction_similarity": float,
                "reason": str
            }
        """
        if not self.use_guard:
            return {"ok": True, "direction_code": 4, "scale_factor": 1.0,
                    "direction_similarity": 1.0, "reason": "guard_disabled"}

        # Guard'lar
        if len(gt_buffer) < self.min_points:
            return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                    "direction_similarity": 0.0, "reason": "insufficient_points"}

        if len(dpvo_buffer) < self.min_points:
            return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                    "direction_similarity": 0.0, "reason": "insufficient_dpvo_points"}

        try:
            gt = np.array(gt_buffer, dtype=np.float64)
            dpvo = np.array(dpvo_buffer, dtype=np.float64)

            # NaN/Inf kontrolü
            if np.any(np.isnan(gt)) or np.any(np.isinf(gt)):
                return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                        "direction_similarity": 0.0, "reason": "gt_contains_nan_inf"}
            if np.any(np.isnan(dpvo)) or np.any(np.isinf(dpvo)):
                return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                        "direction_similarity": 0.0, "reason": "dpvo_contains_nan_inf"}

            # Scale factor hesapla (median of frame-frame norm ratios)
            gt_diffs = np.diff(gt, axis=0)
            dpvo_diffs = np.diff(dpvo, axis=0)

            gt_norms = np.linalg.norm(gt_diffs, axis=1)
            dpvo_norms = np.linalg.norm(dpvo_diffs, axis=1)

            # Zero norm guard
            valid = (gt_norms > 1e-10) & (dpvo_norms > 1e-10)
            if not np.any(valid):
                return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                        "direction_similarity": 0.0, "reason": "zero_movement"}

            scales = dpvo_norms[valid] / gt_norms[valid]
            scale_factor = float(np.median(scales))

            # Scale sınır kontrolü
            if scale_factor < self.min_allowed_scale or scale_factor > self.max_allowed_scale:
                return {"ok": False, "direction_code": -1, "scale_factor": scale_factor,
                        "direction_similarity": 0.0,
                        "reason": f"scale_out_of_bounds ({scale_factor:.4f})"}

            # Toplam yön benzerliği
            total_gt = np.sum(gt_diffs, axis=0)
            total_dpvo = np.sum(dpvo_diffs, axis=0)

            gt_norm_total = np.linalg.norm(total_gt)
            dpvo_norm_total = np.linalg.norm(total_dpvo)

            if gt_norm_total < 1e-10 or dpvo_norm_total < 1e-10:
                direction_similarity = 0.0
            else:
                gt_dir = total_gt / gt_norm_total
                dpvo_dir = total_dpvo / dpvo_norm_total
                direction_similarity = float(np.dot(gt_dir, dpvo_dir))

            # Direction code (eski Calculate_Direction mantığı)
            if direction_similarity < self.min_direction_similarity:
                return {"ok": False, "direction_code": 0, "scale_factor": scale_factor,
                        "direction_similarity": direction_similarity,
                        "reason": f"low_similarity ({direction_similarity:.4f})"}

            # X/Y işaret kodlaması
            if direction_similarity < 0:
                direction_code = 0
            else:
                x_sim = np.sign(total_gt[0]) == np.sign(total_dpvo[0]) if abs(total_gt[0]) > 1e-10 else True
                y_sim = np.sign(total_gt[1]) == np.sign(total_dpvo[1]) if abs(total_gt[1]) > 1e-10 else True
                z_sim = np.sign(total_gt[2]) == np.sign(total_dpvo[2]) if len(total_gt) > 2 and abs(total_gt[2]) > 1e-10 else True

                if not x_sim and y_sim and z_sim:
                    direction_code = 1
                elif x_sim and not y_sim and z_sim:
                    direction_code = 2
                elif not x_sim and not y_sim and z_sim:
                    direction_code = 3
                else:
                    direction_code = 4  # Her şey uyumlu

            return {
                "ok": True,
                "direction_code": direction_code,
                "scale_factor": scale_factor,
                "direction_similarity": direction_similarity,
                "reason": "ok",
            }

        except Exception as e:
            self.logger.warning(f"DirectionGuard değerlendirme hatası: {e}")
            return {"ok": False, "direction_code": -1, "scale_factor": 0.0,
                    "direction_similarity": 0.0, "reason": f"error: {e}"}
