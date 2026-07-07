"""DPVO standalone wrapper - DPVO'yu projeden bağımsız çalıştırmak için.

Bu modül DPVO'nun mevcut Class/DPVO kodunu import eder.
Eğer DPVO kurulu değilse, dummy bir implementasyon kullanılır.
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# DPVO import'larını dene
try:
    import sys
    # DPVO repo yolunu sys.path'e ekle
    dpvo_root = Path(__file__).resolve().parent.parent.parent / "third_party" / "DPVO"
    if not dpvo_root.exists():
        # Fallback: Class/DPVO
        dpvo_root = Path(__file__).resolve().parent.parent.parent.parent / "Class" / "DPVO"
    if dpvo_root.exists():
        sys.path.insert(0, str(dpvo_root))

    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    DPVO_AVAILABLE = True
    logger.info(f"DPVO modülü {dpvo_root} konumundan yüklendi.")
except ImportError as e:
    DPVO_AVAILABLE = False
    logger.warning(f"DPVO modülü yüklenemedi: {e}")
    logger.warning("Dummy DPVO kullanılacak.")


class DPVOStandalone:
    """DPVO standalone wrapper - her frame için pose döndürür."""

    def __init__(self, weight_path: str, config_path: str,
                 calib_path: str, ht: int, wd: int):
        self.weight_path = weight_path
        self.config_path = config_path
        self.calib_path = calib_path
        self.ht = ht
        self.wd = wd
        self.slam = None
        self.initialized = False

        if DPVO_AVAILABLE:
            self._init_dpvo()
        else:
            logger.warning("DPVO bulunamadı, dummy pose döndürülecek.")

    def _init_dpvo(self):
        """DPVO'yu başlat."""
        try:
            # Config dosyasını merge et
            if os.path.exists(self.config_path):
                cfg.merge_from_file(self.config_path)

            cfg.BUFFER_SIZE = 2048  # Hafıza tasarrufu

            self.slam = DPVO(cfg, self.weight_path, ht=self.ht, wd=self.wd, viz=False)
            self.initialized = True
            logger.info("DPVO başarıyla başlatıldı.")
        except Exception as e:
            logger.error(f"DPVO başlatılamadı: {e}")
            self.initialized = False

    def process_frame(self, idx: int, frame_path: str,
                      intrinsics: np.ndarray) -> tuple | None:
        """Bir frame işle ve (x, y, z) pose döndür."""
        if not self.initialized or self.slam is None:
            return None

        try:
            image = cv2.imread(frame_path)
            if image is None:
                return None

            # DPVO tensor formatına çevir
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            if cfg.MIXED_PRECISION:
                image_tensor = image_tensor.half()

            intrinsics_tensor = torch.from_numpy(intrinsics).cuda()

            with torch.no_grad():
                self.slam(idx, image_tensor, intrinsics_tensor)

            # Pose al
            current_pose = self.slam.get_current_pose()
            x = float(current_pose[0, 3])
            y = float(current_pose[1, 3])
            z = float(current_pose[2, 3])

            return (x, y, z)

        except Exception as e:
            logger.error(f"DPVO frame işleme hatası: {e}")
            return None

    def terminate(self):
        """DPVO'yu sonlandır."""
        if self.slam is not None:
            try:
                with torch.no_grad():
                    trajectory = self.slam.terminate()
                torch.cuda.empty_cache()
                return trajectory
            except Exception as e:
                logger.error(f"DPVO terminate hatası: {e}")
        return None
