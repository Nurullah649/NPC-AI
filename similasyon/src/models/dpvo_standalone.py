"""DPVO standalone wrapper - DPVO'yu projeden bağımsız çalıştırmak için.

Bu modül DPVO'nun mevcut Class/DPVO kodunu import eder.
Eğer DPVO kurulu değilse, dummy bir implementasyon kullanılır.
"""
import logging
import os
import random
import gc
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
                 calib_path: str, ht: int, wd: int, *, seed: int = 0,
                 cuda_empty_cache_every: int = 0, buffer_size: int | None = None):
        self.weight_path = weight_path
        self.config_path = config_path
        self.calib_path = calib_path
        self.ht = ht
        self.wd = wd
        self.seed = int(seed)
        self.cuda_empty_cache_every = max(0, int(cuda_empty_cache_every))
        self.buffer_size = buffer_size
        self.cfg = None
        self.slam = None
        self.initialized = False

        if DPVO_AVAILABLE:
            self._init_dpvo()
        else:
            logger.warning("DPVO bulunamadı, dummy pose döndürülecek.")

    def _init_dpvo(self):
        """DPVO'yu başlat."""
        try:
            # Global cfg'yi değiştirmeden bu instance'a özel config kullan.
            self.cfg = cfg.clone()
            if os.path.exists(self.config_path):
                self.cfg.merge_from_file(self.config_path)

            if self.buffer_size is not None:
                self.cfg.BUFFER_SIZE = int(self.buffer_size)

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            self.slam = DPVO(self.cfg, self.weight_path, ht=self.ht, wd=self.wd, viz=False)
            self.initialized = True
            logger.info(
                "DPVO başarıyla başlatıldı. seed=%d, buffer=%d, mixed_precision=%s",
                self.seed,
                self.cfg.BUFFER_SIZE,
                self.cfg.MIXED_PRECISION,
            )
        except Exception as e:
            logger.error(f"DPVO başlatılamadı: {e}")
            self.initialized = False

    def process_frame(self, idx: int, frame_path: str | Path | np.ndarray,
                      intrinsics: np.ndarray) -> tuple | None:
        """Bir frame işle ve (x, y, z) pose döndür."""
        if not self.initialized or self.slam is None:
            return None

        try:
            if isinstance(frame_path, np.ndarray):
                image = frame_path
            else:
                image = cv2.imread(str(frame_path))
            if image is None:
                return None
            image = np.ascontiguousarray(image)

            # DPVO tensor formatına çevir
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
            if self.cfg.MIXED_PRECISION:
                image_tensor = image_tensor.half()

            intrinsics_tensor = torch.as_tensor(
                intrinsics, dtype=torch.float32, device="cuda"
            )

            with torch.no_grad():
                self.slam(idx, image_tensor, intrinsics_tensor)

            # Pose al
            current_pose = self.slam.get_current_pose()
            x = float(current_pose[0, 3])
            y = float(current_pose[1, 3])
            z = float(current_pose[2, 3])

            del image_tensor, intrinsics_tensor
            if self.cuda_empty_cache_every and (idx + 1) % self.cuda_empty_cache_every == 0:
                gc.collect()
                torch.cuda.empty_cache()

            return (x, y, z)

        except Exception as e:
            logger.error(f"DPVO frame işleme hatası: {e}")
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
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
