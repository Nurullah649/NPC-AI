"""Opsiyonel CLAHE ön işleme modülü.

Eski Class/Process_image.py mantığını temiz, config-driven şekilde yeniden uygular.
- Görüntüyü 2x3 tile'a böler
- Her tile için brightness kontrolü yapar
- Karanlık tile'lara CLAHE uygular (LAB renk uzayında sadece L kanalı)
- Tile'ları birleştirir
- Diske yazma yapmaz
"""
import logging
import numpy as np
import cv2


class ImagePreprocessor:
    """Opsiyonel CLAHE preprocessing.

    Config:
        preprocessing:
          use_clahe: false
          clahe_clip_limit: 2.0
          clahe_tile_grid_size: 8
          split_rows: 2
          split_cols: 3
          brightness_threshold: 50
          apply_to_detector: false
          apply_to_landing_classifier: true
          apply_to_reference_matcher: false
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        pre_cfg = config.get("preprocessing", {})
        self.use_clahe = pre_cfg.get("use_clahe", False)
        self.clahe_clip_limit = pre_cfg.get("clahe_clip_limit", 2.0)
        self.clahe_tile_grid_size = pre_cfg.get("clahe_tile_grid_size", 8)
        self.split_rows = pre_cfg.get("split_rows", 2)
        self.split_cols = pre_cfg.get("split_cols", 3)
        self.brightness_threshold = pre_cfg.get("brightness_threshold", 50)
        self.apply_to_detector = pre_cfg.get("apply_to_detector", False)
        self.apply_to_landing_classifier = pre_cfg.get("apply_to_landing_classifier", True)
        self.apply_to_reference_matcher = pre_cfg.get("apply_to_reference_matcher", False)

        if self.use_clahe:
            self.logger.info(
                f"CLAHE preprocessing aktif: clip={self.clahe_clip_limit}, "
                f"grid={self.clahe_tile_grid_size}x{self.clahe_tile_grid_size}, "
                f"split={self.split_rows}x{self.split_cols}"
            )

    def apply(self, image_bgr: np.ndarray, purpose: str = "generic") -> np.ndarray:
        """İşleme uygula.

        Args:
            image_bgr: BGR formatında numpy array.
            purpose: Hangi modül için çağrıldığı ("detector", "landing_classifier", "reference_matcher").

        Returns:
            İşlenmiş BGR numpy array veya hata durumunda orijinal image.
        """
        if image_bgr is None:
            return None

        if not self.use_clahe:
            return image_bgr

        # Purpose-specific config check
        purpose_map = {
            "detector": self.apply_to_detector,
            "landing_classifier": self.apply_to_landing_classifier,
            "reference_matcher": self.apply_to_reference_matcher,
        }
        if not purpose_map.get(purpose, True):
            return image_bgr

        try:
            return self._apply_clahe_tiled(image_bgr)
        except Exception as e:
            self.logger.warning(f"CLAHE preprocessing hatası: {e}")
            return image_bgr

    def _apply_clahe_tiled(self, image_bgr: np.ndarray) -> np.ndarray:
        """2x3 tiled CLAHE uygula.

        Tile sınırlarını tam piksel keskinliği ile hesaplar.
        Edge tile'lar image boyutuna kadar uzar.
        """
        h, w = image_bgr.shape[:2]
        rows = self.split_rows
        cols = self.split_cols

        # Tile sınırlarını hesapla (keskin piksel sınırları)
        row_breaks = [0]
        for i in range(1, rows):
            row_breaks.append(h * i // rows)
        row_breaks.append(h)

        col_breaks = [0]
        for j in range(1, cols):
            col_breaks.append(w * j // cols)
        col_breaks.append(w)

        parts = []
        for ri in range(rows):
            for cj in range(cols):
                tile = image_bgr[
                    row_breaks[ri] : row_breaks[ri + 1],
                    col_breaks[cj] : col_breaks[cj + 1]
                ]
                processed = self._process_tile(tile)
                parts.append(processed)

        # Merge
        rows_merged = []
        for ri in range(rows):
            start = ri * cols
            row = np.hstack(parts[start : start + cols])
            rows_merged.append(row)
        merged = np.vstack(rows_merged)

        return merged

    def _process_tile(self, tile_bgr: np.ndarray) -> np.ndarray:
        """Tek tile'ı işle: brightness kontrolü + opsiyonel CLAHE."""
        # Brightness hesapla (histogram peak)
        gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        brightness = float(np.argmax(hist))

        if brightness > self.brightness_threshold:
            return tile_bgr  # Aydınlık, olduğu gibi bırak

        # Karanlık: CLAHE uygula (LAB renk uzayında sadece L kanalı)
        lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size)
        )
        l2 = clahe.apply(l)
        merged_lab = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def is_active_for(self, purpose: str) -> bool:
        """Belirli bir modül için preprocessing aktif mi?"""
        if not self.use_clahe:
            return False
        purpose_map = {
            "detector": self.apply_to_detector,
            "landing_classifier": self.apply_to_landing_classifier,
            "reference_matcher": self.apply_to_reference_matcher,
        }
        return purpose_map.get(purpose, False)
