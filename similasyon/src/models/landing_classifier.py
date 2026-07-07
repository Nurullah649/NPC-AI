"""ResNet50 tabanlı iniş durumu sınıflandırıcısı.

Eski Class/Does_it_intersect.py içindeki ResNet50 binary classifier mantığını
temiz, config-driven şekilde yeniden uygular.

- UAP/UAI crop'u alır
- ResNet50 binary classifier çalıştırır
- Sigmoid çıktıyı p_inilebilir / p_inilemez olarak yorumlar
- Model yoksa crash etmez, available=False döndürür
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np


class LandingClassifier:
    """ResNet50 binary classifier for landing area cropları."""

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        landing_cfg = config.get("landing", {})
        self.use_classifier = landing_cfg.get("use_classifier", True)
        self.classifier_threshold = landing_cfg.get("classifier_threshold", 0.5)
        self.classifier_dropout = landing_cfg.get("classifier_dropout", 0.36)

        model_paths = config.get("model_paths", {})
        model_rel = model_paths.get("landing_classifier", "weights/landing/son_en_iyi_model.pth")
        self.model_path = self._resolve_path(model_rel)

        self.available = False
        self._model = None
        self._device = None
        self._transform = None

        if not self.use_classifier:
            self.logger.info("LandingClassifier: use_classifier=false, classifier devre dışı.")
            return

        if not os.path.exists(self.model_path):
            self.logger.warning(
                f"Landing classifier modeli bulunamadı: {self.model_path}\n"
                f"  Rule-based fallback kullanılacak."
            )
            return

        self._load_model()

    def _resolve_path(self, rel_path):
        candidates = [
            Path(rel_path),
            Path(".") / rel_path,
            Path(__file__).resolve().parent.parent.parent / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return str(Path(rel_path).resolve())

    def _load_model(self):
        """Modeli yükle (lazy, import anında crash yok)."""
        try:
            import torch
            import torchvision.transforms as transforms
            import torchvision.models as models

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ResNet50 mimarisi (weights=None ile pretrained download yapma)
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_ftrs, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.classifier_dropout),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid(),
            )

            state_dict = torch.load(self.model_path, map_location=self._device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self._model = model

            # Transform
            self._transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.available = True
            self.logger.info(f"Landing classifier loaded: {self.model_path} (device={self._device})")

        except Exception as e:
            self.logger.warning(f"Landing classifier yüklenemedi: {e}")
            self.available = False

    def predict_crop(self, image_bgr: np.ndarray, bbox: tuple) -> dict:
        """Bir UAP/UAI crop'u üzerinde tahmin yap.

        Args:
            image_bgr: BGR frame.
            bbox: (x1, y1, x2, y2) pixel coordinates.

        Returns:
            {
                "available": bool,
                "p_inilebilir": float,
                "p_inilemez": float,
                "status": "1" | "0" | None,
                "reason": str
            }
        """
        if not self.available or self._model is None:
            return {"available": False, "status": None, "reason": "model_missing"}

        if image_bgr is None:
            return {"available": False, "status": None, "reason": "no_image"}

        try:
            # Bbox clip
            H, W = image_bgr.shape[:2]
            x1 = max(0, min(int(bbox[0]), W - 1))
            y1 = max(0, min(int(bbox[1]), H - 1))
            x2 = max(0, min(int(bbox[2]), W - 1))
            y2 = max(0, min(int(bbox[3]), H - 1))

            if x1 >= x2 or y1 >= y2:
                return {"available": False, "status": None, "reason": "invalid_bbox"}

            # Crop
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return {"available": False, "status": None, "reason": "empty_crop"}

            # PIL Image'a çevir
            from PIL import Image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            # Transform ve inference
            import torch
            input_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = self._model(input_tensor)
                p_inilemez = float(output.item())  # Sigmoid output: 1 = inilemez
                p_inilebilir = 1.0 - p_inilemez

            status = "1" if p_inilebilir >= self.classifier_threshold else "0"

            return {
                "available": True,
                "p_inilebilir": p_inilebilir,
                "p_inilemez": p_inilemez,
                "status": status,
                "reason": "ok",
            }

        except Exception as e:
            self.logger.warning(f"Landing classifier predict_crop hatası: {e}")
            return {"available": False, "status": None, "reason": f"error: {e}"}
