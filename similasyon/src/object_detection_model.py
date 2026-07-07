"""
Ana Nesne Tespit Modeli - Tüm görevlerin birleştiği ana akış.

Görev 1: YOLO deteksiyon + hareket durumu + iniş durumu
Görev 2: DPVO pozisyon kestirimi
Görev 3: Referans görüntü eşleme

Her frame'de ObjectDetectionModel.process() veya detect() çağrılır.
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction
from .frame_predictions import FramePredictions

from .models.detector_yolo import DetectorYOLO
from .models.motion_classifier import MotionClassifier
from .models.landing_status import LandingStatusResolver
from .models.positioning_dpvo import PositioningDPVO
from .models.reference_matcher import ReferenceMatcher


class ObjectDetectionModel:
    """Ana model sınıfı - tüm görevleri tek akışta birleştirir."""

    def __init__(self, config_path=None):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Config yükle
        if config_path is None:
            config_path = self._find_config()
        self.config = self._load_config(config_path)

        # Alt modülleri başlat
        self.logger.info("Alt modüller başlatılıyor...")
        self.detector = DetectorYOLO(self.config)
        self.motion_classifier = MotionClassifier(self.config)
        self.landing_resolver = LandingStatusResolver(self.config)
        self.positioning = PositioningDPVO(self.config)
        self.reference_matcher = ReferenceMatcher(self.config)

        # Debug ayarları
        debug_cfg = self.config.get("debug", {})
        self.save_visuals = debug_cfg.get("save_visuals", False)
        self.save_payloads = debug_cfg.get("save_payloads", True)

        # Referans cache
        self.active_refs = []
        self.ref_image_paths = {}
        self.ref_cache_dir = None

        self.frame_count = 0

        self.logger.info("ObjectDetectionModel başarıyla başlatıldı.")

    def _find_config(self):
        """settings.yaml dosyasını bul."""
        candidates = [
            "config/settings.yaml",
            "../config/settings.yaml",
            str(Path(__file__).resolve().parent.parent / "config" / "settings.yaml"),
            str(Path(__file__).resolve().parent.parent.parent / "similasyon" / "config" / "settings.yaml"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        # Varsayılan
        return "config/settings.yaml"

    def _load_config(self, config_path: str) -> dict:
        """YAML config dosyasını yükle."""
        if not os.path.exists(config_path):
            self.logger.warning(f"Config dosyası bulunamadı: {config_path}, varsayılanlar kullanılacak.")
            return {}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.logger.info(f"Config yüklendi: {config_path}")
        return config

    def set_references(self, active_refs: list, ref_image_paths: dict, cache_dir: str):
        """Aktif referansları ayarla ve cache'le.

        Args:
            active_refs: Referans URL listesi.
            ref_image_paths: {ref_url: image_url} dict.
            cache_dir: Referans görüntülerinin kaydedileceği dizin.
        """
        self.active_refs = active_refs or []
        self.ref_image_paths = ref_image_paths or {}
        self.ref_cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Referansları önceden cache'le
        for ref in self.active_refs:
            ref_url = ref if isinstance(ref, str) else ref.get('url', '')
            if not ref_url:
                continue
            img_url = self.ref_image_paths.get(ref_url, '')
            if not img_url:
                continue

            # Referans görüntüsünü indir
            ref_filename = ref_url.split("/")[-1] + ".jpg"
            ref_path = os.path.join(cache_dir, ref_filename)

            if not os.path.exists(ref_path) and img_url:
                try:
                    import requests
                    response = requests.get(img_url, timeout=60)
                    with open(ref_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    self.logger.error(f"Referans indirilemedi {ref_url}: {e}")
                    continue

            # Feature'ları önceden hesapla
            self.reference_matcher.precompute_reference(ref_url, ref_path)

        self.logger.info(f"{len(self.active_refs)} referans hazırlandı.")

    def detect(self, frame_img: np.ndarray,
               frame_url: str = "",
               health_status: str = "0",
               gt_x: float = 0.0, gt_y: float = 0.0, gt_z: float = 0.0,
               frame_path: str = None) -> FramePredictions:
        """Ana detection akışı - tüm görevleri çalıştırır.

        Args:
            frame_img: BGR frame (H, W, 3)
            frame_url: Frame'in sunucudaki URL'si
            health_status: '0' veya '1'
            gt_x, gt_y, gt_z: Health=1 iken gelen ground truth değerler
            frame_path: Frame'in yerel dosya yolu (DPVO için)

        Returns:
            FramePredictions objesi (payload'a çevrilebilir).
        """
        self.frame_count += 1
        H, W = frame_img.shape[:2]

        # --- Görev 1: YOLO Deteksiyonu ---
        detections = self.detector.detect(frame_img)

        # --- Görev 1 ek: Hareket Durumu ---
        gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        detections = self.motion_classifier.update(detections, gray)

        # --- Görev 1 ek: İniş Durumu ---
        detections = self.landing_resolver.resolve(detections, frame_img.shape)

        # --- Görev 2: Pozisyon Kestirimi ---
        if frame_path and os.path.exists(frame_path):
            tx, ty, tz = self.positioning.process_frame(
                self.frame_count, frame_path,
                health_status, gt_x, gt_y, gt_z
            )
        else:
            tx, ty, tz = gt_x, gt_y, gt_z

        # --- FramePredictions oluştur ---
        pred = FramePredictions(
            frame_url=frame_url,
            image_url="",  # doldurulacak
            video_name="",
            gt_translation_x=gt_x,
            gt_translation_y=gt_y,
            gt_translation_z=gt_z,
        )

        # DetectedObject'leri ekle
        for det in detections:
            bbox = det['bbox']
            d_obj = DetectedObject(
                cls=det['cls'],
                landing_status=det.get('landing_status', "-1"),
                moving_status=det.get('moving_status', "-1"),
                top_left_x=bbox[0],
                top_left_y=bbox[1],
                bottom_right_x=bbox[2],
                bottom_right_y=bbox[3],
            )
            pred.add_detected_object(d_obj)

        # DetectedTranslation ekle
        t_obj = DetectedTranslation(tx, ty, tz)
        pred.add_translation_object(t_obj)

        # --- Görev 3: Referans Eşleme ---
        for ref in self.active_refs:
            ref_url = ref if isinstance(ref, str) else ref.get('url', '')
            if not ref_url:
                continue

            img_url = self.ref_image_paths.get(ref_url, '')
            if not img_url:
                continue

            # Cache'te referans yolu
            ref_path = None
            if self.ref_cache_dir:
                ref_filename = ref_url.split("/")[-1] + ".jpg"
                ref_path = os.path.join(self.ref_cache_dir, ref_filename)

            # Eşleme yap
            bbox = self.reference_matcher.match(ref_url, ref_path, frame_img)

            if bbox is not None:
                ref_pred = ReferencePrediction(
                    reference_url=ref_url,
                    frame_url=frame_url,
                    top_left_x=bbox[0],
                    top_left_y=bbox[1],
                    bottom_right_x=bbox[2],
                    bottom_right_y=bbox[3],
                )
                pred.add_reference_prediction(ref_pred)

        # Debug: payload validasyonu
        self._validate_payload(pred)

        # Debug: görsel kaydet
        if self.save_visuals:
            self._save_debug_visual(frame_img, detections, pred, self.frame_count)

        return pred

    def _validate_payload(self, pred: FramePredictions):
        """Payload'ı doğrula, hataları logla."""
        for d_obj in pred.detected_objects:
            # moving_status kontrol
            if d_obj.moving_status not in ["-1", "0", "1"]:
                self.logger.warning(f"Geçersiz moving_status: {d_obj.moving_status}")

        for t_obj in pred.detected_translations:
            if t_obj.translation_z is None:
                self.logger.warning("translation_z None!")

        # reference_predictions
        if self.active_refs and not pred.reference_predictions:
            self.logger.debug("Aktif referans var ama eşleşme bulunamadı.")

    def _save_debug_visual(self, frame, detections, pred, frame_count):
        """Debug için görsel kaydet."""
        debug_dir = f"_debug/session_{frame_count // 100}"
        os.makedirs(debug_dir, exist_ok=True)

        vis = frame.copy()
        for det in detections:
            bbox = det['bbox']
            cls_name = det['cls_name']
            cv2.rectangle(vis, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            label = f"{cls_name} {det.get('moving_status', '?')} {det.get('landing_status', '?')}"
            cv2.putText(vis, label, (int(bbox[0]), int(bbox[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(f"{debug_dir}/frame_{frame_count:06d}.jpg", vis)
