"""Gerçek HYZ 2026 model orkestratörü.

Tüm alt modülleri başlatır ve her frame'de sırasıyla:
  Görev 1: DetectorYOLO → MotionClassifier → LandingStatusResolver
  Görev 2: PositioningDPVO
  Görev 3: ReferenceMatcher
"""
import logging
import os
import time
from pathlib import Path

import cv2
import requests

from .config_loader import load_settings
from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction
from .frame_predictions import FramePredictions


class ObjectDetectionModel:
    """Ana model sınıfı - tüm görevleri tek akışta birleştirir."""

    def __init__(self, evaluation_server_url: str, allow_dummy: bool = False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.evaluation_server = evaluation_server_url.rstrip("/") + "/"
        self.allow_dummy = allow_dummy
        self.frame_idx = 0

        # Config yükle
        self.settings = load_settings()
        self.logger.info("Config yüklendi.")

        # Alt modülleri başlat
        self._init_modules()

    def _init_modules(self):
        """Tüm alt modülleri başlat. allow_dummy=True ise başarısız modülleri dummy ile değiştir."""
        from .models.detector_yolo import DetectorYOLO
        from .models.motion_classifier import MotionClassifier
        from .models.landing_status import LandingStatusResolver
        from .models.positioning_dpvo import PositioningDPVO
        from .models.reference_pipeline import ReferenceDetectionPipeline

        # DetectorYOLO
        try:
            self.detector = DetectorYOLO(self.settings)
            self.logger.info("✅ DetectorYOLO başlatıldı.")
        except FileNotFoundError as e:
            if self.allow_dummy:
                self.logger.warning(f"⚠️ DetectorYOLO yüklenemedi: {e}")
                self.logger.warning("   Dummy detector kullanılacak (dry-run modu).")
                self.detector = _DummyDetector()
            else:
                raise

        # MotionClassifier (her zaman çalışır - bağımlılığı yok)
        self.motion = MotionClassifier(self.settings)
        self.logger.info("✅ MotionClassifier başlatıldı.")

        # LandingStatusResolver
        self.landing = LandingStatusResolver(self.settings)
        self.logger.info("✅ LandingStatusResolver başlatıldı.")

        # PositioningDPVO
        try:
            self.positioning = PositioningDPVO(self.settings)
            self.logger.info("✅ PositioningDPVO başlatıldı.")
        except Exception as e:
            if self.allow_dummy:
                self.logger.warning(f"⚠️ PositioningDPVO yüklenemedi: {e}")
                self.logger.warning("   Dummy positioning kullanılacak (dry-run modu).")
                self.positioning = _DummyPositioning()
            else:
                raise

        # ReferenceMatcher + ROI/tracker/sahne-hafizasi pipeline'i
        try:
            self.reference_matcher = ReferenceDetectionPipeline(self.settings)
            self.logger.info("✅ ReferenceDetectionPipeline başlatıldı.")
        except Exception as e:
            if self.allow_dummy:
                self.logger.warning(f"⚠️ ReferenceMatcher yüklenemedi: {e}")
                self.logger.warning("   Dummy reference matcher kullanılacak (dry-run modu).")
                self.reference_matcher = _DummyReferenceMatcher()
            else:
                raise

        # ImagePreprocessor
        from .models.image_preprocess import ImagePreprocessor
        self.preprocessor = ImagePreprocessor(self.settings)
        self.logger.info("✅ ImagePreprocessor başlatıldı.")

        self.logger.info("✅ Tüm modüller başlatıldı.")

    def register_references(self, references, ref_image_paths):
        """Sunucunun yayinladigi tum referanslari bir kez feature cache'e al."""
        register = getattr(self.reference_matcher, "register_references", None)
        if callable(register):
            register(references or [], ref_image_paths or {})

    @staticmethod
    def _frame_sequence(image_url, fallback):
        """frame_001956.webp benzeri URL'den gercek sira numarasini al."""
        try:
            stem = Path(str(image_url).split("/")[-1]).stem
            return int(stem.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            return int(fallback)

    def _resize_for_detector(self, image):
        """Detector/SAHI girişini opsiyonel olarak küçült.

        Sunucuya gönderilecek bbox koordinatları orijinal frame koordinatında
        kalmalı. Bu yüzden detector çıktılarını daha sonra ters ölçekliyoruz.
        Config:
            detector:
              resize_max_height: 1080  # 0/None ise kapalı
        """
        det_cfg = self.settings.get("detector", {})
        max_h = det_cfg.get("resize_max_height")
        if not max_h:
            return image, 1.0, 1.0

        try:
            max_h = int(max_h)
        except (TypeError, ValueError):
            return image, 1.0, 1.0

        if max_h <= 0:
            return image, 1.0, 1.0

        h, w = image.shape[:2]
        if h <= max_h:
            return image, 1.0, 1.0

        scale = max_h / float(h)
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(image, (new_w, max_h), interpolation=cv2.INTER_AREA)
        sx = w / float(new_w)
        sy = h / float(max_h)
        return resized, sx, sy

    @staticmethod
    def _scale_detection_bboxes(detections, sx, sy, original_shape):
        """Detector çıktılarını orijinal frame koordinatlarına geri ölçekle."""
        if sx == 1.0 and sy == 1.0:
            return detections

        h, w = original_shape[:2]
        scaled = []
        for det in detections:
            det = dict(det)
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0.0, min(float(x1) * sx, w - 1))
            y1 = max(0.0, min(float(y1) * sy, h - 1))
            x2 = max(0.0, min(float(x2) * sx, w - 1))
            y2 = max(0.0, min(float(y2) * sy, h - 1))
            if x1 >= x2 or y1 >= y2:
                continue
            det["bbox"] = (x1, y1, x2, y2)
            scaled.append(det)
        return scaled

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        """Frame görüntüsünü indir. Başarılı olursa True döndür."""
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]

        if image_name not in images_files:
            headers = {'Authorization': f'Token {auth_token}'} if auth_token else {}
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, headers=headers, timeout=60)
                    response.raise_for_status()
                    img_bytes = response.content
                    with open(os.path.join(images_folder, image_name), 'wb') as img_file:
                        img_file.write(img_bytes)
                    t2 = time.perf_counter()
                    logging.getLogger(__name__).info(
                        f'{img_url} - Downloaded in {t2 - t1:.2f}s to {images_folder + image_name}'
                    )
                    return True
                except requests.exceptions.RequestException as e:
                    logging.getLogger(__name__).error(
                        f"Download failed for {img_url} (attempt {attempt + 1}): {e}"
                    )
                    time.sleep(wait_time)
                    wait_time *= 2
            logging.getLogger(__name__).error(f"Failed to download {img_url} after {retries} attempts.")
            return False
        else:
            logging.getLogger(__name__).info(f'{image_name} already exists, skipping download.')
            return True

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files,
                active_refs=None, ref_image_paths=None, auth_token=None):
        """Frame işleme ana akışı.

        Args:
            prediction: FramePredictions objesi
            evaluation_server_url: Sunucu URL'si
            health_status: '0', '1' veya None
            images_folder: İndirilen frame'lerin bulunduğu klasör
            images_files: Klasördeki dosya listesi
            active_refs: Aktif referans listesi (Görev 3)
            ref_image_paths: {ref_url: yerel_dosya_yolu} dict
            auth_token: Auth token

        Returns:
            Güncellenmiş FramePredictions objesi
        """
        # Frame'i indir
        img_url = evaluation_server_url.rstrip("/") + "/media" + prediction.image_url
        self.download_image(img_url, images_folder, images_files, auth_token=auth_token)

        frame_image_path = os.path.join(images_folder, prediction.image_url.split("/")[-1])

        # FramePredictions'in image_url ve video_name'ini güncelle (create_payload için)
        prediction.image_url = prediction.image_url
        prediction.video_name = prediction.video_name

        # Detect çağır
        frame_results = self.detect(
            prediction=prediction,
            health_status=health_status,
            active_refs=active_refs or [],
            ref_image_paths=ref_image_paths or {},
            frame_image_path=frame_image_path,
        )

        return frame_results

    def detect(self, prediction, health_status, active_refs=None, ref_image_paths=None, frame_image_path=None):
        """Tüm görevları sırasıyla çalıştır.

        Args:
            prediction: FramePredictions objesi
            health_status: '0', '1' veya None
            active_refs: Aktif referans listesi
            ref_image_paths: {ref_url: yerel_dosya_yolu}
            frame_image_path: Frame'in yerel yolu

        Returns:
            Güncellenmiş FramePredictions
        """
        self.frame_idx += 1
        active_refs = active_refs or []
        ref_image_paths = ref_image_paths or {}

        # Frame'i oku
        image = None
        if frame_image_path and os.path.exists(frame_image_path):
            image = cv2.imread(frame_image_path)
            if image is None:
                self.logger.error(f"Frame okunamadı: {frame_image_path}")

        # --- Görev 1: YOLO Deteksiyonu ---
        detections = []
        if image is not None:
            # Opsiyonel preprocessing (detector)
            detect_img = image
            if self.preprocessor.is_active_for("detector"):
                processed = self.preprocessor.apply(image, purpose="detector")
                if processed is not None:
                    detect_img = processed
            detect_img, det_sx, det_sy = self._resize_for_detector(detect_img)
            try:
                detections = self.detector.detect(detect_img)
                detections = self._scale_detection_bboxes(detections, det_sx, det_sy, image.shape)
                self.logger.debug(f"YOLO: {len(detections)} nesne tespit edildi.")
            except Exception as e:
                self.logger.error(f"DetectorYOLO hatası: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        # --- Görev 1 ek: Hareket Durumu ---
        if image is not None and detections:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detections = self.motion.update(detections, gray)
            except Exception as e:
                self.logger.error(f"MotionClassifier hatası: {e}")

        # --- Görev 1 ek: İniş Durumu ---
        if image is not None and detections:
            try:
                detections = self.landing.resolve(detections, image)
            except Exception as e:
                self.logger.error(f"LandingStatusResolver hatası: {e}")

        # Her detection için DetectedObject oluştur
        for det in detections:
            cls_id = det["cls"]
            landing_status = det.get("landing_status", "-1")
            moving_status = det.get("moving_status", "-1")
            bbox = det["bbox"]

            # cls tuple formatına çevir (resmi API: (cls_id,) formatında)
            cls_tuple = (cls_id,)

            d_obj = DetectedObject(
                cls=cls_tuple,
                landing_status=landing_status,
                moving_status=moving_status,
                top_left_x=bbox[0],
                top_left_y=bbox[1],
                bottom_right_x=bbox[2],
                bottom_right_y=bbox[3],
                track_id=det.get("_track_id"),
                _motion_score=det.get("_motion_score"),
            )
            prediction.add_detected_object(d_obj)

        # --- Görev 2: Pozisyon Kestirimi ---
        try:
            if health_status is None:
                self.logger.info("health_status=None, translation eklenmiyor.")
            elif health_status == '1' or health_status == 1:
                # GT passthrough - PositioningDPVO kalibrasyon için yine de çağrılır
                gt_x = float(prediction.gt_translation_x) if prediction.gt_translation_x is not None else 0.0
                gt_y = float(prediction.gt_translation_y) if prediction.gt_translation_y is not None else 0.0
                gt_z = float(prediction.gt_translation_z) if prediction.gt_translation_z is not None else 0.0

                if frame_image_path and os.path.exists(frame_image_path):
                    tx, ty, tz = self.positioning.process_frame(
                        self.frame_idx, frame_image_path, '1', gt_x, gt_y, gt_z,
                        image=image,
                    )
                else:
                    tx, ty, tz = gt_x, gt_y, gt_z

                prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
            else:
                # health_status == '0': DPVO/fallback tahmini
                gt_x = float(prediction.gt_translation_x) if prediction.gt_translation_x is not None else 0.0
                gt_y = float(prediction.gt_translation_y) if prediction.gt_translation_y is not None else 0.0
                gt_z = float(prediction.gt_translation_z) if prediction.gt_translation_z is not None else 0.0

                if frame_image_path and os.path.exists(frame_image_path):
                    tx, ty, tz = self.positioning.process_frame(
                        self.frame_idx, frame_image_path, '0', gt_x, gt_y, gt_z,
                        image=image,
                    )
                else:
                    tx, ty, tz = gt_x, gt_y, gt_z

                prediction.add_translation_object(DetectedTranslation(tx, ty, tz))
        except Exception as e:
            self.logger.error(f"PositioningDPVO hatası: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Bir modül hatasında koordinatı aniden (0,0,0)'a sıçratma.
            last_position = getattr(self.positioning, "last_known_position", (0.0, 0.0, 0.0))
            prediction.add_translation_object(
                DetectedTranslation(
                    float(last_position[0]),
                    float(last_position[1]),
                    float(last_position[2]),
                )
            )

        # --- Görev 3: Referans Nesne Tespiti ---
        if image is not None:
            for ref in active_refs:
                try:
                    ref_url = ref.get('url', '') if isinstance(ref, dict) else ''
                    if not ref_url:
                        continue

                    # Pencere kontrolü (frame aralığı)
                    if isinstance(ref, dict):
                        start_img = ref.get('frame_start_image_url', '')
                        end_img = ref.get('frame_end_image_url', '')
                        if start_img and end_img:
                            if not (start_img <= prediction.image_url <= end_img):
                                continue

                    ref_path = ref_image_paths.get(ref_url)
                    if not ref_path or not os.path.exists(ref_path):
                        self.logger.debug(f"Referans yolu yok: {ref_url} -> {ref_path}")
                        continue

                    match_reference = getattr(
                        self.reference_matcher, 'match_reference', None
                    )
                    if callable(match_reference):
                        ref_result = match_reference(
                            reference=ref,
                            ref_path=ref_path,
                            frame=image,
                            detections=detections,
                            frame_idx=self._frame_sequence(
                                prediction.image_url, self.frame_idx
                            ),
                        )
                        bbox = ref_result.bbox
                        match_source = ref_result.source
                    else:
                        bbox = self.reference_matcher.match(ref_url, ref_path, image)
                        match_source = 'legacy'
                    if bbox is not None:
                        prediction.add_reference_prediction(
                            ReferencePrediction(ref_url, prediction.frame_url, *bbox)
                        )
                        self.logger.debug(
                            "Referans eşleşti: %s source=%s bbox=%s",
                            ref_url,
                            match_source,
                            bbox,
                        )
                except Exception as e:
                    self.logger.warning(f"ReferenceMatcher hatası ({ref}): {e}")
                    continue

        return prediction


class _DummyDetector:
    """Dry-run modu için dummy detector - gerçek ağırlık yokken kullanılır."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect(self, image):
        self.logger.info("DummyDetector: rastgele tespit üretiliyor (dry-run).")
        import numpy as np
        h, w = image.shape[:2]
        detections = []
        # Rastgele 0-3 arası nesne üret
        import random
        for _ in range(random.randint(0, 3)):
            x1 = random.uniform(50, w // 2)
            y1 = random.uniform(50, h // 2)
            x2 = x1 + random.uniform(50, 200)
            y2 = y1 + random.uniform(50, 200)
            cls_id = random.choice([0, 1, 2, 3])
            detections.append({
                'cls': cls_id,
                'cls_name': ['Tasit', 'Insan', 'UAP', 'UAI'][cls_id],
                'conf': random.uniform(0.5, 0.99),
                'bbox': (x1, y1, x2, y2),
            })
        return detections


class _DummyPositioning:
    """Dry-run modu için dummy positioning."""
    def __init__(self):
        self.last_pos = (0.0, 0.0, 0.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_frame(self, frame_idx, frame_path, health_status, gt_x=0.0, gt_y=0.0,
                      gt_z=0.0, image=None):
        if health_status == '1':
            self.last_pos = (gt_x, gt_y, gt_z)
        else:
            # Küçük bir rastgele sapma ekle
            import random
            self.last_pos = (
                self.last_pos[0] + random.uniform(-0.5, 0.5),
                self.last_pos[1] + random.uniform(-0.5, 0.5),
                self.last_pos[2] + random.uniform(-0.1, 0.1),
            )
        return self.last_pos


class _DummyReferenceMatcher:
    """Dry-run modu için dummy referans matcher - her zaman None döndürür."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_cache = {}

    def precompute_reference(self, ref_url, ref_path):
        pass

    def match(self, ref_url, ref_path, frame_img):
        return None  # Dry-run'da referans eşleme yapma
