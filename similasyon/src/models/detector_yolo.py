import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..constants import classes


class DetectorYOLO:
    """YOLO tabanlı nesne dedektörü - Görev 1.

    Detection akışı:
    1. Full-frame YOLO: Tasit, Insan, UAP, UAI
    2. Opsiyonel SAHI second-pass: sadece İnsan (config ile açılır)
    3. YOLO insanları + SAHI insanları IoU/NMS merge
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        model_path = config.get("model_paths", {}).get("detector", "weights/detector/best.pt")
        self._resolve_model_path(model_path)

        # Detector ayarları
        det_cfg = config.get("detector", {})
        self.imgsz = det_cfg.get("imgsz", 1280)
        self.iou_threshold = det_cfg.get("iou_threshold", 0.7)
        self.max_det = det_cfg.get("max_det", 300)
        self.conf_thresholds = det_cfg.get("conf_thresholds", {
            "Tasit": 0.25, "Insan": 0.20, "UAP": 0.20, "UAI": 0.20,
        })

        # SAHI ayarları (varsayılan kapalı)
        self.use_sahi_for_person = det_cfg.get("use_sahi_for_person", False)
        self.sahi_slice_height = det_cfg.get("sahi_slice_height", 640)
        self.sahi_slice_width = det_cfg.get("sahi_slice_width", 640)
        self.sahi_overlap_height_ratio = det_cfg.get("sahi_overlap_height_ratio", 0.25)
        self.sahi_overlap_width_ratio = det_cfg.get("sahi_overlap_width_ratio", 0.25)
        self.sahi_person_conf = det_cfg.get("sahi_person_conf", 0.15)
        self.sahi_person_iou_merge = det_cfg.get("sahi_person_iou_merge", 0.5)
        self.sahi_max_extra_persons = det_cfg.get("sahi_max_extra_persons", 100)

        # SAHI availability (lazy check)
        self._sahi_available = None  # None = unchecked, cached after first check

        # Class ID mapping
        self.class_to_id = {v: k for k, v in classes.items()}
        self.name_to_id = classes

        self.logger.info(f"DetectorYOLO başlatıldı. Model: {self.model_path}")
        self.logger.info(f"Conf thresholds: {self.conf_thresholds}")
        if self.use_sahi_for_person:
            self.logger.info(f"SAHI person second-pass AKTİF: slice={self.sahi_slice_height}x{self.sahi_slice_width}, "
                           f"overlap={self.sahi_overlap_height_ratio}x{self.sahi_overlap_width_ratio}, "
                           f"person_conf={self.sahi_person_conf}")
        else:
            self.logger.info("SAHI person second-pass: kapalı (config ile açılabilir)")

    def _resolve_model_path(self, rel_path: str):
        """Model yolunu çözümle."""
        candidates = [
            Path(rel_path),
            Path(".") / rel_path,
            Path("..") / rel_path,
            Path.home() / "NPC-AI" / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                self.model_path = str(cand.resolve())
                self.logger.info(f"Model bulundu: {self.model_path}")
                break
        else:
            self.model_path = rel_path
            self.logger.warning(f"Model {rel_path} bulunamadı, olduğu gibi kullanılıyor.")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model ağırlığı bulunamadı: {self.model_path}\n"
                f"Lütfen best.pt dosyasını similasyon/weights/detector/ altına kopyalayın."
            )
        self.model = YOLO(self.model_path)

    def _check_sahi(self) -> bool:
        """SAHI kullanılabilir mi? (lazy check, cached)."""
        if self._sahi_available is not None:
            return self._sahi_available
        try:
            from sahi.predict import get_sliced_prediction
            from sahi import AutoDetectionModel
            self._sahi_available = True
            self.logger.info("SAHI kütüphanesi mevcut, sliced prediction kullanılabilir.")
        except ImportError:
            self._sahi_available = False
            self.logger.warning("SAHI kütüphanesi bulunamadı. Sliced prediction devre dışı. "
                               "Kurmak için: pip install sahi")
        return self._sahi_available

    def _run_sahi_for_persons(self, image_bgr: np.ndarray) -> list:
        """SAHI ile sadece İnsan sınıfı için sliced inference yap.

        Returns:
            İnsan detection listesi (full image koordinat sisteminde).
        """
        from sahi.predict import get_sliced_prediction
        from sahi import AutoDetectionModel

        # SAHI modelini aynı YOLO weight ile başlat
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.model_path,
            confidence_threshold=self.sahi_person_conf,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        result = get_sliced_prediction(
            image_bgr,
            sahi_model,
            slice_height=self.sahi_slice_height,
            slice_width=self.sahi_slice_width,
            overlap_height_ratio=self.sahi_overlap_height_ratio,
            overlap_width_ratio=self.sahi_overlap_width_ratio,
        )

        persons = []
        H, W = image_bgr.shape[:2]
        for obj in result.object_prediction_list:
            cls_id = int(obj.category.id)
            # Sadece İnsan (cls=1)
            if cls_id != 1:
                continue
            conf = float(obj.score.value)
            bbox = (obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy)  # (x1, y1, x2, y2)
            x1 = max(0, min(float(bbox[0]), W - 1))
            y1 = max(0, min(float(bbox[1]), H - 1))
            x2 = max(0, min(float(bbox[2]), W - 1))
            y2 = max(0, min(float(bbox[3]), H - 1))
            if x1 >= x2 or y1 >= y2:
                continue
            persons.append({
                "cls": 1,
                "cls_name": "Insan",
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "source": "sahi",
            })

        return persons

    def _merge_persons(self, yolo_persons: list, sahi_persons: list) -> list:
        """YOLO insanları ve SAHI insanlarını IoU merge et.

        - SAHI'den gelen duplicate bbox'ları (YOLO ile yüksek IoU) kaldır.
        - Azami `sahi_max_extra_persons` limiti uygula.
        """
        if not sahi_persons:
            return yolo_persons

        merged = list(yolo_persons)
        extra_count = 0

        for sp in sahi_persons:
            sp_box = sp["bbox"]
            # YOLO insanlarıyla IoU kontrolü
            duplicate = False
            for yp in yolo_persons:
                yp_box = yp["bbox"]
                iou = self._box_iou(sp_box, yp_box)
                if iou > self.sahi_person_iou_merge:
                    duplicate = True
                    break
            if not duplicate:
                if extra_count < self.sahi_max_extra_persons:
                    merged.append(sp)
                    extra_count += 1
                else:
                    break

        if extra_count > 0:
            self.logger.debug(f"SAHI merge: {extra_count} yeni insan eklendi (toplam={len(merged)})")
        return merged

    @staticmethod
    def _box_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x1 >= x2 or y1 >= y2:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0

    def detect(self, image: np.ndarray) -> list:
        """Tek bir frame üzerinde nesne tespiti yapar.

        Args:
            image: BGR formatında numpy array (H, W, 3)

        Returns:
            Her biri dict olan detection listesi:
            [{
                'cls': int (0-3 arası class id),
                'cls_name': str,
                'conf': float,
                'bbox': (x1, y1, x2, y2) - pixel coordinates,
            }, ...]
        """
        H, W = image.shape[:2]

        # --- Adım 1: Full-frame YOLO ---
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=0.01,  # Per-class threshold sonradan uygulanacak
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )

        yolo_detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(boxes, confs, clss):
                cls_id = int(cls_id)
                cls_name = self.class_to_id.get(cls_id, "Bilinmiyor")

                min_conf = self.conf_thresholds.get(cls_name, 0.25)
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = box
                x1 = max(0, min(float(x1), W - 1))
                y1 = max(0, min(float(y1), H - 1))
                x2 = max(0, min(float(x2), W - 1))
                y2 = max(0, min(float(y2), H - 1))

                if x1 >= x2 or y1 >= y2:
                    continue

                yolo_detections.append({
                    'cls': cls_id,
                    'cls_name': cls_name,
                    'conf': float(conf),
                    'bbox': (x1, y1, x2, y2),
                })

        # --- Adım 2: Opsiyonel SAHI person-only second-pass ---
        if self.use_sahi_for_person:
            try:
                sahi_ok = self._check_sahi()
                if sahi_ok:
                    sahi_persons = self._run_sahi_for_persons(image)
                    # YOLO insanlarını ayır
                    yolo_persons = [d for d in yolo_detections if d['cls'] == 1]
                    yolo_others = [d for d in yolo_detections if d['cls'] != 1]
                    # Merge
                    merged_persons = self._merge_persons(yolo_persons, sahi_persons)
                    final_detections = yolo_others + merged_persons
                    self.logger.debug(
                        f"SAHI: YOLO insan={len(yolo_persons)}, "
                        f"SAHI insan={len(sahi_persons)}, "
                        f"toplam insan={len(merged_persons)}"
                    )
                    return final_detections
            except Exception as e:
                self.logger.warning(f"SAHI person detection hatası: {e}, full-frame YOLO ile devam.")

        return yolo_detections
