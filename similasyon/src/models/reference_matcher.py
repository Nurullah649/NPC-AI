"""
Görev 3: Referans Görüntü Eşleme Modülü.

Primary: LightGlue + ALIKED/SuperPoint
Fallback: DINOv2 embedding + ROI search
Classical fallback: ORB + template matching
"""
import logging
import os
from pathlib import Path

import cv2
import numpy as np


class ReferenceMatcher:
    """Referans görüntüleri frame içinde bulur.

    - LightGlue + ALIKED/SuperPoint feature matching
    - DINOv2 embedding fallback (opsiyonel)
    - ORB + template matching (son çare)
    - Referans feature cache
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        ref_cfg = config.get("reference", {})
        self.matcher_type = ref_cfg.get("matcher", "lightglue")
        self.extractor_type = ref_cfg.get("extractor", "aliked")
        self.min_matches = ref_cfg.get("min_matches", 10)
        self.min_inliers = ref_cfg.get("min_inliers", 8)
        self.min_inlier_ratio = ref_cfg.get("min_inlier_ratio", 0.20)
        self.max_num_keypoints = int(ref_cfg.get("max_num_keypoints", 2048))
        self.input_mode = str(ref_cfg.get("input_mode", "grayscale")).strip().lower()
        self.ransac_threshold = float(ref_cfg.get("ransac_threshold", 4.0))
        self.max_reprojection_error = float(
            ref_cfg.get("max_reprojection_error", 4.0)
        )
        self.min_reference_hull_coverage = float(
            ref_cfg.get("min_reference_hull_coverage", 0.01)
        )
        self.min_frame_hull_coverage = float(
            ref_cfg.get("min_frame_hull_coverage", 0.001)
        )
        self.min_projected_area_ratio = float(
            ref_cfg.get("min_projected_area_ratio", 0.0001)
        )
        self.max_projected_area_ratio = float(
            ref_cfg.get("max_projected_area_ratio", 0.65)
        )
        self.min_projected_visible_ratio = float(
            ref_cfg.get("min_projected_visible_ratio", 0.35)
        )
        self.max_corner_margin_ratio = float(
            ref_cfg.get("max_corner_margin_ratio", 0.75)
        )
        self.max_bbox_aspect_ratio = float(
            ref_cfg.get("max_bbox_aspect_ratio", 12.0)
        )
        self.use_dinov2 = ref_cfg.get("use_dinov2_fallback", False)
        self.device_preference = str(ref_cfg.get("device", "auto")).strip().lower()
        self.last_match_diagnostics = {}
        self.last_match_attempts = []

        # Feature cache: {ref_url: {'features': ..., 'image': np.ndarray, 'path': str}}
        self.feature_cache = {}

        # LightGlue matcher
        self._matcher = None
        self._extractor = None
        self._device = None
        self._init_matcher()

        self.logger.info(
            f"ReferenceMatcher başlatıldı. "
            f"matcher={self.matcher_type}, extractor={self.extractor_type}"
        )

    def _init_matcher(self):
        """LightGlue ve feature extractor'ı başlat."""
        try:
            # LightGlue'ü dene
            import torch
            from lightglue import LightGlue, ALIKED, SuperPoint, DISK

            if self.device_preference not in {"auto", "cpu", "cuda"}:
                raise ValueError(
                    "reference.device 'auto', 'cpu' veya 'cuda' olmalı; "
                    f"gelen={self.device_preference!r}"
                )
            if self.device_preference == "cuda" and not torch.cuda.is_available():
                self.logger.warning(
                    "ReferenceMatcher için CUDA istendi ancak kullanılamıyor; CPU kullanılacak."
                )
            use_cuda = (
                self.device_preference != "cpu" and torch.cuda.is_available()
            )
            self._device = torch.device("cuda" if use_cuda else "cpu")

            if self.extractor_type == "aliked":
                extractor = ALIKED(max_num_keypoints=self.max_num_keypoints)
            elif self.extractor_type == "superpoint":
                extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints)
            elif self.extractor_type == "disk":
                extractor = DISK(max_num_keypoints=self.max_num_keypoints)
            else:
                extractor = ALIKED(max_num_keypoints=self.max_num_keypoints)

            # Extractor, matcher ve giriş tensörleri mutlaka aynı cihazda olmalı.
            self._extractor = extractor.eval().to(self._device)
            self._matcher = (
                LightGlue(features=self.extractor_type).eval().to(self._device)
            )
            self._lightglue_available = True
            self.logger.info(
                "LightGlue + %s başarıyla yüklendi (device=%s).",
                self.extractor_type,
                self._device,
            )

        except ImportError:
            self._lightglue_available = False
            self.logger.warning(
                "LightGlue modülü bulunamadı. "
                "ORB/template matching fallback kullanılacak. "
                "Kurmak için: pip install lightglue"
            )
        except Exception as e:
            self._lightglue_available = False
            self.logger.warning(f"LightGlue yüklenemedi: {e}")

        # DINOv2 fallback
        if self.use_dinov2:
            try:
                import torch
                self._dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                dinov2_device = self._device or torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._dinov2.eval().to(dinov2_device)
                self._dinov2_available = True
                self.logger.info("DINOv2 fallback hazır.")
            except Exception as e:
                self._dinov2_available = False
                self.logger.warning(f"DINOv2 yüklenemedi: {e}")
        else:
            self._dinov2_available = False

    def _load_image(self, path: str) -> np.ndarray | None:
        """Görüntüyü yükle, başarısız olursa None."""
        if path is None or not os.path.exists(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        return img

    def _preprocess_for_matching(self, img: np.ndarray) -> np.ndarray:
        """LightGlue girişini seçilen moda göre hazırla.

        ALIKED/LightGlue doğal görüntülerle eğitildiği için üretim varsayılanı
        yalnız gri-seviyedir. Eski CLAHE+Sobel yolu, simetrik yapılarda sahte
        kenar eşleşmelerini güçlendirdiğinden sadece açıkça ``edge`` seçilirse
        kullanılır.
        """
        input_mode = getattr(self, "input_mode", "grayscale")
        if input_mode == "raw":
            return img.copy()

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if input_mode in {"grayscale", "gray", "grey"}:
            return gray
        if input_mode != "edge":
            self.logger.warning(
                "Bilinmeyen reference.input_mode=%r; grayscale kullanılacak.",
                input_mode,
            )
            return gray

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        combined = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)
        return combined

    def _to_lightglue_tensor(self, img: np.ndarray):
        """OpenCV görüntüsünü LightGlue ile aynı cihazdaki RGB tensöre çevir."""
        import torch

        processed = self._preprocess_for_matching(img)
        if len(processed.shape) == 2:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        device = self._device or torch.device("cpu")
        return (
            torch.from_numpy(processed_rgb)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device)
            / 255.0
        )

    @staticmethod
    def _remove_batch(value):
        """LightGlue'ün tek-elemanlı batch/list çıktısını örnek düzeyine indir."""
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        if hasattr(value, "ndim") and value.ndim >= 3 and value.shape[0] == 1:
            return value[0]
        return value

    def precompute_reference(self, ref_url: str, ref_path: str):
        """Referans görüntü feature'larını çıkar ve cache'e ekle.

        Args:
            ref_url: Referansın sunucudaki URL'si.
            ref_path: İndirilmiş referans görüntünün yerel yolu.
        """
        if ref_url in self.feature_cache:
            return  # Zaten cache'lenmiş

        img = self._load_image(ref_path)
        if img is None:
            self.logger.error(f"Referans görüntü yüklenemedi: {ref_path}")
            return

        features = {}
        if self._lightglue_available:
            try:
                import torch

                with torch.inference_mode():
                    feats = self._extractor.extract(self._to_lightglue_tensor(img))
                features['lightglue'] = feats
            except Exception as e:
                self.logger.warning(f"LightGlue feature çıkarma hatası (ref): {e}")

        # ORB features (fallback)
        processed = self._preprocess_for_matching(img)
        orb = cv2.ORB.create(nfeatures=2000)
        kp, des = orb.detectAndCompute(processed, None)
        features['orb'] = (kp, des)

        # ORB template (bütün görüntü)
        features['template'] = processed

        features['shape'] = img.shape[:2]

        self.feature_cache[ref_url] = {
            'features': features,
            'path': ref_path,
            'shape': img.shape[:2],
        }
        self.logger.debug(f"Referans cache'e eklendi: {ref_url} ({len(self.feature_cache)} cached)")

    @staticmethod
    def _hull_coverage(points: np.ndarray, width: int, height: int) -> float:
        if len(points) < 3:
            return 0.0
        hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
        return float(cv2.contourArea(hull) / max(1.0, float(width * height)))

    @staticmethod
    def _projected_visible_ratio(points: np.ndarray, width: int, height: int) -> float:
        polygon = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
        polygon_area = float(cv2.contourArea(polygon))
        if polygon_area <= 1.0:
            return 0.0
        frame_polygon = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        try:
            intersection, _ = cv2.intersectConvexConvex(polygon, frame_polygon)
        except cv2.error:
            return 0.0
        return float(intersection / polygon_area)

    def _validate_homography_projection(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        homography: np.ndarray,
        inlier_mask: np.ndarray,
        ref_shape: tuple[int, int],
        frame_shape: tuple[int, int],
        method: str,
    ) -> tuple[tuple[float, float, float, float] | None, dict]:
        """Homografiyi kırpmadan önce geometrik olarak doğrula.

        Yalnız inlier sayısı yeterli değildir: destek noktalarının dışına taşan
        neredeyse-tekil bir homografi, referans köşesini on binlerce piksele
        fırlatıp kırpıldıktan sonra makul görünen dev bir kutu üretebilir.
        """
        source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
        destination = np.asarray(destination, dtype=np.float32).reshape(-1, 2)
        mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        match_count = int(len(source))
        inlier_count = int(mask.sum())
        inlier_ratio = inlier_count / max(1, match_count)
        diagnostics = {
            "method": method,
            "matches": match_count,
            "inliers": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            "accepted": False,
            "reason": "not_evaluated",
        }

        def reject(reason: str):
            diagnostics["reason"] = reason
            return None, diagnostics

        if match_count < int(getattr(self, "min_matches", 10)):
            return reject("too_few_matches")
        if inlier_count < int(getattr(self, "min_inliers", 8)):
            return reject("too_few_inliers")
        if inlier_ratio < float(getattr(self, "min_inlier_ratio", 0.20)):
            return reject("low_inlier_ratio")
        if homography is None or not np.isfinite(homography).all():
            return reject("invalid_homography")

        projected_inliers = cv2.perspectiveTransform(
            source[mask].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected_inliers - destination[mask], axis=1)
        reprojection_error = float(np.median(errors)) if len(errors) else float("inf")
        diagnostics["reprojection_error"] = reprojection_error
        if reprojection_error > float(getattr(self, "max_reprojection_error", 4.0)):
            return reject("high_reprojection_error")

        ref_height, ref_width = ref_shape
        frame_height, frame_width = frame_shape
        reference_hull_coverage = self._hull_coverage(
            source[mask], ref_width, ref_height
        )
        frame_hull_coverage = self._hull_coverage(
            destination[mask], frame_width, frame_height
        )
        diagnostics["reference_hull_coverage"] = reference_hull_coverage
        diagnostics["frame_hull_coverage"] = frame_hull_coverage
        if reference_hull_coverage < float(
            getattr(self, "min_reference_hull_coverage", 0.01)
        ):
            return reject("low_reference_hull_coverage")
        if frame_hull_coverage < float(
            getattr(self, "min_frame_hull_coverage", 0.001)
        ):
            return reject("low_frame_hull_coverage")

        corners = np.array(
            [
                [0, 0],
                [ref_width - 1, 0],
                [ref_width - 1, ref_height - 1],
                [0, ref_height - 1],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        diagnostics["projected_corners"] = projected.astype(float).tolist()
        if not np.isfinite(projected).all():
            return reject("non_finite_projection")
        if not cv2.isContourConvex(projected.astype(np.float32).reshape(-1, 1, 2)):
            return reject("non_convex_projection")

        projected_area_ratio = float(
            abs(cv2.contourArea(projected.astype(np.float32)))
            / max(1.0, float(frame_width * frame_height))
        )
        diagnostics["projected_area_ratio"] = projected_area_ratio
        if projected_area_ratio < float(
            getattr(self, "min_projected_area_ratio", 0.0001)
        ):
            return reject("projection_too_small")
        if projected_area_ratio > float(
            getattr(self, "max_projected_area_ratio", 0.65)
        ):
            return reject("projection_too_large")

        margin_ratio = float(getattr(self, "max_corner_margin_ratio", 0.75))
        if (
            projected[:, 0].min() < -frame_width * margin_ratio
            or projected[:, 0].max() > frame_width * (1.0 + margin_ratio)
            or projected[:, 1].min() < -frame_height * margin_ratio
            or projected[:, 1].max() > frame_height * (1.0 + margin_ratio)
        ):
            return reject("projection_outside_margin")

        visible_ratio = self._projected_visible_ratio(
            projected, frame_width, frame_height
        )
        diagnostics["projected_visible_ratio"] = visible_ratio
        if visible_ratio < float(
            getattr(self, "min_projected_visible_ratio", 0.35)
        ):
            return reject("low_projected_visible_ratio")

        x1 = float(np.clip(projected[:, 0].min(), 0, frame_width - 1))
        y1 = float(np.clip(projected[:, 1].min(), 0, frame_height - 1))
        x2 = float(np.clip(projected[:, 0].max(), 0, frame_width - 1))
        y2 = float(np.clip(projected[:, 1].max(), 0, frame_height - 1))
        if x1 >= x2 or y1 >= y2:
            return reject("empty_projection")
        aspect_ratio = max(
            (x2 - x1) / max(1e-6, y2 - y1),
            (y2 - y1) / max(1e-6, x2 - x1),
        )
        diagnostics["bbox_aspect_ratio"] = float(aspect_ratio)
        if aspect_ratio > float(getattr(self, "max_bbox_aspect_ratio", 12.0)):
            return reject("implausible_bbox_aspect")

        bbox = (x1, y1, x2, y2)
        diagnostics["bbox"] = list(bbox)
        diagnostics["accepted"] = True
        diagnostics["reason"] = "accepted"
        return bbox, diagnostics

    def _match_lightglue(self, ref_feats: dict, frame_img: np.ndarray) -> tuple | None:
        """LightGlue ile referans eşleme yap.

        Returns:
            (x1, y1, x2, y2) bbox veya None
        """
        if not self._lightglue_available or 'lightglue' not in ref_feats:
            self.last_match_diagnostics = {
                "method": "lightglue",
                "accepted": False,
                "reason": "lightglue_unavailable",
            }
            return None

        try:
            import torch

            with torch.inference_mode():
                frame_feats = self._extractor.extract(
                    self._to_lightglue_tensor(frame_img)
                )
                match_output = self._matcher({
                    'image0': ref_feats['lightglue'],
                    'image1': frame_feats,
                })

            if not isinstance(match_output, dict):
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "accepted": False,
                    "reason": "invalid_match_output",
                }
                return None

            # Kurulu LightGlue sürümü `matches` değerini batch başına bir
            # [K, 2] indeks tensörü içeren liste olarak döndürüyor. Eski kod
            # listenin uzunluğunu eşleşme sayısı sanıyor ve her zaman 1 görüyordu.
            pairs = self._remove_batch(match_output.get('matches'))
            if pairs is None or not torch.is_tensor(pairs):
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "accepted": False,
                    "reason": "invalid_match_pairs",
                }
                return None
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError(f"Beklenmeyen LightGlue matches şekli: {pairs.shape}")
            if pairs.shape[0] < self.min_matches:
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "matches": int(pairs.shape[0]),
                    "accepted": False,
                    "reason": "too_few_matches",
                }
                return None

            ref_keypoints = self._remove_batch(
                ref_feats['lightglue'].get('keypoints')
            )
            frame_keypoints = self._remove_batch(frame_feats.get('keypoints'))
            if ref_keypoints is None or frame_keypoints is None:
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "accepted": False,
                    "reason": "missing_keypoints",
                }
                return None

            mkpts0 = ref_keypoints[pairs[:, 0]]
            mkpts1 = frame_keypoints[pairs[:, 1]]

            if len(mkpts0) < self.min_inliers:
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "matches": int(len(mkpts0)),
                    "accepted": False,
                    "reason": "too_few_inliers",
                }
                return None

            # Homography ile bbox tahmini
            src_pts = mkpts0.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
            dst_pts = mkpts1.detach().cpu().numpy().reshape(-1, 1, 2).astype(np.float32)

            H_mat, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                float(getattr(self, "ransac_threshold", 4.0)),
            )
            if H_mat is None or mask is None:
                self.last_match_diagnostics = {
                    "method": "lightglue",
                    "matches": int(len(mkpts0)),
                    "accepted": False,
                    "reason": "homography_failed",
                }
                return None

            bbox, diagnostics = self._validate_homography_projection(
                src_pts,
                dst_pts,
                H_mat,
                mask,
                ref_feats.get(
                    'shape', (frame_img.shape[0], frame_img.shape[1])
                ),
                frame_img.shape[:2],
                "lightglue",
            )
            self.last_match_diagnostics = diagnostics
            if bbox is None:
                self.logger.debug(
                    "LightGlue homography reddedildi: reason=%s inliers=%s/%s ratio=%.3f",
                    diagnostics.get("reason"),
                    diagnostics.get("inliers", 0),
                    diagnostics.get("matches", 0),
                    diagnostics.get("inlier_ratio", 0.0),
                )
                return None

            self.logger.debug(
                "LightGlue match: %d/%d inliers, ratio=%.3f, bbox=(%.0f,%.0f,%.0f,%.0f)",
                diagnostics["inliers"],
                diagnostics["matches"],
                diagnostics["inlier_ratio"],
                *bbox,
            )
            return bbox

        except Exception as e:
            self.last_match_diagnostics = {
                "method": "lightglue",
                "accepted": False,
                "reason": f"exception:{type(e).__name__}",
            }
            self.logger.warning(f"LightGlue match hatası: {e}")
            return None

    def _match_orb(self, ref_features: dict, frame_img: np.ndarray) -> tuple | None:
        """ORB + homography ile referans eşleme.

        Returns:
            (x1, y1, x2, y2) bbox veya None
        """
        if 'orb' not in ref_features:
            self.last_match_diagnostics = {
                "method": "orb", "accepted": False, "reason": "orb_unavailable"
            }
            return None

        ref_kp, ref_des = ref_features['orb']
        if ref_des is None or len(ref_kp) < 4:
            self.last_match_diagnostics = {
                "method": "orb", "accepted": False,
                "reason": "insufficient_reference_features",
            }
            return None

        processed = self._preprocess_for_matching(frame_img)
        orb = cv2.ORB.create(nfeatures=2000)
        frame_kp, frame_des = orb.detectAndCompute(processed, None)

        if frame_des is None or len(frame_kp) < 4:
            self.last_match_diagnostics = {
                "method": "orb", "accepted": False,
                "reason": "insufficient_frame_features",
            }
            return None

        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ref_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < self.min_matches:
            self.last_match_diagnostics = {
                "method": "orb", "matches": len(matches), "accepted": False,
                "reason": "too_few_matches",
            }
            return None

        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H_mat, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            float(getattr(self, "ransac_threshold", 4.0)),
        )
        if H_mat is None or mask is None:
            self.last_match_diagnostics = {
                "method": "orb", "matches": len(matches), "accepted": False,
                "reason": "homography_failed",
            }
            return None

        bbox, diagnostics = self._validate_homography_projection(
            src_pts,
            dst_pts,
            H_mat,
            mask,
            ref_features.get(
                'shape', (frame_img.shape[0], frame_img.shape[1])
            ),
            frame_img.shape[:2],
            "orb",
        )
        self.last_match_diagnostics = diagnostics
        return bbox

    def match(self, ref_url: str, ref_path: str, frame_img: np.ndarray) -> tuple | None:
        """Referans görüntüyü frame içinde bul.

        Args:
            ref_url: Referans URL'si (cache anahtarı).
            ref_path: Referans görüntü dosya yolu.
            frame_img: Frame görüntüsü (BGR numpy array).

        Returns:
            (x1, y1, x2, y2) bbox veya None (eşleşme yok).
        """
        # Cache'te yoksa önceden hesapla
        if ref_url not in self.feature_cache:
            self.precompute_reference(ref_url, ref_path)

        cache_entry = self.feature_cache.get(ref_url)
        if cache_entry is None:
            return None

        features = cache_entry['features']
        self.last_match_attempts = []

        # 1. LightGlue
        bbox = self._match_lightglue(features, frame_img)
        self.last_match_attempts.append(dict(self.last_match_diagnostics or {}))
        if bbox is not None:
            return bbox

        # 2. ORB fallback
        bbox = self._match_orb(features, frame_img)
        self.last_match_attempts.append(dict(self.last_match_diagnostics or {}))
        if bbox is not None:
            return bbox

        return None  # Eşleşme yok
