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
        self.min_inlier_ratio = ref_cfg.get("min_inlier_ratio", 0.25)
        self.use_dinov2 = ref_cfg.get("use_dinov2_fallback", False)

        # Feature cache: {ref_url: {'features': ..., 'image': np.ndarray, 'path': str}}
        self.feature_cache = {}

        # LightGlue matcher
        self._matcher = None
        self._extractor = None
        self._init_matcher()

        self.logger.info(
            f"ReferenceMatcher başlatıldı. "
            f"matcher={self.matcher_type}, extractor={self.extractor_type}"
        )

    def _init_matcher(self):
        """LightGlue ve feature extractor'ı başlat."""
        try:
            # LightGlue'ü dene
            import lightglue
            from lightglue import LightGlue, ALIKED, SuperPoint, DISK

            if self.extractor_type == "aliked":
                self._extractor = ALIKED(max_num_keypoints=1024).eval()
            elif self.extractor_type == "superpoint":
                self._extractor = SuperPoint(max_num_keypoints=1024).eval()
            elif self.extractor_type == "disk":
                self._extractor = DISK(max_num_keypoints=1024).eval()
            else:
                self._extractor = ALIKED(max_num_keypoints=1024).eval()

            self._matcher = LightGlue(features=self.extractor_type).eval()
            self._lightglue_available = True
            self.logger.info(f"LightGlue + {self.extractor_type} başarıyla yüklendi.")

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
                self._dinov2.eval()
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
        """Termal/RGB farkı için ön işleme.

        1. Grayscale
        2. CLAHE
        3. Edge enhancement (Sobel)
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge enhancement
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Combine
        combined = cv2.addWeighted(enhanced, 0.7, edges, 0.3, 0)
        return combined

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
                from lightglue import viz2d

                processed = self._preprocess_for_matching(img)
                # LightGlue RGB bekler, 3 kanala çevir
                if len(processed.shape) == 2:
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                else:
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                feats = self._extractor.extract(
                    torch.from_numpy(processed_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                )
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

    def _match_lightglue(self, ref_feats: dict, frame_img: np.ndarray) -> tuple | None:
        """LightGlue ile referans eşleme yap.

        Returns:
            (x1, y1, x2, y2) bbox veya None
        """
        if not self._lightglue_available or 'lightglue' not in ref_feats:
            return None

        try:
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            processed = self._preprocess_for_matching(frame_img)
            if len(processed.shape) == 2:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

            frame_tensor = torch.from_numpy(processed_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            # Frame feature'larını çıkar
            frame_feats = self._extractor.extract(frame_tensor)

            # LightGlue ile eşle
            matches = self._matcher({'image0': ref_feats['lightglue'], 'image1': frame_feats})

            if matches is None or len(matches['matches']) < self.min_matches:
                return None

            matches_data = matches['matches']
            mkpts0 = matches_data['keypoints0']
            mkpts1 = matches_data['keypoints1']

            if len(mkpts0) < self.min_inliers:
                return None

            # Homography ile bbox tahmini
            src_pts = mkpts0.cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
            dst_pts = mkpts1.cpu().numpy().reshape(-1, 1, 2).astype(np.float32)

            H_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H_mat is None:
                return None

            inlier_count = np.sum(mask) if mask is not None else len(mkpts0)
            inlier_ratio = inlier_count / len(mkpts0) if len(mkpts0) > 0 else 0

            if inlier_ratio < self.min_inlier_ratio:
                self.logger.debug(f"Inlier ratio düşük: {inlier_ratio:.3f}")
                return None

            # Referans görüntünün dört köşesini frame'e dönüştür
            ref_h, ref_w = ref_feats.get('shape', (frame_img.shape[0], frame_img.shape[1]))
            corners = np.array([
                [0, 0],
                [ref_w - 1, 0],
                [ref_w - 1, ref_h - 1],
                [0, ref_h - 1],
            ], dtype=np.float32).reshape(-1, 1, 2)

            transformed = cv2.perspectiveTransform(corners, H_mat)
            transformed = transformed.squeeze()

            # Bbox hesapla
            x1 = np.min(transformed[:, 0])
            y1 = np.min(transformed[:, 1])
            x2 = np.max(transformed[:, 0])
            y2 = np.max(transformed[:, 1])

            # Frame sınırlarına clip et
            H, W = frame_img.shape[:2]
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))

            if x1 >= x2 or y1 >= y2:
                return None

            self.logger.debug(
                f"LightGlue match: {inlier_count}/{len(mkpts0)} inliers, "
                f"ratio={inlier_ratio:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )
            return (x1, y1, x2, y2)

        except Exception as e:
            self.logger.warning(f"LightGlue match hatası: {e}")
            return None

    def _match_orb(self, ref_features: dict, frame_img: np.ndarray) -> tuple | None:
        """ORB + homography ile referans eşleme.

        Returns:
            (x1, y1, x2, y2) bbox veya None
        """
        if 'orb' not in ref_features:
            return None

        ref_kp, ref_des = ref_features['orb']
        if ref_des is None or len(ref_kp) < 4:
            return None

        processed = self._preprocess_for_matching(frame_img)
        orb = cv2.ORB.create(nfeatures=2000)
        frame_kp, frame_des = orb.detectAndCompute(processed, None)

        if frame_des is None or len(frame_kp) < 4:
            return None

        # BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(ref_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < self.min_matches:
            return None

        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_mat is None:
            return None

        inlier_count = np.sum(mask) if mask is not None else len(matches)
        inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0

        if inlier_ratio < 0.2:
            return None

        # Referans köşelerini dönüştür
        ref_h, ref_w = ref_features.get('shape', (frame_img.shape[0], frame_img.shape[1]))
        corners = np.array([
            [0, 0], [ref_w - 1, 0],
            [ref_w - 1, ref_h - 1], [0, ref_h - 1],
        ], dtype=np.float32).reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(corners, H_mat).squeeze()

        H, W = frame_img.shape[:2]
        x1 = max(0, min(np.min(transformed[:, 0]), W - 1))
        y1 = max(0, min(np.min(transformed[:, 1]), H - 1))
        x2 = max(0, min(np.max(transformed[:, 0]), W - 1))
        y2 = max(0, min(np.max(transformed[:, 1]), H - 1))

        if x1 >= x2 or y1 >= y2:
            return None

        return (x1, y1, x2, y2)

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

        # 1. LightGlue
        bbox = self._match_lightglue(features, frame_img)
        if bbox is not None:
            return bbox

        # 2. ORB fallback
        bbox = self._match_orb(features, frame_img)
        if bbox is not None:
            return bbox

        return None  # Eşleşme yok
