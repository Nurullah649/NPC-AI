import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .direction_guard import DirectionGuard


class PositioningDPVO:
    """DPVO tabanlı pozisyon kestirimi - Görev 2.

    - health_status == '1' iken GT x/y/z değerlerini aynen gönderir ve
      çevrimiçi hizalama örneklerini biriktirir.
    - health_status == '0' iken DPVO camera-to-world pozunu seçili
      linear/Sim3/delta hizalamasıyla NED koordinatlarına dönüştürür.
    - DPVO fail ederse sınırlı ve sönümlü velocity fallback kullanır.
    """

    # The base live positioner intentionally cannot consume a global-BA gauge
    # event. Experimental subclasses must opt in explicitly before DPVO loop
    # closure is allowed to initialize.
    supports_loop_gauge_repair = False

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        dpvo_cfg = config.get("dpvo", {})
        self.min_calib_frames = dpvo_cfg.get("min_calib_frames", 50)
        self.fit_method = dpvo_cfg.get("fit_method", "sim3")
        self.anchor_at_calib_end = bool(dpvo_cfg.get("anchor_at_calib_end", True))
        self.fallback_velocity = dpvo_cfg.get("fallback_velocity", True)
        self.fallback_velocity_decay = min(
            1.0, max(0.0, float(dpvo_cfg.get("fallback_velocity_decay", 0.98)))
        )
        self.max_fallback_frames = max(0, int(dpvo_cfg.get("max_fallback_frames", 30)))
        self.input_width = dpvo_cfg.get("input_width", 960)
        self.input_height = dpvo_cfg.get("input_height", 540)
        self.calibration_width = dpvo_cfg.get("calibration_width", 1920)
        self.calibration_height = dpvo_cfg.get("calibration_height", 1080)
        self.random_seed = dpvo_cfg.get("random_seed", 0)
        self.cuda_empty_cache_every = dpvo_cfg.get("cuda_empty_cache_every", 25)
        self.buffer_size = dpvo_cfg.get("buffer_size", 4096)
        self.runtime_log_every = max(0, int(dpvo_cfg.get("runtime_log_every", 25)))
        self.delta_window = max(1, int(dpvo_cfg.get("delta_window", 9)))
        self.delta_ridge_alpha = max(0.0, float(dpvo_cfg.get("delta_ridge_alpha", 0.0)))
        self.delta_fit_intercept = bool(dpvo_cfg.get("delta_fit_intercept", False))
        self.absolute_ridge_alpha = max(0.0, float(dpvo_cfg.get("absolute_ridge_alpha", 0.1)))
        self.linear_condition_max = max(1.0, float(dpvo_cfg.get("linear_condition_max", 1e6)))

        # Model path'leri
        model_paths = config.get("model_paths", {})
        self.dpvo_weight = self._resolve_path(model_paths.get("dpvo", "weights/dpvo/dpvo.pth"))
        self.dpvo_cfg_path = self._resolve_path(model_paths.get("dpvo_cfg", "config/dpvo/npc.yaml"))
        self.calib_path = self._resolve_path(model_paths.get("camera_calib", "config/camera/calib.txt"))

        # Camera profile is the canonical source for new sessions. The legacy
        # numeric K path remains for consumers that cannot read a profile.
        camera_cfg = config.get("camera", {})
        self.camera_profile_path = None
        self.camera_profile = None
        self.camera_profile_id = None
        self.camera_distortion = None
        self.camera_undistort = False
        self._undistort_map_cache = {}
        profile_path = camera_cfg.get("profile")
        if profile_path:
            self.camera_profile_path = self._resolve_path(profile_path)
            self.camera_profile = self._load_camera_profile(self.camera_profile_path)
            self.camera_profile_id = self.camera_profile["profile_id"]
            native_size = self.camera_profile["native_size"]
            profile_width = int(native_size["width"])
            profile_height = int(native_size["height"])
            if (int(self.calibration_width), int(self.calibration_height)) != (
                profile_width,
                profile_height,
            ):
                raise ValueError(
                    "DPVO calibration size and selected camera profile disagree: "
                    f"dpvo={self.calibration_width}x{self.calibration_height}, "
                    f"profile={profile_width}x{profile_height}"
                )
            distortion_cfg = self.camera_profile["distortion"]
            self.camera_distortion = np.asarray(
                distortion_cfg["coefficients"], dtype=np.float64
            )
            profile_undistort = bool(
                self.camera_profile.get("runtime", {}).get("undistort", False)
            )
            # Keep production behaviour profile-driven, while allowing an
            # explicit experiment override without duplicating calibration.
            self.camera_undistort = bool(
                camera_cfg.get("undistort", profile_undistort)
            )

        # DPVO instance (lazy init)
        self.slam = None
        self._dpvo_available = False
        self._dpvo_init_attempted = False

        # Kalibrasyon / hizalama buffer'ları
        self.gt_buffer = []       # (x, y, z)
        self.dpvo_buffer = []     # (x, y, z)
        self.frame_indices = []   # frame idx

        # Hizalama parametreleri
        self.sim3_R = np.eye(3)
        self.sim3_t = np.zeros(3)
        self.sim3_s = 1.0
        self.alignment_anchor_offset = np.zeros(3, dtype=np.float64)
        self.is_calibrated = False
        self._calibrated_sample_count = 0
        self.delta_coef = np.eye(3, dtype=np.float64)
        self.delta_intercept = np.zeros(3, dtype=np.float64)

        # Fallback
        self.last_known_position = np.array([0.0, 0.0, 0.0])
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        self.previous_health_status = None
        self.fallback_frame_count = 0
        self.last_dpvo_raw = None
        self.raw_delta_history = []
        self.delta_raw_buffer = []
        self.delta_gt_buffer = []

        # Kamera kalibrasyonu
        self.intrinsics = self._load_calibration()

        # Yarışma protokolüne uygun, geçmiş tahmini değiştirmeyen düzlem
        # odometrisi + calibration-map relocalization füzyonu. Bağımlılık veya
        # vocabulary yoksa yardımcı sınıf kendi içinde fail-closed olur ve bu
        # sınıfın mevcut DPVO çıktısı değişmeden kalır.
        self.causal_fusion = None
        try:
            from .positioning_causal_fusion import CausalPositionFusion

            fusion_cfg = dpvo_cfg.get("causal_fusion", {})
            fusion_width = max(1, int(fusion_cfg.get("width", 640)))
            fusion_height = max(1, int(fusion_cfg.get("height", 360)))
            self.causal_fusion = CausalPositionFusion(
                config,
                self._scaled_intrinsics(fusion_width, fusion_height),
            )
        except Exception as exc:
            self.logger.error(
                "Causal position fusion başlatılamadı; baseline korunuyor: %s",
                exc,
            )

        # DirectionGuard (kalibrasyon sanity check)
        self.direction_guard = DirectionGuard(config)

        # Frame sayacı
        self.frame_counter = 0
        self._runtime_H = None
        self._runtime_W = None
        self._input_geometry_logged = False

        self.logger.info(
            f"PositioningDPVO başlatıldı. "
            f"min_calib_frames={self.min_calib_frames}, "
            f"fit_method={self.fit_method}, "
            f"input={self.input_width}x{self.input_height}, "
            f"calibration_native={self.calibration_width}x{self.calibration_height}, "
            f"camera_profile={self.camera_profile_id or 'legacy'}, "
            f"undistort={self.camera_undistort}, "
            f"delta_window={self.delta_window}, "
            f"causal_fusion={bool(self.causal_fusion and self.causal_fusion.enabled)}, "
            f"direction_guard={'aktif' if self.direction_guard.use_guard else 'pasif'}"
        )

    def _resolve_path(self, rel_path: str) -> str:
        candidates = [
            Path(rel_path),
            Path(".") / rel_path,
            Path(__file__).resolve().parent.parent.parent / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return str(Path(rel_path).resolve())

    def _load_camera_profile(self, path: str) -> dict:
        """Load and validate the versioned camera-profile contract.

        A profile prevents a camera's K, distortion values, and native size
        from being selected independently. Distortion is retained for
        experiments, but is not applied unless an explicitly validated path
        enables it.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            profile = yaml.safe_load(handle) or {}

        profile_id = profile.get("profile_id")
        native_size = profile.get("native_size")
        intrinsics_cfg = profile.get("intrinsics")
        distortion_cfg = profile.get("distortion")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"Camera profile has no valid profile_id: {path}")
        if not isinstance(native_size, dict):
            raise ValueError(f"Camera profile has no native_size mapping: {path}")
        width = int(native_size.get("width", 0))
        height = int(native_size.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera profile has invalid native_size: {path}")
        if not isinstance(intrinsics_cfg, dict) or intrinsics_cfg.get("model") != "pinhole":
            raise ValueError(f"Camera profile must define pinhole intrinsics: {path}")
        matrix = np.asarray(intrinsics_cfg.get("matrix"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Camera profile has invalid 3x3 intrinsic matrix: {path}")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"Camera profile focal lengths must be positive: {path}")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError(f"Camera profile intrinsic matrix last row must be [0, 0, 1]: {path}")
        if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
            raise ValueError(f"Camera profile principal point lies outside native image: {path}")
        if not isinstance(distortion_cfg, dict):
            raise ValueError(f"Camera profile has no distortion mapping: {path}")
        if distortion_cfg.get("model") != "opencv_radial_tangential":
            raise ValueError(f"Unsupported camera-profile distortion model: {path}")
        coefficients = np.asarray(distortion_cfg.get("coefficients"), dtype=np.float64)
        if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"Camera profile distortion must contain finite [k1,k2,p1,p2]: {path}")

        self.logger.info(
            "Kamera profili yüklendi: id=%s path=%s native=%dx%d "
            "profile_undistort_default=%s",
            profile_id,
            path,
            width,
            height,
            bool(profile.get("runtime", {}).get("undistort", False)),
        )
        return profile

    def _load_calibration(self) -> np.ndarray:
        if self.camera_profile is not None:
            intrinsics = np.asarray(
                self.camera_profile["intrinsics"]["matrix"], dtype=np.float64
            )
            self.logger.info(
                "Kamera profilinden K yüklendi: %s", self.camera_profile_id
            )
            return intrinsics.copy()
        try:
            if os.path.exists(self.calib_path):
                intrinsics = np.loadtxt(self.calib_path)
                if intrinsics.shape == (3, 3):
                    self.logger.info(f"Kalibrasyon yüklendi: {self.calib_path}")
                    return intrinsics
                flat = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
                if flat.size >= 4:
                    fx, fy, cx, cy = flat[:4]
                    self.logger.info(f"Vektör kalibrasyon yüklendi: {self.calib_path}")
                    return np.array(
                        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
            self.logger.warning(f"Kalibrasyon dosyası bulunamadı: {self.calib_path}, varsayılan kullanılıyor.")
        except Exception as e:
            self.logger.warning(f"Kalibrasyon yüklenemedi: {e}, varsayılan kullanılıyor.")
        return np.array([
            [1413.3, 0.0, 950.0639],
            [0.0, 1418.8, 543.3796],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _intrinsics_matrix_to_vec(intrinsics: np.ndarray) -> np.ndarray:
        """DPVO'nun beklediği [fx, fy, cx, cy] formatına çevir."""
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.shape == (3, 3):
            return np.array(
                [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
                dtype=np.float32,
            )
        if intrinsics.shape == (4,):
            return intrinsics.astype(np.float32)
        raise ValueError(f"Desteklenmeyen intrinsics shape: {intrinsics.shape}")

    def _scaled_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Kalibrasyonun doğal çözünürlüğünden hedef çözünürlüğe ölçekle."""
        native_w = float(self.calibration_width)
        native_h = float(self.calibration_height)
        if native_w <= 0 or native_h <= 0:
            raise ValueError(
                "calibration_width ve calibration_height pozitif olmalı; "
                f"gelen={self.calibration_width}x{self.calibration_height}"
            )
        scaled = self.intrinsics.copy().astype(np.float64)
        sx = float(width) / native_w
        sy = float(height) / native_h
        scaled[0, 0] *= sx
        scaled[0, 2] *= sx
        scaled[1, 1] *= sy
        scaled[1, 2] *= sy
        return scaled

    def _init_dpvo(self, H: int, W: int):
        if self._dpvo_available or self._dpvo_init_attempted:
            return
        self._dpvo_init_attempted = True
        try:
            from .dpvo_standalone import DPVOStandalone
            self.slam = DPVOStandalone(
                weight_path=self.dpvo_weight,
                config_path=self.dpvo_cfg_path,
                calib_path=self.calib_path,
                ht=H, wd=W,
                seed=self.random_seed,
                cuda_empty_cache_every=self.cuda_empty_cache_every,
                buffer_size=self.buffer_size,
            )
            if not self.slam.initialized:
                raise RuntimeError("DPVOStandalone initialize edilemedi.")
            loop_enabled = bool(getattr(self.slam.cfg, "LOOP_CLOSURE", False))
            classic_loop_enabled = bool(
                getattr(self.slam.cfg, "CLASSIC_LOOP_CLOSURE", False)
            )
            periodic_normalize_frequency = int(
                getattr(self.slam.cfg, "PERIODIC_NORMALIZE_FREQ", 0)
            )
            if (
                loop_enabled
                or classic_loop_enabled
                or periodic_normalize_frequency > 0
            ) and not self.supports_loop_gauge_repair:
                # A static DPVO->NED transform is invalid after any gauge
                # change (global BA or explicit periodic normalization).
                # Keep production fail-closed rather than silently emitting a
                # potentially kilometre-scale coordinate jump.
                self.slam = None
                raise RuntimeError(
                    "Gauge değiştiren DPVO modu (loop/periodic normalize) canlı "
                    "PositioningDPVO için kapalı olmalı; yalnız gauge-aware "
                    "deneysel positioner ile desteklenir."
                )
            self._dpvo_available = True
            self.logger.info("DPVO başarıyla başlatıldı.")
        except ImportError:
            self.logger.warning("DPVO modülü bulunamadı. DPVO olmadan çalışılıyor.")
            self._dpvo_available = False
        except Exception as e:
            self.logger.error(f"DPVO başlatılamadı: {e}")
            self._dpvo_available = False

    def _prepare_dpvo_input(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        """DPVO görüntüsünü hazırla ve kamera matrisini doğru temelden ölçekle.

        DPVO belleği çözünürlüğe çok hassas. 4K frame'i doğrudan vermek 6 GB
        VRAM'de OOM yaratıyor. Bu metot VO tarafını config'teki çözünürlüğe
        indirir; detector payload koordinatlarını etkilemez. Kamera matrisi
        kaynak frame boyutundan değil, kalibrasyonun doğal çözünürlüğünden
        ölçeklenir. Böylece 4K kaynak -> 1080p DPVO durumunda odak uzaklığı
        yanlışlıkla ikinci kez yarıya düşürülmez.
        """
        h, w = image.shape[:2]

        try:
            target_w = int(self.input_width) if self.input_width else 0
            target_h = int(self.input_height) if self.input_height else 0
        except (TypeError, ValueError):
            target_w = target_h = 0

        if target_w <= 0 or target_h <= 0:
            target_w, target_h = w, h

        if w == target_w and h == target_h:
            prepared = image
        else:
            prepared = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        scaled_intrinsics = self._scaled_intrinsics(target_w, target_h)
        if self.camera_undistort:
            if self.camera_distortion is None:
                raise RuntimeError(
                    "Kamera undistort istendi ancak distortion katsayilari yok."
                )
            cache_key = (
                target_w,
                target_h,
                tuple(np.asarray(scaled_intrinsics, dtype=np.float64).ravel()),
                tuple(np.asarray(self.camera_distortion, dtype=np.float64).ravel()),
            )
            maps = self._undistort_map_cache.get(cache_key)
            if maps is None:
                maps = cv2.initUndistortRectifyMap(
                    scaled_intrinsics,
                    self.camera_distortion,
                    None,
                    scaled_intrinsics,
                    (target_w, target_h),
                    cv2.CV_32FC1,
                )
                self._undistort_map_cache[cache_key] = maps
            prepared = cv2.remap(
                prepared,
                maps[0],
                maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        if not self._input_geometry_logged:
            self.logger.info(
                "DPVO giriş geometrisi: source=%dx%d target=%dx%d K=%s undistort=%s",
                w,
                h,
                target_w,
                target_h,
                self._intrinsics_matrix_to_vec(scaled_intrinsics).tolist(),
                self.camera_undistort,
            )
            self._input_geometry_logged = True
        return (
            np.ascontiguousarray(prepared),
            self._intrinsics_matrix_to_vec(scaled_intrinsics),
            target_h,
            target_w,
        )

    def _sim3_umeyama(self, src: np.ndarray, dst: np.ndarray) -> tuple:
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
            raise ValueError(f"Sim3 için Nx3 eş boyutlu veri gerekli: src={src.shape}, dst={dst.shape}")
        if src.shape[0] < 3:
            raise ValueError("Sim3 için en az 3 nokta gerekli.")
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        count = src.shape[0]
        covariance = (dst_centered.T @ src_centered) / count
        U, singular, Vt = np.linalg.svd(covariance)
        reflection_guard = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            reflection_guard[-1, -1] = -1
        R = U @ reflection_guard @ Vt
        src_var = np.sum(np.linalg.norm(src_centered, axis=1) ** 2) / src.shape[0]
        if src_var <= 1e-10:
            raise ValueError("Sim3 scale hesaplanamadı: DPVO trajectory varyansı sıfıra yakın.")
        s = float(np.trace(np.diag(singular) @ reflection_guard) / src_var)
        t = dst_mean - s * R @ src_mean
        return R, t, s

    def _align_dpvo_to_gt(self, dpvo_pos: np.ndarray) -> np.ndarray:
        if not self.is_calibrated:
            return dpvo_pos
        return (
            self.sim3_s * (self.sim3_R @ dpvo_pos)
            + self.sim3_t
            + self.alignment_anchor_offset
        )

    @staticmethod
    def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Delta dizisi Nx3 olmalı, gelen={values.shape}")
        if window <= 1:
            return values.copy()
        output = np.empty_like(values)
        cumulative = np.vstack([np.zeros((1, 3)), np.cumsum(values, axis=0)])
        for index in range(len(values)):
            start = max(0, index - window + 1)
            output[index] = (
                cumulative[index + 1] - cumulative[start]
            ) / (index - start + 1)
        return output

    def _fit_delta_mapping(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_delta = np.asarray(self.delta_raw_buffer, dtype=np.float64)
        gt_delta = np.asarray(self.delta_gt_buffer, dtype=np.float64)
        if raw_delta.shape != gt_delta.shape or len(raw_delta) < max(3, self.min_calib_frames - 1):
            raise ValueError(
                "Delta kalibrasyonu için yeterli eşleşmiş ardışık örnek yok: "
                f"raw={raw_delta.shape}, gt={gt_delta.shape}"
            )

        features = self._causal_moving_average(raw_delta, self.delta_window)
        if self.delta_fit_intercept:
            design = np.column_stack([features, np.ones(len(features))])
            regularizer = np.eye(4, dtype=np.float64) * self.delta_ridge_alpha
            regularizer[-1, -1] = 0.0
        else:
            design = features
            regularizer = np.eye(3, dtype=np.float64) * self.delta_ridge_alpha

        if self.delta_ridge_alpha > 0:
            weights = np.linalg.solve(design.T @ design + regularizer, design.T @ gt_delta)
        else:
            weights = np.linalg.lstsq(design, gt_delta, rcond=None)[0]

        if self.delta_fit_intercept:
            coef = weights[:3].T
            intercept = weights[3]
        else:
            coef = weights.T
            intercept = np.zeros(3, dtype=np.float64)
        predicted_delta = features @ coef.T + intercept
        return coef, intercept, predicted_delta, gt_delta

    def _predict_delta_position(self, raw_delta: np.ndarray) -> np.ndarray:
        if raw_delta is None:
            raise ValueError("Delta hizalama için ardışık DPVO pozu gerekli.")
        recent = np.asarray(self.raw_delta_history[-self.delta_window :], dtype=np.float64)
        smoothed = recent.mean(axis=0)
        predicted_delta = self.delta_coef @ smoothed + self.delta_intercept
        return self.last_known_position + predicted_delta

    def _advance_velocity_fallback(self) -> np.ndarray:
        if self.fallback_frame_count >= self.max_fallback_frames:
            return self.last_known_position.copy()
        decay = self.fallback_velocity_decay ** self.fallback_frame_count
        self.last_known_position = self.last_known_position + decay * self.last_velocity
        self.fallback_frame_count += 1
        return self.last_known_position.copy()

    def _update_calibration(self):
        """Birikmiş GT/DPVO çiftleriyle seçili hizalamayı güncelle.

        DirectionGuard ile doğrula; guard fail ederse calibration'ı kabul etme.
        """
        if len(self.gt_buffer) < self.min_calib_frames:
            return
        if len(self.gt_buffer) == self._calibrated_sample_count:
            return

        src = np.array(self.dpvo_buffer)
        dst = np.array(self.gt_buffer)

        try:
            if self.fit_method == "delta_linear":
                coef, intercept, predicted_delta, gt_delta = self._fit_delta_mapping()
                if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(intercept)):
                    raise ValueError("Delta kalibrasyonu NaN/Inf parametre üretti.")
                candidate_aligned = np.vstack(
                    [np.zeros((1, 3)), np.cumsum(predicted_delta, axis=0)]
                )
                candidate_gt = np.vstack([np.zeros((1, 3)), np.cumsum(gt_delta, axis=0)])
                guard_result = self.direction_guard.evaluate(
                    candidate_gt.tolist(), candidate_aligned.tolist()
                )
                if not guard_result["ok"]:
                    self.logger.warning(
                        f"Delta kalibrasyonu REDDEDİLDİ (guard): {guard_result['reason']} "
                        f"sim={guard_result['direction_similarity']:.3f}, "
                        f"scale={guard_result['scale_factor']:.3f}"
                    )
                    return

                self.delta_coef = coef
                self.delta_intercept = intercept
                self.alignment_anchor_offset = np.zeros(3, dtype=np.float64)
                self.is_calibrated = True
                self._calibrated_sample_count = len(self.gt_buffer)
                errors = np.linalg.norm(candidate_aligned - candidate_gt, axis=1)
                self.logger.info(
                    f"Delta kalibrasyonu güncellendi. N={len(predicted_delta)}, "
                    f"window={self.delta_window}, E={np.mean(errors):.4f}m, "
                    f"RMSE={np.sqrt(np.mean(errors ** 2)):.4f}m"
                )
                return

            if self.fit_method == "sim3":
                R, t, s = self._sim3_umeyama(src, dst)
            elif self.fit_method in {"linear", "ridge"}:
                src_mean = src.mean(axis=0)
                dst_mean = dst.mean(axis=0)
                src_centered = src - src_mean
                dst_centered = dst - dst_mean
                if self.fit_method == "linear":
                    rank = int(np.linalg.matrix_rank(src_centered))
                    condition = float(np.linalg.cond(src_centered))
                    if rank == 3 and np.isfinite(condition) and condition <= self.linear_condition_max:
                        weights = np.linalg.lstsq(src_centered, dst_centered, rcond=None)[0]
                    else:
                        self.logger.warning(
                            "Linear hizalama kötü koşullu (rank=%d, cond=%.3e); Ridge %.3g kullanılıyor.",
                            rank,
                            condition,
                            self.absolute_ridge_alpha,
                        )
                        regularizer = np.eye(3, dtype=np.float64) * self.absolute_ridge_alpha
                        weights = np.linalg.solve(
                            src_centered.T @ src_centered + regularizer,
                            src_centered.T @ dst_centered,
                        )
                else:
                    regularizer = np.eye(3, dtype=np.float64) * self.absolute_ridge_alpha
                    weights = np.linalg.solve(
                        src_centered.T @ src_centered + regularizer,
                        src_centered.T @ dst_centered,
                    )
                R = weights.T
                t = dst_mean - R @ src_mean
                s = 1.0
            else:
                raise ValueError(f"Desteklenmeyen dpvo.fit_method: {self.fit_method}")

            if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)) or not np.isfinite(s):
                raise ValueError("Kalibrasyon NaN/Inf parametre üretti.")
            if self.fit_method == "sim3" and s <= 0:
                raise ValueError(f"Geçersiz Sim3 scale: {s}")

            candidate_aligned = np.asarray(
                [s * (R @ point) + t for point in src], dtype=np.float64
            )

            # Guard ham koordinatları doğrudan karşılaştıramaz; Sim3 eksenleri
            # zaten döndürebilir. Aday dönüşüm uygulandıktan sonra doğrula.
            guard_result = self.direction_guard.evaluate(
                dst.tolist(), candidate_aligned.tolist()
            )
            if not guard_result["ok"]:
                self.logger.warning(
                    f"Kalibrasyon REDDEDİLDİ (guard): {guard_result['reason']} "
                    f"sim={guard_result['direction_similarity']:.3f}, "
                    f"scale={guard_result['scale_factor']:.3f}"
                )
                if self.is_calibrated:
                    # Eski iyi kalibrasyonu koru
                    self.logger.info("Eski kalibrasyon korunuyor.")
                return  # Yeni calibration'ı kabul etme

            self.sim3_R = R
            self.sim3_t = t
            self.sim3_s = s
            if self.anchor_at_calib_end:
                self.alignment_anchor_offset = dst[-1] - candidate_aligned[-1]
            else:
                self.alignment_anchor_offset = np.zeros(3, dtype=np.float64)
            self.is_calibrated = True
            self._calibrated_sample_count = len(self.gt_buffer)

            errors = np.linalg.norm(candidate_aligned - dst, axis=1)
            error = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            self.logger.info(
                f"Kalibrasyon güncellendi. "
                f"N={len(self.gt_buffer)}, "
                f"E={error:.4f}m, RMSE={rmse:.4f}m, scale={s:.4f}, "
                f"anchor={self.alignment_anchor_offset.tolist()}"
            )
        except Exception as e:
            self.logger.error(f"Kalibrasyon hatası: {e}")

    def process_frame(self, frame_idx: int, frame_path: str,
                      health_status: str,
                      gt_x: float = 0.0, gt_y: float = 0.0, gt_z: float = 0.0,
                      image: np.ndarray | None = None) -> tuple:
        self.frame_counter += 1
        status = None if health_status is None else str(health_status)
        position_before_frame = self.last_known_position.copy()
        if image is None:
            image = cv2.imread(frame_path)
        if image is None:
            self.logger.error(f"Frame okunamadı: {frame_path}")
            return (self.last_known_position[0],
                    self.last_known_position[1],
                    self.last_known_position[2])

        if self.causal_fusion is not None:
            try:
                healthy_gt = None
                if status == '1':
                    healthy_gt = np.asarray(
                        [float(gt_x), float(gt_y), float(gt_z)],
                        dtype=np.float64,
                    )
                self.causal_fusion.observe_frame(image, status, healthy_gt)
            except Exception as exc:
                # Positioning görevi yardımcı füzyondaki tek bir görsel
                # hatadan dolayı tahmin üretmeyi bırakmamalı.
                self.logger.error(
                    "Causal fusion frame gözlemi başarısız; baseline kullanılıyor: %s",
                    exc,
                )
        dpvo_image, dpvo_intrinsics, dpvo_H, dpvo_W = self._prepare_dpvo_input(image)

        if not self._dpvo_available:
            self._runtime_H = dpvo_H
            self._runtime_W = dpvo_W
            self._init_dpvo(dpvo_H, dpvo_W)

        dpvo_raw = None
        if self._dpvo_available and self.slam is not None:
            try:
                dpvo_raw = self.slam.process_frame(frame_idx, dpvo_image, dpvo_intrinsics)
            except Exception as e:
                self.logger.error(f"DPVO process_frame hatası: {e}")
                dpvo_raw = None

        dpvo_arr = None
        raw_delta = None
        if dpvo_raw is not None:
            dpvo_arr = np.asarray(dpvo_raw[:3], dtype=np.float64)
            if self.last_dpvo_raw is not None:
                raw_delta = dpvo_arr - self.last_dpvo_raw
                self.raw_delta_history.append(raw_delta.copy())
                if len(self.raw_delta_history) > max(100, self.delta_window * 4):
                    self.raw_delta_history = self.raw_delta_history[-max(100, self.delta_window * 4) :]
            self.last_dpvo_raw = dpvo_arr.copy()
        else:
            # Bir sonraki ölçümü birden fazla frame'in tek-frame deltası gibi
            # yorumlamamak için ardışıklığı sıfırla.
            self.last_dpvo_raw = None
            self.raw_delta_history = []

        # GT'nin bittiği ilk karede, örnek sayısı 10'un katı olmasa bile son
        # kullanılabilir kalibrasyonu mutlaka hesapla.
        if status == '0' and self.previous_health_status == '1':
            self._update_calibration()

        if status == '1':
            result = (float(gt_x), float(gt_y), float(gt_z))
            if dpvo_raw is not None:
                self.gt_buffer.append([float(gt_x), float(gt_y), float(gt_z)])
                self.dpvo_buffer.append(dpvo_arr.tolist())
                self.frame_indices.append(frame_idx)
                if self.previous_health_status == '1' and raw_delta is not None:
                    current_gt = np.array([float(gt_x), float(gt_y), float(gt_z)], dtype=np.float64)
                    self.delta_raw_buffer.append(raw_delta.copy())
                    self.delta_gt_buffer.append(current_gt - self.last_known_position)
                if len(self.gt_buffer) % 10 == 0:
                    self._update_calibration()
            gt_position = np.array([float(gt_x), float(gt_y), float(gt_z)], dtype=np.float64)
            if self.previous_health_status == '1':
                self.last_velocity = gt_position - self.last_known_position
            self.last_known_position = gt_position
            self.fallback_frame_count = 0
        elif status == '0':
            aligned = None
            if dpvo_arr is not None and self.is_calibrated:
                if self.fit_method == "delta_linear":
                    if raw_delta is not None and self.raw_delta_history:
                        aligned = self._predict_delta_position(raw_delta)
                else:
                    aligned = self._align_dpvo_to_gt(dpvo_arr)

            if aligned is not None:
                result = (float(aligned[0]), float(aligned[1]), float(aligned[2]))
                self.last_velocity = aligned - self.last_known_position
                self.last_known_position = aligned
                self.fallback_frame_count = 0
            elif dpvo_arr is not None and not self.is_calibrated:
                # Ham DPVO koordinatı NED değildir; kalibrasyon yokken bunu
                # göndermek ani ve anlamsız bir koordinat sıçraması üretir.
                if self.fallback_velocity:
                    fallback = self._advance_velocity_fallback()
                    result = tuple(float(value) for value in fallback)
                else:
                    result = tuple(float(value) for value in self.last_known_position)
                self.logger.warning("DPVO kalibrasyonu hazır değil; NED velocity fallback kullanıldı.")
            elif self.fallback_velocity:
                fallback = self._advance_velocity_fallback()
                result = (float(fallback[0]), float(fallback[1]), float(fallback[2]))
                self.logger.debug(f"Velocity fallback kullanıldı: {result}")
            else:
                result = (float(self.last_known_position[0]),
                          float(self.last_known_position[1]),
                          float(self.last_known_position[2]))
        else:
            result = (float(self.last_known_position[0]),
                      float(self.last_known_position[1]),
                      float(self.last_known_position[2]))

        if self.causal_fusion is not None:
            try:
                # Plane/relocalization yalnız yeni ve kalibre bir DPVO mutlak
                # pozu varsa uygulanır. DPVO init/dropout durumunda velocity
                # fallback'e correction sıçraması enjekte edilmez.
                fusion_status = status
                if status == '0' and aligned is None:
                    fusion_status = None
                fused = self.causal_fusion.fuse_position(
                    np.asarray(result, dtype=np.float64), fusion_status
                )
                if fused.shape != (3,) or not np.all(np.isfinite(fused)):
                    raise ValueError(f"geçersiz fused position: {fused}")
                result = tuple(float(value) for value in fused)
                if status == '0':
                    correction_delta = np.asarray(
                        self.causal_fusion.last_applied_correction_delta,
                        dtype=np.float64,
                    )
                    self.last_velocity = (
                        fused - position_before_frame - correction_delta
                    )
                    self.last_known_position = fused.copy()
            except Exception as exc:
                self.logger.error(
                    "Causal fusion çıktı hatası; baseline korunuyor: %s", exc
                )
        self.previous_health_status = status
        if self.runtime_log_every and self.frame_counter % self.runtime_log_every == 0:
            raw_text = "None" if dpvo_arr is None else np.array2string(dpvo_arr, precision=5)
            self.logger.info(
                "DPVO runtime frame=%d status=%s calibrated=%s raw=%s ned=(%.5f,%.5f,%.5f)",
                frame_idx,
                status,
                self.is_calibrated,
                raw_text,
                result[0],
                result[1],
                result[2],
            )
        return result
