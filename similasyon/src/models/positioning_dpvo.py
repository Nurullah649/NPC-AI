import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch


class PositioningDPVO:
    """DPVO tabanlı pozisyon kestirimi - Görev 2.

    - health_status == '1' iken GT x/y/z değerlerini aynen gönderir
      ve DPVO raw pose ile GT arasında Sim(3) hizalama biriktirir.
    - health_status == '0' iken DPVO raw pose'u hizalama ile
      referans koordinata dönüştürür ve gönderir.
    - DPVO fail ederse velocity/Kalman fallback kullanır.
    """

    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        dpvo_cfg = config.get("dpvo", {})
        self.min_calib_frames = dpvo_cfg.get("min_calib_frames", 50)
        self.fit_method = dpvo_cfg.get("fit_method", "sim3")
        self.fallback_velocity = dpvo_cfg.get("fallback_velocity", True)

        # Model path'leri
        model_paths = config.get("model_paths", {})
        self.dpvo_weight = self._resolve_path(model_paths.get("dpvo", "weights/dpvo/dpvo.pth"))
        self.dpvo_cfg_path = self._resolve_path(model_paths.get("dpvo_cfg", "config/dpvo/npc.yaml"))
        self.calib_path = self._resolve_path(model_paths.get("camera_calib", "config/camera/calib.txt"))

        # DPVO instance (lazy init)
        self.slam = None
        self._dpvo_available = False

        # Kalibrasyon / hizalama buffer'ları
        self.gt_buffer = []       # (x, y, z)
        self.dpvo_buffer = []     # (x, y, z)
        self.frame_indices = []   # frame idx

        # Hizalama parametreleri
        self.sim3_R = np.eye(3)
        self.sim3_t = np.zeros(3)
        self.sim3_s = 1.0
        self.is_calibrated = False

        # Fallback
        self.last_known_position = np.array([0.0, 0.0, 0.0])
        self.last_velocity = np.array([0.0, 0.0, 0.0])
        self.last_health_was_one = False

        # Kamera kalibrasyonu
        self.intrinsics = self._load_calibration()

        # Frame sayacı
        self.frame_counter = 0

        self.logger.info(
            f"PositioningDPVO başlatıldı. "
            f"min_calib_frames={self.min_calib_frames}, "
            f"fit_method={self.fit_method}"
        )

    def _resolve_path(self, rel_path: str) -> str:
        """Dosya yolunu similasyon/ köküne göre çözümle."""
        candidates = [
            Path(rel_path),
            Path(".") / rel_path,
            Path(__file__).resolve().parent.parent.parent / rel_path,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return str(Path(rel_path).resolve())

    def _load_calibration(self) -> np.ndarray:
        """Kamera kalibrasyon dosyasını yükle."""
        try:
            if os.path.exists(self.calib_path):
                intrinsics = np.loadtxt(self.calib_path)
                if intrinsics.shape == (3, 3):
                    self.logger.info(f"Kalibrasyon yüklendi: {self.calib_path}")
                    return intrinsics
            self.logger.warning(f"Kalibrasyon dosyası bulunamadı: {self.calib_path}, varsayılan kullanılıyor.")
        except Exception as e:
            self.logger.warning(f"Kalibrasyon yüklenemedi: {e}, varsayılan kullanılıyor.")

        # Varsayılan intrinsics
        return np.array([
            [1413.3, 0.0, 950.0639],
            [0.0, 1418.8, 543.3796],
            [0.0, 0.0, 1.0],
        ])

    def _init_dpvo(self, H: int, W: int):
        """DPVO'yu lazy initialize et."""
        if self._dpvo_available:
            return

        try:
            from .dpvo_standalone import DPVOStandalone
            self.slam = DPVOStandalone(
                weight_path=self.dpvo_weight,
                config_path=self.dpvo_cfg_path,
                calib_path=self.calib_path,
                ht=H, wd=W,
            )
            self._dpvo_available = True
            self.logger.info("DPVO başarıyla başlatıldı.")
        except ImportError:
            self.logger.warning(
                "DPVO modülü bulunamadı. DPVO olmadan çalışılıyor. "
                "Lütfen third_party/DPVO kurulumunu yapın."
            )
            self._dpvo_available = False
        except Exception as e:
            self.logger.error(f"DPVO başlatılamadı: {e}")
            self._dpvo_available = False

    def _sim3_umeyama(self, src: np.ndarray, dst: np.ndarray) -> tuple:
        """Umeyama Sim(3) hizalama.

        Args:
            src: (N, 3) kaynak noktalar (DPVO raw)
            dst: (N, 3) hedef noktalar (GT)

        Returns:
            (R, t, s) 3x3 rotation, 3 translation, scalar scale
        """
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H_mat = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H_mat)

        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        src_var = np.sum(np.linalg.norm(src_centered, axis=1) ** 2) / src.shape[0]
        s = np.trace(np.diag(S)) / src_var if src_var > 1e-10 else 1.0

        t = dst_mean - s * R @ src_mean

        return R, t, s

    def _align_dpvo_to_gt(self, dpvo_pos: np.ndarray) -> np.ndarray:
        """Hizalanmış DPVO pozisyonunu GT koordinatına dönüştür."""
        if not self.is_calibrated:
            return dpvo_pos  # Henüz kalibrasyon yok
        return self.sim3_s * (self.sim3_R @ dpvo_pos) + self.sim3_t

    def _update_calibration(self):
        """Birikmiş verilerle Sim(3) hizalamasını güncelle."""
        if len(self.gt_buffer) < self.min_calib_frames:
            return

        src = np.array(self.dpvo_buffer)
        dst = np.array(self.gt_buffer)

        try:
            if self.fit_method == "sim3":
                R, t, s = self._sim3_umeyama(src, dst)
            else:
                # Ridge regression fallback
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, fit_intercept=True)
                model.fit(src, dst)
                R = model.coef_
                t = model.intercept_
                s = 1.0

            self.sim3_R = R
            self.sim3_t = t
            self.sim3_s = s
            self.is_calibrated = True

            # Kalibrasyon hatasını hesapla
            aligned = np.array([self._align_dpvo_to_gt(p) for p in src])
            error = np.mean(np.linalg.norm(aligned - dst, axis=1))
            self.logger.info(
                f"Kalibrasyon güncellendi. "
                f"N={len(self.gt_buffer)}, "
                f"Ortalama hata={error:.4f}m, "
                f"scale={s:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Kalibrasyon hatası: {e}")

    def process_frame(self, frame_idx: int, frame_path: str,
                      health_status: str,
                      gt_x: float = 0.0, gt_y: float = 0.0, gt_z: float = 0.0) -> tuple:
        """Tek frame işle.

        Returns:
            (predicted_x, predicted_y, predicted_z)
        """
        self.frame_counter += 1

        # Frame'i oku
        image = cv2.imread(frame_path)
        if image is None:
            self.logger.error(f"Frame okunamadı: {frame_path}")
            return (self.last_known_position[0],
                    self.last_known_position[1],
                    self.last_known_position[2])

        H, W = image.shape[:2]

        # DPVO'yu başlat (lazy)
        if not self._dpvo_available:
            self._init_dpvo(H, W)

        # DPVO raw pose al
        dpvo_raw = None
        if self._dpvo_available and self.slam is not None:
            try:
                dpvo_raw = self.slam.process_frame(frame_idx, frame_path, self.intrinsics)
            except Exception as e:
                self.logger.error(f"DPVO process_frame hatası: {e}")
                dpvo_raw = None

        if health_status == '1':
            # GT'yi aynen gönder, kalibrasyon biriktir
            result = (float(gt_x), float(gt_y), float(gt_z))

            if dpvo_raw is not None:
                self.gt_buffer.append([float(gt_x), float(gt_y), float(gt_z)])
                self.dpvo_buffer.append([float(dpvo_raw[0]), float(dpvo_raw[1]), float(dpvo_raw[2])])
                self.frame_indices.append(frame_idx)

                # Periyodik kalibrasyon güncellemesi
                if len(self.gt_buffer) % 10 == 0:
                    self._update_calibration()

            self.last_known_position = np.array([float(gt_x), float(gt_y), float(gt_z)])
            self.last_health_was_one = True

        elif health_status == '0':
            if dpvo_raw is not None and self.is_calibrated:
                # Hizalanmış DPVO kullan
                dpvo_arr = np.array([dpvo_raw[0], dpvo_raw[1], dpvo_raw[2]])
                aligned = self._align_dpvo_to_gt(dpvo_arr)
                result = (float(aligned[0]), float(aligned[1]), float(aligned[2]))

                # Velocity hesapla
                self.last_velocity = aligned - self.last_known_position
                self.last_known_position = aligned
                self.last_health_was_one = False

            elif dpvo_raw is not None and not self.is_calibrated:
                # Kalibrasyon yok, raw DPVO'yu döndür
                result = (float(dpvo_raw[0]), float(dpvo_raw[1]), float(dpvo_raw[2]))
                self.last_known_position = np.array(result)

            elif self.fallback_velocity and self.last_health_was_one:
                # DPVO fail: velocity fallback
                fallback = self.last_known_position + self.last_velocity
                result = (float(fallback[0]), float(fallback[1]), float(fallback[2]))
                self.last_known_position = fallback
                self.logger.debug(f"Velocity fallback kullanıldı: {result}")
            else:
                # Son bilinen pozisyon
                result = (float(self.last_known_position[0]),
                          float(self.last_known_position[1]),
                          float(self.last_known_position[2]))
        else:
            # Bilinmeyen health_status
            result = (float(self.last_known_position[0]),
                      float(self.last_known_position[1]),
                      float(self.last_known_position[2]))

        return result
