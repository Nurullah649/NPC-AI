"""
TEKNOFEST 2026 Havacılıkta Yapay Zeka - Bağlantı Yöneticisi.

- login
- progress check
- active session kontrolü
- sıradaki tek frame çekme
- sıradaki translation çekme
- referans objeleri çekme/cacheleme
- referans görüntüleri indirme
- prediction gönderme
- rate limit yönetimi
"""
import json
import logging
import os
import time
from pathlib import Path

import requests
from decouple import config


class ConnectionHandler:
    """Yarışma sunucusu ile tüm iletişimi yönetir."""

    def __init__(self, base_url, username=None, password=None):
        self.base_url = base_url.rstrip("/") + "/"
        self.auth_token = None
        self.session_name = None
        self.progress = None
        self.active_refs = None
        self.ref_image_paths = {}

        # API endpoint'leri
        self.url_login = self.base_url + "auth/"
        self.url_frames = self.base_url + "frames/"
        self.url_translations = self.base_url + "translation/"
        self.url_prediction = self.base_url + "prediction/"
        self.url_session = self.base_url + "session/"
        self.url_progress = self.base_url + "progress/"
        self.url_references = self.base_url + "references/"

        # Rate limit yönetimi
        self.last_prediction_time = 0
        self.min_frame_interval = 0.75  # 80 frame/min -> ~0.75s

        self.logger = logging.getLogger(self.__class__.__name__)

        if username and password:
            self.login(username, password)

    def login(self, username, password):
        """Sunucuya giriş yap."""
        payload = {'username': username, 'password': password}
        try:
            response = requests.post(self.url_login, data=payload, timeout=10)
            if response.status_code == 200:
                self.auth_token = response.json()['token']
                self.logger.info(f"Giriş başarılı: {username}")
                return True
            else:
                self.logger.error(f"Giriş başarısız: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Giriş isteği başarısız: {e}")
            return False

    def get_progress(self):
        """Progress bilgisini çek - sıradaki frame indeksini öğren."""
        if not self.auth_token:
            self.logger.error("Progress: Auth token yok")
            return None
        headers = {'Authorization': f'Token {self.auth_token}'}
        try:
            response = requests.get(self.url_progress, headers=headers, timeout=30)
            if response.status_code == 200:
                self.progress = response.json()
                return self.progress
            else:
                self.logger.error(f"Progress alınamadı: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Progress hatası: {e}")
            return None

    def get_next_frame(self):
        """Sıradaki frame'i çek (tek frame)."""
        if not self.auth_token:
            self.logger.error("get_next_frame: Auth token yok")
            return None
        headers = {'Authorization': f'Token {self.auth_token}'}
        try:
            response = requests.get(self.url_frames, headers=headers, timeout=60)
            if response.status_code == 200:
                frames = response.json()
                if isinstance(frames, list) and len(frames) > 0:
                    return frames[0]
                elif isinstance(frames, dict):
                    return frames
                return frames
            else:
                self.logger.error(f"Frame alınamadı: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Frame hatası: {e}")
            return None

    def get_next_translation(self):
        """Sıradaki translation'ı çek (tek translation)."""
        if not self.auth_token:
            self.logger.error("get_next_translation: Auth token yok")
            return None
        headers = {'Authorization': f'Token {self.auth_token}'}
        try:
            response = requests.get(self.url_translations, headers=headers, timeout=60)
            if response.status_code == 200:
                translations = response.json()
                if isinstance(translations, list) and len(translations) > 0:
                    return translations[0]
                elif isinstance(translations, dict):
                    return translations
                return translations
            else:
                self.logger.error(f"Translation alınamadı: {response.text}")
                return None
        except Exception as e:
            self.logger.error(f"Translation hatası: {e}")
            return None

    def get_references(self):
        """Aktif referans objelerini çek ve cache'le."""
        if not self.auth_token:
            self.logger.error("get_references: Auth token yok")
            return [], {}
        headers = {'Authorization': f'Token {self.auth_token}'}
        try:
            response = requests.get(self.url_references, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.active_refs = data.get('active_refs', [])
                ref_images = data.get('ref_image_paths', {})
                self.ref_image_paths.update(ref_images)
                self.logger.info(f"{len(self.active_refs)} aktif referans yüklendi.")
                return self.active_refs, self.ref_image_paths
            else:
                self.logger.warning(f"Referanslar alınamadı: {response.text}")
                return [], {}
        except Exception as e:
            self.logger.error(f"Referans hatası: {e}")
            return [], {}

    def download_reference_image(self, ref_url, save_dir):
        """Referans görüntüsünü indir/cache'le.

        Returns:
            Yerel dosya yolu veya None.
        """
        os.makedirs(save_dir, exist_ok=True)
        ref_filename = ref_url.split("/")[-1]
        ref_path = os.path.join(save_dir, ref_filename)

        if os.path.exists(ref_path):
            return ref_path

        try:
            response = requests.get(ref_url, timeout=60)
            response.raise_for_status()
            with open(ref_path, 'wb') as f:
                f.write(response.content)
            self.logger.info(f"Referans indirildi: {ref_url} -> {ref_path}")
            return ref_path
        except Exception as e:
            self.logger.error(f"Referans indirme hatası {ref_url}: {e}")
            return None

    def download_frame(self, img_url, save_dir):
        """Frame görüntüsünü indir.

        Returns:
            Yerel dosya yolu veya None.
        """
        os.makedirs(save_dir, exist_ok=True)
        img_name = img_url.split("/")[-1]
        img_path = os.path.join(save_dir, img_name)

        if os.path.exists(img_path):
            return img_path

        try:
            response = requests.get(img_url, timeout=60)
            response.raise_for_status()
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
        except Exception as e:
            self.logger.error(f"Frame indirme hatası {img_url}: {e}")
            return None

    def send_prediction(self, predictions):
        """Tahmini sunucuya gönder. Rate limit uygula."""
        # Rate limit
        elapsed = time.time() - self.last_prediction_time
        if elapsed < self.min_frame_interval:
            time.sleep(self.min_frame_interval - elapsed)

        if not self.auth_token:
            self.logger.error("send_prediction: Auth token yok")
            return None

        payload = json.dumps(predictions.create_payload(self.base_url))
        headers = {
            'Authorization': f'Token {self.auth_token}',
            'Content-Type': 'application/json',
        }

        for attempt in range(3):
            try:
                response = requests.post(
                    self.url_prediction, headers=headers,
                    data=payload, timeout=60
                )
                self.last_prediction_time = time.time()

                if response.status_code == 201:
                    self.logger.info("Tahmin başarıyla gönderildi.")
                    return response
                elif response.status_code == 406:
                    self.logger.warning("Tahmin daha önce gönderilmiş (406).")
                    return response
                else:
                    self.logger.error(
                        f"Tahmin gönderilemedi ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
                    time.sleep(1)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Tahmin gönderme hatası: {e}")
                time.sleep(2)

        self.logger.error("Tahmin gönderilemedi (3 deneme).")
        return None

    def get_session_info(self):
        """Aktif session bilgisini al."""
        if not self.auth_token:
            return None
        headers = {'Authorization': f'Token {self.auth_token}'}
        try:
            response = requests.get(self.url_session, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
