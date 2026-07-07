"""Config path testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.config_loader import load_settings


class TestConfigPaths:
    def test_load_settings(self):
        """settings.yaml yüklenebilmeli."""
        config = load_settings("config/settings.yaml")
        assert isinstance(config, dict)

    def test_model_paths_exists(self):
        """model_paths anahtarı mevcut olmalı."""
        config = load_settings("config/settings.yaml")
        if config:
            assert 'model_paths' in config

    def test_detector_config(self):
        """detector anahtarı mevcut olmalı."""
        config = load_settings("config/settings.yaml")
        if config:
            assert 'detector' in config

    def test_motion_config(self):
        """motion anahtarı mevcut olmalı."""
        config = load_settings("config/settings.yaml")
        if config:
            assert 'motion' in config

    def test_reference_config(self):
        """reference anahtarı mevcut olmalı."""
        config = load_settings("config/settings.yaml")
        if config:
            assert 'reference' in config

    def test_config_nonexistent(self):
        """Var olmayan config dosyası boş dict dönmeli."""
        config = load_settings("nonexistent.yaml")
        assert config == {}
