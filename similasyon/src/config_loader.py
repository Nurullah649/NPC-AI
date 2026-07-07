"""Config yükleme yardımcısı."""
import os
from pathlib import Path

import yaml


def load_settings(config_path: str = None) -> dict:
    """settings.yaml dosyasını yükle.

    Args:
        config_path: Config dosya yolu. None ise otomatik ara.

    Returns:
        Dict config.
    """
    if config_path is None:
        candidates = [
            "config/settings.yaml",
            "../config/settings.yaml",
            str(Path(__file__).resolve().parent.parent / "config" / "settings.yaml"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                config_path = cand
                break

    if not config_path or not os.path.exists(config_path):
        return {}  # Empty config

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
