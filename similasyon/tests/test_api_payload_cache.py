"""Sunucudan alınan frame/translation JSON kayıt testleri."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.connection_handler import ConnectionHandler


def make_handler(tmp_path):
    handler = ConnectionHandler("http://example.test/")
    handler.img_save_path = str(tmp_path)
    handler.video_name = "test_session/"
    return handler


def test_frame_and_translation_records_are_saved(tmp_path):
    handler = make_handler(tmp_path)
    frame = {"url": "/frames/1/", "image_url": "/media/frame_000001.webp"}
    translation = {
        "url": "/translation/1/",
        "image_url": "/media/frame_000001.webp",
        "translation_x": 1.25,
    }

    handler.save_frame_to_file(frame)
    handler.save_translation_to_file(translation)

    session_dir = tmp_path / "test_session"
    assert json.loads((session_dir / "frames.json").read_text()) == [frame]
    assert json.loads((session_dir / "translations.json").read_text()) == [translation]


def test_repeated_frame_is_updated_instead_of_duplicated(tmp_path):
    handler = make_handler(tmp_path)
    handler.save_frame_to_file({
        "url": "/frames/1/",
        "image_url": "/media/frame_000001.webp",
        "video_name": "old",
    })
    updated = {
        "url": "/frames/1/",
        "image_url": "/media/frame_000001.webp",
        "video_name": "new",
    }

    handler.save_frame_to_file(updated)

    saved = json.loads(
        (tmp_path / "test_session" / "frames.json").read_text()
    )
    assert saved == [updated]
