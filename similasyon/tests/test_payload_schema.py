"""Payload schema testleri."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.constants import classes, landing_statuses, moving_statuses
from src.detected_object import DetectedObject
from src.detected_translation import DetectedTranslation
from src.reference_prediction import ReferencePrediction
from src.frame_predictions import FramePredictions


class TestDetectedObject:
    def test_basic(self):
        obj = DetectedObject(0, "1", "0", 10, 20, 100, 200)
        assert obj.cls == 0
        assert obj.landing_status == "1"
        assert obj.moving_status == "0"

    def test_payload_has_moving_status(self):
        obj = DetectedObject(1, "-1", "-1", 10, 20, 100, 200)
        payload = obj.create_payload("http://localhost:1025/")
        assert 'moving_status' in payload
        assert payload['moving_status'] == '-1'

    def test_payload_contains_all_fields(self):
        obj = DetectedObject(2, "1", "-1", 50, 50, 150, 150)
        payload = obj.create_payload("http://localhost:1025/")
        assert 'cls' in payload
        assert 'landing_status' in payload
        assert 'moving_status' in payload
        assert 'top_left_x' in payload
        assert 'top_left_y' in payload
        assert 'bottom_right_x' in payload
        assert 'bottom_right_y' in payload

    def test_cls_tuple_normalization(self):
        obj = DetectedObject((2,), "1", "-1", 0, 0, 10, 10)
        payload = obj.create_payload("http://localhost:1025/")
        assert 'cls' in payload

    def test_bbox_numeric(self):
        obj = DetectedObject(0, "0", "0", 10.5, 20.7, 100.3, 200.9)
        payload = obj.create_payload("http://localhost:1025/")
        assert payload['top_left_x'] == '10'
        assert payload['top_left_y'] == '20'
        assert payload['bottom_right_x'] == '100'
        assert payload['bottom_right_y'] == '200'


class TestDetectedTranslation:
    def test_basic(self):
        t = DetectedTranslation(1.0, 2.0, 3.0)
        assert t.translation_x == 1.0
        assert t.translation_y == 2.0
        assert t.translation_z == 3.0

    def test_default_z(self):
        t = DetectedTranslation(1.0, 2.0)
        assert t.translation_z == 0.0

    def test_payload_has_z(self):
        t = DetectedTranslation(1.0, 2.0, 3.0)
        payload = t.create_payload()
        assert 'translation_z' in payload
        assert payload['translation_z'] == '3.0'

    def test_sanitize_nan(self):
        t = DetectedTranslation(float('nan'), 2.0, 3.0)
        assert t.translation_x == 0.0
        assert t.translation_y == 2.0

    def test_sanitize_none(self):
        t = DetectedTranslation(None, 2.0, 3.0)
        assert t.translation_x == 0.0

    def test_sanitize_inf(self):
        t = DetectedTranslation(1.0, float('inf'), 3.0)
        assert t.translation_y == 0.0


class TestReferencePrediction:
    def test_basic(self):
        r = ReferencePrediction("ref_url", "frame_url", 10, 20, 100, 200)
        assert r.reference_url == "ref_url"
        assert r.frame_url == "frame_url"

    def test_payload(self):
        r = ReferencePrediction("ref/1/", "frame/1/", 10.0, 20.0, 100.0, 200.0)
        payload = r.create_payload()
        assert payload['reference_url'] == "ref/1/"
        assert payload['frame_url'] == "frame/1/"
        assert payload['top_left_x'] == '10'


class TestFramePredictions:
    def test_basic(self):
        fp = FramePredictions("frame/1/", "img/1.jpg", "video1")
        assert len(fp.detected_objects) == 0
        assert len(fp.detected_translations) == 0
        assert len(fp.reference_predictions) == 0

    def test_add_objects(self):
        fp = FramePredictions("f/1", "i/1", "v1")
        obj = DetectedObject(0, "1", "0", 10, 20, 100, 200)
        fp.add_detected_object(obj)
        assert len(fp.detected_objects) == 1

    def test_add_translation(self):
        fp = FramePredictions("f/1", "i/1", "v1")
        t = DetectedTranslation(1.0, 2.0, 3.0)
        fp.add_translation_object(t)
        assert len(fp.detected_translations) == 1

    def test_add_ref_prediction(self):
        fp = FramePredictions("f/1", "i/1", "v1")
        r = ReferencePrediction("ref", "frame", 0, 0, 10, 10)
        fp.add_reference_prediction(r)
        assert len(fp.reference_predictions) == 1

    def test_payload_contains_reference_predictions(self):
        fp = FramePredictions("f/1", "i/1", "v1")
        r = ReferencePrediction("ref/1", "frame/1", 10, 20, 100, 200)
        fp.add_reference_prediction(r)
        payload = fp.create_payload("http://localhost:1025/")
        assert 'reference_predictions' in payload
        assert len(payload['reference_predictions']) == 1

    def test_payload_all_fields(self):
        fp = FramePredictions("f/1", "i/1", "v1")
        fp.add_detected_object(DetectedObject(0, "1", "0", 0, 0, 10, 10))
        fp.add_translation_object(DetectedTranslation(1, 2, 3))
        fp.add_reference_prediction(ReferencePrediction("r/1", "f/1", 0, 0, 10, 10))
        payload = fp.create_payload("http://localhost:1025/")
        assert 'frame' in payload
        assert 'detected_objects' in payload
        assert 'detected_translations' in payload
        assert 'reference_predictions' in payload
