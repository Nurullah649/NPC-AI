"""Insan tracker birim testleri (Faz P4).

Test senaryolari (plan §4 Faz P4):
- Sabit kamera + yuruyen insan
- Kamera pan/tilt + sabit insan
- Bir ve iki kare detector dropout
- Iki insanin kesismesi
- Frame kenarindan yeni insan girisi
- Tile sinirindan gecen insan
- Ayni insanin YOLO ve crop modelden cift gelmesi
- Uzun kayipta yanlis bbox uretiminin durmasu
- Ilk detection bastirilmamali
- Dusuk guvenli detection track baslatabilmeli
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from person.tracker import PersonTrackerV2, PersonTrackV2, TrackState
from motion.camera_motion import CameraTransform, CameraModel


@pytest.fixture
def config():
    return {
        "person_tracking": {
            "enabled": True,
            "high_conf_threshold": 0.40,
            "low_conf_threshold": 0.10,
            "new_track_threshold": 0.30,
            "max_age": 4,
            "max_coast_output_frames": 1,
            "use_camera_compensation": True,
            "confirm_hits": 2,
            "confirm_window": 6,
            "kalman": {
                "std_weight_position": 0.05,
                "std_weight_velocity": 0.00625,
                "min_position_noise": 0.1,
                "max_position_noise": 100.0,
            },
        }
    }


@pytest.fixture
def tracker(config):
    return PersonTrackerV2(config)


def make_shift_transform(dx, dy, confidence=0.8):
    t = CameraTransform()
    t.model_type = CameraModel.SHIFT
    t.shift = np.array([dx, dy], dtype=np.float32)
    t.match_count = 50
    t.inlier_count = 40
    t.inlier_ratio = 0.8
    t.reprojection_error = 1.0
    t.confidence = confidence
    return t


def make_person_det(bbox, conf=0.6):
    x1, y1, x2, y2 = bbox
    return {"cls": 1, "cls_name": "Insan", "conf": conf, "bbox": (x1, y1, x2, y2)}


class TestPersonTrackV2:
    """PersonTrackV2 birim testleri."""

    def test_init(self, config):
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        assert track.id == 0
        assert track.state == TrackState.TENTATIVE
        assert track.hits == 1
        assert track.kf is not None

    def test_predict_does_not_increment_age(self, config):
        """M0 düzeltme: predict() age'i artirmaz."""
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        track.predict()
        assert track.age == 0, "predict() age'i artirmamali"

    def test_mark_missed_increments_age(self, config):
        """age yalnizca mark_missed()'de artar."""
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        track.mark_missed(1)
        assert track.age == 1

    def test_confirm_after_hits(self, config):
        """confirm_hits sayida update sonrasi CONFIRMED olmali."""
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        assert track.state == TrackState.TENTATIVE
        track.update((102, 100, 152, 250), 1, 0.6)
        assert track.state == TrackState.CONFIRMED  # confirm_hits=2, hits=2

    def test_single_miss_does_not_lost(self, config):
        """Tek kaçırmada LOST'a gecis yok (M0 düzeltme)."""
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        track.update((102, 100, 152, 250), 1, 0.6)
        assert track.state == TrackState.CONFIRMED
        track.mark_missed(2)
        assert track.state == TrackState.CONFIRMED, \
            "Tek kaçırmada CONFIRMED -> LOST olmamali"

    def test_predict_with_camera(self, config):
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        cam = make_shift_transform(20.0, 0.0)
        pred = track.predict(cam)
        assert pred["center"][0] > 100  # kamera ile tasindi

    def test_coast_bbox_limited(self, config):
        """Coast bbox max_coast_output_frames ile sinirli."""
        track = PersonTrackV2(0, (100, 100, 150, 250), 0, 0.6, config)
        track.update((102, 100, 152, 250), 1, 0.6)
        track.mark_missed(2)
        coast = track.get_coast_bbox()
        assert coast is not None  # 1 kare coast
        track.mark_missed(3)
        coast2 = track.get_coast_bbox()
        assert coast2 is None  # max_coast_output_frames=1 asildi


class TestPersonTrackerV2:
    """PersonTrackerV2 integration testleri."""

    def test_first_detection_not_suppressed(self, tracker):
        """Detector'in eşiği geçen ilk gozlemi bastirilmamali (M0 düzeltme)."""
        detections = [make_person_det((100, 100, 150, 250), conf=0.6)]
        result = tracker.update(detections, 0, None)
        assert len(result) >= 1, "Ilk detection bastirilmamali"
        assert result[0]["cls"] == 1

    def test_low_conf_new_track(self, tracker):
        """Dusuk guvenli detection (new_track_threshold uzerinde) track baslatabilmeli."""
        detections = [make_person_det((100, 100, 150, 250), conf=0.35)]
        result = tracker.update(detections, 0, None)
        assert len(result) >= 1
        assert len(tracker.tracks) == 1

    def test_walking_person_tracked(self, tracker):
        """Sabit kamera + yuruyen insan track edilmeli."""
        track_id = None
        for frame in range(10):
            x = 100 + frame * 15
            detections = [make_person_det((x, 100, x + 50, 250), conf=0.6)]
            result = tracker.update(detections, frame, None)

            for det in result:
                if det.get("cls") == 1 and "_track_id" in det:
                    tid = det["_track_id"]
                    if track_id is None:
                        track_id = tid
                    else:
                        assert tid == track_id, \
                            f"Frame {frame}: track_id degisti {track_id} -> {tid}"

        assert track_id is not None, "Track olusturulmadi"

    def test_camera_pan_stationary_person(self, tracker):
        """Kamera pan + sabit insan -> ayni track ID korunmali.

        Kamera saga pan yapinca, dunyada sabit olan insanin goruntudeki
        pozisyonu sola kayar. Kamera transformu bu kaymayi kompanze eder.
        """
        cam = make_shift_transform(-20.0, 0.0)  # objects shift left
        track_id = None

        for frame in range(8):
            x_offset = -frame * 20  # person shifts left in image
            bbox = (500 + x_offset, 100, 550 + x_offset, 250)  # dunyada sabit
            detections = [make_person_det(bbox, conf=0.6)]
            result = tracker.update(detections, frame, cam)

            for det in result:
                if det.get("cls") == 1 and "_track_id" in det:
                    tid = det["_track_id"]
                    if track_id is None:
                        track_id = tid
                    else:
                        assert tid == track_id

        assert track_id is not None

    def test_detector_dropout_one_frame(self, tracker):
        """1 kare detector dropout -> track kaybolmamali."""
        track_id = None
        for frame in range(8):
            if frame == 4:
                detections = []
            else:
                x = 100 + frame * 10
                detections = [make_person_det((x, 100, x + 50, 250), conf=0.6)]
            result = tracker.update(detections, frame, None)

            for det in result:
                if det.get("cls") == 1 and "_track_id" in det:
                    if track_id is None:
                        track_id = det["_track_id"]

        assert track_id is not None
        assert len(tracker.tracks) > 0, "Track dropout sonrasi kaybolmamali"

    def test_two_persons_crossing(self, tracker):
        """Iki insan kesisirse track ID'ler karismamali (mumkun oldugunca)."""
        for frame in range(10):
            x1 = 100 + frame * 20
            x2 = 600 - frame * 20
            d1 = make_person_det((x1, 100, x1 + 40, 250), conf=0.6)
            d2 = make_person_det((x2, 100, x2 + 40, 250), conf=0.6)
            tracker.update([d1, d2], frame, None)

        assert len(tracker.tracks) >= 2, "Iki track olmali"

    def test_frame_edge_entry(self, tracker):
        """Frame kenarindan yeni insan girisi track baslatabilmeli."""
        detections = [make_person_det((0, 100, 30, 250), conf=0.5)]
        result = tracker.update(detections, 0, None)
        assert len(result) >= 1
        assert len(tracker.tracks) == 1

    def test_duplicate_detection_merged(self, tracker):
        """Ayni insanin YOLO ve crop'tan gelen cift gozlemi handle edilmeli."""
        bbox = (100, 100, 150, 250)
        d1 = make_person_det(bbox, conf=0.6)
        d2 = make_person_det((102, 101, 152, 251), conf=0.5)
        d2["_source"] = "crop"
        result = tracker.update([d1, d2], 0, None)
        assert len(tracker.tracks) == 1, "Cift gozlem tek track olmali"

    def test_long_miss_stops_bbox(self, tracker):
        """Uzun kayipta yanlis bbox uretimi durmali."""
        for frame in range(3):
            detections = [make_person_det((100, 100, 150, 250), conf=0.6)]
            tracker.update(detections, frame, None)

        for frame in range(3, 15):
            result = tracker.update([], frame, None)

        coast_bboxes = [d for d in result if d.get("_source") == "tracker_coast"]
        assert len(coast_bboxes) == 0, "Uzun kayipta coast bbox uretilmemeli"

    def test_non_person_status_preserved(self, tracker):
        """Insan disi siniflar tracker'dan etkilenmemeli."""
        detections = [
            {"cls": 0, "cls_name": "Tasit", "conf": 0.8, "bbox": (500, 500, 600, 600)},
            make_person_det((100, 100, 150, 250), conf=0.6),
        ]
        result = tracker.update(detections, 0, None)
        vehicle_dets = [d for d in result if d["cls"] == 0]
        assert len(vehicle_dets) == 1
        assert "moving_status" not in vehicle_dets[0] or \
            vehicle_dets[0].get("moving_status") is None or \
            "_track_id" not in vehicle_dets[0]

    def test_active_track_bboxes_for_scheduler(self, tracker):
        """get_active_track_bboxes tile scheduler icin dogru format vermeli."""
        for frame in range(5):
            x = 100 + frame * 10
            detections = [make_person_det((x, 100, x + 50, 250), conf=0.6)]
            tracker.update(detections, frame, None)

        bboxes = tracker.get_active_track_bboxes()
        assert len(bboxes) > 0
        for b in bboxes:
            assert "bbox" in b
            assert "id" in b
            assert len(b["bbox"]) == 4
