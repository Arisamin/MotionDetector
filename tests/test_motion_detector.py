"""Unit and integration tests for MotionDetector modules."""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from src.motion_detector import MotionDetector
from src.classifier import ObjectClassifier
from src.logger import MotionLogger
from src.capture import create_capture_source


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for logs and snapshots."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_motion_detector_detects_movement():
    """Verify MotionDetector detects moving white square on black background."""
    detector = MotionDetector(min_area=100)

    # Static background frames to train MOG2
    bg_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(15):
        detector.detect(bg_frame)

    # Frame with moving object (white rectangle)
    moving_frame = bg_frame.copy()
    cv2.rectangle(moving_frame, (200, 200), (300, 300), (255, 255, 255), -1)

    result = detector.detect(moving_frame)
    assert result["has_motion"] is True
    assert len(result["boxes"]) >= 1


def test_object_classifier_category_mapping():
    """Verify class mapping rules."""
    classifier = ObjectClassifier(model_name="yolov8n.pt")

    assert classifier.map_class_to_category("person") == ObjectClassifier.CATEGORY_HUMAN
    assert classifier.map_class_to_category("car") == ObjectClassifier.CATEGORY_CAR
    assert classifier.map_class_to_category("truck") == ObjectClassifier.CATEGORY_CAR
    assert classifier.map_class_to_category("motorcycle") == ObjectClassifier.CATEGORY_CAR
    assert classifier.map_class_to_category("bus") == ObjectClassifier.CATEGORY_CAR
    assert classifier.map_class_to_category("dog") == ObjectClassifier.CATEGORY_OTHER
    assert classifier.map_class_to_category("chair") == ObjectClassifier.CATEGORY_OTHER


def test_object_classifier_fallback_motion_boxes():
    """Verify that unrecognized motion blobs are categorized as 'other'."""
    classifier = ObjectClassifier(model_name="yolov8n.pt")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    motion_boxes = [(50, 50, 100, 100)]

    detections = classifier.classify_frame(blank_frame, motion_boxes=motion_boxes)
    assert len(detections) >= 1
    assert detections[0]["category"] == ObjectClassifier.CATEGORY_OTHER


def test_logger_creates_csv_and_json(temp_workspace):
    """Verify MotionLogger writes valid CSV, JSONL, and snapshot files."""
    log_dir = os.path.join(temp_workspace, "logs")
    snap_dir = os.path.join(temp_workspace, "snaps")
    logger = MotionLogger(log_dir=log_dir, snapshot_dir=snap_dir, cooldown_seconds=0.0)

    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        {"category": "human", "raw_label": "person", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
        {"category": "car", "raw_label": "car", "confidence": 0.88, "bbox": [60, 60, 120, 120]},
    ]

    event = logger.log_event(dummy_frame, detections, frame_idx=1, fps=30.0)
    assert event is not None
    assert "human" in event["categories"]
    assert "car" in event["categories"]
    assert event["counts"]["human"] == 1
    assert event["counts"]["car"] == 1

    # Check files exist
    assert os.path.exists(logger.csv_path)
    assert os.path.exists(logger.json_path)
    assert os.path.exists(event["snapshot"])


def test_logger_cooldown_debounces_reports(temp_workspace):
    """Verify that events within cooldown_seconds are throttled."""
    import time
    log_dir = os.path.join(temp_workspace, "logs")
    snap_dir = os.path.join(temp_workspace, "snaps")
    logger = MotionLogger(log_dir=log_dir, snapshot_dir=snap_dir, cooldown_seconds=1.0)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [{"category": "human", "raw_label": "person", "confidence": 0.9, "bbox": [10, 10, 50, 50]}]

    # First event should be logged
    evt1 = logger.log_event(dummy_frame, detections)
    assert evt1 is not None

    # Immediate second event (within 1.0s) should be suppressed
    evt2 = logger.log_event(dummy_frame, detections)
    assert evt2 is None

    # Wait for cooldown to expire
    time.sleep(1.05)
    evt3 = logger.log_event(dummy_frame, detections)
    assert evt3 is not None



def test_end_to_end_synthetic_video_pipeline(temp_workspace):
    """Generate a synthetic video and run the pipeline from end to end."""
    video_path = os.path.join(temp_workspace, "test_feed.mp4")
    log_dir = os.path.join(temp_workspace, "logs")
    snap_dir = os.path.join(temp_workspace, "snaps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (320, 240))

    # 10 static frames
    for _ in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out.write(frame)

    # 10 frames with moving shape
    for i in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 50 + i * 10
        cv2.rectangle(frame, (x, 100), (x + 40, 140), (255, 255, 255), -1)
        out.write(frame)

    out.release()

    # Run detection on synthetic video
    source = create_capture_source(video_path)
    detector = MotionDetector(min_area=100)
    classifier = ObjectClassifier(model_name="yolov8n.pt")
    logger = MotionLogger(log_dir=log_dir, snapshot_dir=snap_dir, cooldown_seconds=0.0)

    events_logged = 0
    while True:
        ret, frame = source.read()
        if not ret or frame is None:
            break
        motion = detector.detect(frame)
        if motion["has_motion"]:
            dets = classifier.classify_frame(frame, motion_boxes=motion["boxes"])
            if dets:
                evt = logger.log_event(frame, dets)
                if evt:
                    events_logged += 1

    source.release()
    assert events_logged > 0
    assert os.path.exists(logger.csv_path)
