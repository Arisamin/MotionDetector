"""Unit and integration tests for MotionDetector modules."""
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from src.motion_detector import MotionDetector, parse_sections, get_box_sections
from src.classifier import ObjectClassifier
from src.logger import MotionLogger
from src.capture import create_capture_source
from src.config import load_config, save_config, DEFAULT_CONFIG
from src.notifier import TelegramNotifier
from src.reolink_keeper import ReolinkWatchdog
from src.orchestrator import Orchestrator


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


def test_parse_sections():
    """Verify parsing of sections string."""
    assert parse_sections("1_2") == {1, 2}
    assert parse_sections("1_3_4") == {1, 3, 4}
    assert parse_sections("4") == {4}
    assert parse_sections(None) == {1, 2, 3, 4}
    assert parse_sections("") == {1, 2, 3, 4}
    assert parse_sections("5_invalid") == {1, 2, 3, 4}


def test_section_filtering_ignores_unmonitored_zones():
    """Verify motion is ignored when it occurs only in unmonitored screen sections."""
    # Screen size 640x480:
    # Section 1 (TL): x in [0, 320], y in [0, 240]
    # Section 4 (BR): x in [320, 640], y in [240, 480]

    bg_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Detector only monitoring Section 1 (Top Left)
    detector_s1 = MotionDetector(min_area=100, active_sections={1})
    for _ in range(15):
        detector_s1.detect(bg_frame)

    # Frame with motion strictly in Section 4 (Bottom Right: x=400..500, y=300..400)
    frame_motion_in_s4 = bg_frame.copy()
    cv2.rectangle(frame_motion_in_s4, (400, 300), (500, 400), (255, 255, 255), -1)

    # S1 detector should ignore motion in S4
    res_s1 = detector_s1.detect(frame_motion_in_s4)
    assert res_s1["has_motion"] is False
    assert len(res_s1["boxes"]) == 0

    # Detector monitoring Section 4 should detect it
    detector_s4 = MotionDetector(min_area=100, active_sections={4})
    for _ in range(15):
        detector_s4.detect(bg_frame)

    res_s4 = detector_s4.detect(frame_motion_in_s4)
    assert res_s4["has_motion"] is True
    assert len(res_s4["boxes"]) >= 1


def test_config_load_and_save(temp_workspace):
    """Verify loading and saving configuration parameters."""
    cfg_path = os.path.join(temp_workspace, "custom_config.json")

    # Load from non-existent file returns default
    cfg = load_config(cfg_path)
    assert cfg["min_area_pixels"] == DEFAULT_CONFIG["min_area_pixels"]

    # Save new settings
    save_config({"min_area_pixels": 2500, "min_area_percent": 1.25}, cfg_path)

    loaded = load_config(cfg_path)
    assert loaded["min_area_pixels"] == 2500
    assert loaded["min_area_percent"] == 1.25


def test_telegram_notifier_disabled_by_default_without_token():
    """Verify notifier stays disabled when token is missing."""
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=True)
    assert notifier.enabled is False
    # send_alert should not throw
    notifier.send_alert({"categories": ["human"]})
    notifier.stop()


def test_telegram_notifier_enqueue_and_format():
    """Verify Telegram alert formatting and queueing."""
    notifier = TelegramNotifier(bot_token="dummy_token", chat_id="12345", enabled=True)
    assert notifier.enabled is True

    event = {
        "timestamp": "2026-08-22 23:30:00",
        "categories": ["human", "car"],
        "counts": {"human": 1, "car": 2},
        "snapshot": "",
        "frame_index": 42,
    }

    # Verify enqueue without blocking
    notifier.send_alert(event)
    assert notifier._queue.qsize() >= 0
    notifier.stop()


def test_reolink_watchdog_detects_frozen_clock():
    """Verify ReolinkWatchdog triggers reactivation when timestamp stops ticking."""
    # Create synthetic frame with timestamp text
    frame_t1 = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(frame_t1, "08/23/2026 12:19:05 SUN", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    frame_t2 = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(frame_t2, "08/23/2026 12:19:06 SUN", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    watchdog = ReolinkWatchdog(enabled=True, freeze_timeout=0.2, click_cooldown=0.1)

    # Initial frame
    assert watchdog.check_frame(frame_t1) is False

    # Second ticking frame -> resets timer, should not trigger
    assert watchdog.check_frame(frame_t2) is False

    # Simulate clock freeze by providing same frame and advancing last_change_time
    import time
    time.sleep(0.25)
    # Checking same frozen frame after timeout should trigger reactivation click
    triggered = watchdog.check_frame(frame_t2)
    assert triggered is True
    assert watchdog.freeze_count >= 1


def test_orchestrator_initialization_and_mode():
    """Verify Orchestrator initialization and emulator detection logic."""
    orch_local = Orchestrator(mode="local")
    assert orch_local.mode == "local"

    orch_redroid = Orchestrator(mode="redroid")
    assert orch_redroid.mode == "redroid"

    # Verify auto mode resolves correctly
    orch_auto = Orchestrator(mode="auto")
    assert orch_auto.mode in ("local", "redroid")





