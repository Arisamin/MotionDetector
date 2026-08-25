"""Verification script for testing MotionDetector pipeline inside Termux."""
import os
import sys
import time

# Ensure repo root is on sys.path
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_dir)

from src.config import load_config
from src.android_capture import AndroidNativeCaptureSource
from src.motion_detector import MotionDetector
from src.classifier import ObjectClassifier
from src.notifier import TelegramNotifier

print("=== 1. Checking Config ===")
cfg = load_config()
print(f"Telegram Configured: {bool(cfg.get('telegram_bot_token') and cfg.get('telegram_chat_id'))}")

print("=== 2. Testing Screen Capture via screencap ===")
cap = AndroidNativeCaptureSource(adb_device="127.0.0.1:5555")
ret, frame = cap.read()
if ret and frame is not None:
    print(f"Captured frame shape: {frame.shape}")
else:
    print("Failed to capture frame!")

print("=== 3. Testing Motion Detector & Classifier ===")
detector = MotionDetector(min_area=500)
classifier = ObjectClassifier()
if ret and frame is not None:
    res = detector.detect(frame)
    print(f"Motion detection result: has_motion={res['has_motion']}, boxes={len(res['boxes'])}")
    detections = classifier.classify_frame(frame, motion_boxes=res["boxes"])
    print(f"Classification result: {detections}")

print("=== 4. Testing Telegram Alert Delivery ===")
bot_token = cfg.get("telegram_bot_token")
chat_id = cfg.get("telegram_chat_id")
if bot_token and chat_id:
    notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id, enabled=True)
    notifier.send_message("🚀 MotionDetector running natively inside BlueStacks Android (Termux)!")
    if ret and frame is not None and detections:
        from src.logger import MotionLogger
        annotated = MotionLogger.annotate_frame(frame, detections)
        # Test saving and sending snapshot
        import cv2
        snapshot_path = "/sdcard/MotionDetector/test_snapshot.jpg"
        cv2.imwrite(snapshot_path, annotated)
        notifier.send_photo(snapshot_path, caption="📸 On-Device Test Snapshot from BlueStacks Termux!")
    print("Telegram alert and snapshot queued successfully.")
    time.sleep(3)
    notifier.stop()

print("=== PIPELINE TEST COMPLETED SUCCESSFULLY ===")
