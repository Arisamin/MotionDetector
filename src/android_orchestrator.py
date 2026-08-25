"""Native Android Orchestrator for running directly inside BlueStacks / Termux / ReDroid."""
import os
import sys
import subprocess
import time
import argparse
import logging
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.notifier import TelegramNotifier
from src.android_capture import AndroidNativeCaptureSource
from src.motion_detector import MotionDetector
from src.classifier import ObjectClassifier
from src.logger import MotionLogger
from src.reolink_keeper import ReolinkWatchdog

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AndroidOrchestrator")


class AndroidOrchestrator:
    """Manages Reolink app launch and on-device motion detection with Telegram alerts."""

    PACKAGE_NAME = "com.mcu.reolink"

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.notifier = None
        bot_token = self.config.get("telegram_bot_token")
        chat_id = self.config.get("telegram_chat_id")
        if bot_token and chat_id:
            self.notifier = TelegramNotifier(
                bot_token=bot_token,
                chat_id=chat_id,
                enabled=self.config.get("telegram_enabled", True),
            )
        # Ensure ADB local daemon is connected
        try:
            subprocess.run(["adb", "connect", "127.0.0.1:5555"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    def launch_reolink(self) -> bool:
        """Launches the Reolink app using Android monkey/am command or ADB."""
        logger.info("Launching Reolink app on Android...")
        commands = [
            ["am", "start", "-n", "com.mcu.reolink/com.android.bc.login.WelcomeActivity"],
            ["/system/bin/am", "start", "-n", "com.mcu.reolink/com.android.bc.login.WelcomeActivity"],
            ["adb", "-s", "127.0.0.1:5555", "shell", "am", "start", "-n", "com.mcu.reolink/com.android.bc.login.WelcomeActivity"],
            ["adb", "-s", "emulator-5554", "shell", "am", "start", "-n", "com.mcu.reolink/com.android.bc.login.WelcomeActivity"],
            ["adb", "-s", "127.0.0.1:5555", "shell", "monkey", "-p", self.PACKAGE_NAME, "-c", "android.intent.category.LAUNCHER", "1"],
            ["/system/bin/monkey", "-p", self.PACKAGE_NAME, "-c", "android.intent.category.LAUNCHER", "1"],
        ]
        for cmd in commands:
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                if res.returncode == 0:
                    logger.info(f"Reolink launch command succeeded via: {' '.join(cmd[:4])}")
                    return True
            except Exception:
                continue
        logger.warning("Could not launch Reolink automatically. Please ensure Reolink is open.")
        return False

    def tap_screen(self, x: int = 800, y: int = 450):
        """Simulates a screen tap using ADB or Android input tap."""
        commands = [
            ["adb", "-s", "127.0.0.1:5555", "shell", "input", "tap", str(x), str(y)],
            ["adb", "-s", "emulator-5554", "shell", "input", "tap", str(x), str(y)],
            ["adb", "shell", "input", "tap", str(x), str(y)],
            ["/system/bin/input", "tap", str(x), str(y)],
            ["input", "tap", str(x), str(y)],
        ]
        for cmd in commands:
            try:
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if res.returncode == 0:
                    logger.info(f"Tapped screen at ({x}, {y})")
                    return
            except Exception:
                continue
        logger.warning(f"Could not simulate tap at ({x}, {y})")

    def run_pipeline(
        self,
        min_area: int = 2500,
        conf_threshold: float = 0.35,
        cooldown: float = 3.0,
        mask_top_percent: float = 0.12,
        mask_bottom_percent: float = 0.12,
        freeze_timeout: float = 10.0,
    ):
        """Runs the motion detection loop on Android."""
        logger.info(f"Starting on-device motion detection (min_area={min_area} px, mask_top={mask_top_percent:.0%}, mask_bottom={mask_bottom_percent:.0%}, freeze_timeout={freeze_timeout}s)...")
        
        capture_source = AndroidNativeCaptureSource()
        detector = MotionDetector(
            min_area=min_area,
            mask_top_percent=mask_top_percent,
            mask_bottom_percent=mask_bottom_percent,
            consecutive_frames_required=2,
        )
        classifier = ObjectClassifier(confidence_threshold=conf_threshold)
        logger_service = MotionLogger(cooldown_seconds=cooldown)
        watchdog = ReolinkWatchdog(enabled=True, freeze_timeout=freeze_timeout)

        if self.notifier:
            self.notifier.send_message("Android Orchestrator started. Monitoring Reolink stream on-device.")

        frame_count = 0
        suppress_alerts_until = time.time() + 4.0  # Initial 4s startup warmup

        try:
            while True:
                ret, frame = capture_source.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

                frame_count += 1
                now = time.time()

                # Reolink stream freeze watchdog (checks clock in unmasked top HUD)
                clicked = watchdog.check_frame(frame)
                if clicked:
                    logger.info("Watchdog detected frozen clock - dispatched wake-up tap.")
                    detector.reset_background()
                    suppress_alerts_until = now + 8.0
                    logger.info("Wake-up active: Suppressing motion alerts for 8.0s while stream stabilizes...")

                # Motion detection (with HUD & controls masked out + temporal persistence)
                motion_res = detector.detect(frame)

                # Skip alert dispatch during startup / wake-up stabilization window
                is_suppressed = now < suppress_alerts_until
                if is_suppressed:
                    time.sleep(0.1)
                    continue

                if motion_res["has_motion"]:
                    detections = classifier.classify_frame(frame, motion_boxes=motion_res["boxes"])
                    if detections:
                        logged = logger_service.log_event(frame, detections, frame_idx=frame_count)
                        if logged and self.notifier:
                            self.notifier.send_alert(logged)
                            summary = ", ".join(f"{d['category']} ({d['confidence']:.0%})" for d in detections)
                            logger.info(f"Alert dispatched: {summary}")

                # Sleep slightly to conserve CPU/battery
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
        finally:
            capture_source.release()
            if self.notifier:
                self.notifier.stop()


def main():
    parser = argparse.ArgumentParser(description="Android Native Motion Detector Orchestrator")
    parser.add_argument("--tap-x", type=int, default=800, help="X coordinate to tap camera card (default: 800)")
    parser.add_argument("--tap-y", type=int, default=450, help="Y coordinate to tap camera card (default: 450)")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum contour area in px (default: from config.json or 2500 px)")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Alert cooldown in seconds (default: 3.0s)")
    parser.add_argument("--mask-top", type=float, default=0.08, help="Fraction of top screen to mask (default: 0.08)")
    parser.add_argument("--freeze-timeout", type=float, default=10.0, help="Seconds before reactivating frozen stream (default: 10.0s)")
    args = parser.parse_args()

    orchestrator = AndroidOrchestrator()
    orchestrator.launch_reolink()
    
    cfg = orchestrator.config
    min_area = args.min_area if args.min_area is not None else cfg.get("min_area_pixels", 2500)
    freeze_timeout = args.freeze_timeout if args.freeze_timeout is not None else cfg.get("freeze_timeout_seconds", 10.0)

    # Wait for app to open then tap to start stream
    logger.info("Waiting 8 seconds for Reolink app to initialize...")
    time.sleep(8)
    orchestrator.tap_screen(x=args.tap_x, y=args.tap_y)
    time.sleep(4)

    orchestrator.run_pipeline(
        min_area=min_area,
        conf_threshold=args.conf,
        cooldown=args.cooldown,
        mask_top_percent=args.mask_top,
        freeze_timeout=freeze_timeout,
    )


if __name__ == "__main__":
    main()
