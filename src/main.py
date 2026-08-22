"""Main CLI entry point for the MotionDetector agent."""
import argparse
import sys
import time
import cv2

from src.motion_detector import MotionDetector, parse_sections
from src.classifier import ObjectClassifier
from src.logger import MotionLogger
from src.capture import create_capture_source
from src.config import load_config
from src.calibrator import run_calibration
from src.notifier import TelegramNotifier


def parse_args():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Autonomous Motion Detector & Object Classifier Agent for Camera Feeds."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: video file path (.mp4/.avi), RTSP/HTTP stream URL, webcam index (e.g. '0'), or 'screen'.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch interactive visual calibration mode to adjust and save the minimum motion square size.",
    )
    parser.add_argument(
        "--sections",
        "--zones",
        type=str,
        default=None,
        dest="sections",
        help="Underscore-separated screen sections to monitor: 1=Top-Left, 2=Top-Right, 3=Bottom-Left, 4=Bottom-Right (e.g. '1_2', '1_3_4'). Default: all sections.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI window displayed). Recommended for servers / background runs.",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        default=cfg.get("telegram_enabled", False),
        help="Enable Telegram notifications for motion events.",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_false",
        dest="telegram",
        help="Disable Telegram notifications.",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=cfg.get("telegram_bot_token", ""),
        help="Telegram Bot API Token (can also be configured via config.json or TELEGRAM_BOT_TOKEN env var).",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=str,
        default=cfg.get("telegram_chat_id", ""),
        help="Telegram Chat/Group ID (e.g. -1004474060156).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=cfg.get("model", "yolov8n.pt"),
        help="YOLO model weights to use (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=cfg.get("confidence_threshold", 0.35),
        help="Classifier confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--min-area",
        type=str,
        default=None,
        help="Minimum motion area in pixels (e.g. '500') or screen percentage (e.g. '1.0%%'). Default: loaded from config.json.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=cfg.get("cooldown_seconds", 1.0),
        help="Minimum seconds after motion detection before reporting/logging another motion event (default: 1.0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process before exiting (0 = infinite).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save CSV/JSON event logs (default: logs).",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="snapshots",
        help="Directory to save snapshot images (default: snapshots).",
    )
    return parser.parse_args()


def run_pipeline(args):
    if args.calibrate:
        run_calibration(source_str=args.source)
        return 0

    cfg = load_config()

    # Determine sections
    sections_str = args.sections if args.sections is not None else cfg.get("sections", "1_2_3_4")
    active_sections = parse_sections(sections_str)
    sections_display = "_".join(str(s) for s in sorted(active_sections)) if active_sections != {1, 2, 3, 4} else "All (1_2_3_4)"

    print("=" * 60)
    print("      Starting MotionDetector Agent")
    print(f" Source:      {args.source}")
    print(f" Sections:    {sections_display} (1=TL, 2=TR, 3=BL, 4=BR)")
    print(f" Headless:    {args.headless}")
    print(f" Model:       {args.model} (conf={args.conf})")
    print(f" Cooldown:    {args.cooldown} s")
    print(f" Logs Dir:    {args.log_dir}")
    print(f" Snapshot:    {args.snapshot_dir}")
    print("=" * 60)

    try:
        source = create_capture_source(args.source)
    except Exception as e:
        print(f"[ERROR] Failed to initialize source: {e}", file=sys.stderr)
        return 1

    # Read first frame to determine resolution and area threshold
    ret, first_frame = source.read()
    if not ret or first_frame is None:
        print("[ERROR] Could not read initial frame from source.", file=sys.stderr)
        source.release()
        return 1

    h_frame, w_frame = first_frame.shape[:2]
    total_area = w_frame * h_frame

    # Compute min area in pixels
    min_area_val = args.min_area
    if min_area_val is not None:
        min_area_str = str(min_area_val).strip()
        if min_area_str.endswith("%"):
            pct = float(min_area_str.replace("%", ""))
            min_area_px = int((pct / 100.0) * total_area)
        else:
            min_area_px = int(min_area_str)
    else:
        # Use config.json values
        if "min_area_percent" in cfg:
            min_area_px = int((cfg["min_area_percent"] / 100.0) * total_area)
        else:
            min_area_px = int(cfg.get("min_area_pixels", 500))

    min_area_px = max(10, min_area_px)
    area_percent = (min_area_px / total_area) * 100.0
    print(f"[INFO] Frame Resolution: {w_frame}x{h_frame} | Min Motion Area: {min_area_px:,} px^2 ({area_percent:.2f}% of screen)")

    detector = MotionDetector(min_area=min_area_px, active_sections=active_sections)
    classifier = ObjectClassifier(model_name=args.model, confidence_threshold=args.conf)
    logger = MotionLogger(
        log_dir=args.log_dir,
        snapshot_dir=args.snapshot_dir,
        cooldown_seconds=args.cooldown,
        active_sections=active_sections,
    )
    notifier = TelegramNotifier(
        bot_token=args.telegram_token,
        chat_id=args.telegram_chat_id,
        enabled=args.telegram,
    )
    if notifier.enabled:
        print("[INFO] Telegram Notifications: ENABLED (Group: ReolinkMotionDetector)")
    else:
        print("[INFO] Telegram Notifications: DISABLED")

    frame_count = 0
    total_events = 0
    category_totals = {"human": 0, "car": 0, "other": 0}
    start_time = time.time()
    last_fps_time = start_time
    fps = 0.0

    frame = first_frame
    has_frame = True

    try:
        while has_frame and frame is not None:
            frame_count += 1
            now = time.time()

            # Calculate running FPS every 10 frames
            if frame_count % 10 == 0:
                elapsed = now - last_fps_time
                if elapsed > 0:
                    fps = 10.0 / elapsed
                last_fps_time = now

            # Step 1: Detect motion
            motion_res = detector.detect(frame)

            # Step 2: If motion is present, classify objects
            detections = []
            if motion_res["has_motion"]:
                detections = classifier.classify_frame(frame, motion_boxes=motion_res["boxes"])

            # Step 3: Log events, save snapshots, and notify Telegram
            if detections:
                event = logger.log_event(frame, detections, frame_idx=frame_count, fps=fps)
                if event:
                    total_events += 1
                    cats = event["categories"]
                    for c in cats:
                        category_totals[c] = category_totals.get(c, 0) + 1
                    cats_str = ", ".join(cats)
                    print(f"[{event['timestamp']}] [Frame #{frame_count:05d}] Motion Detected: [{cats_str}] (FPS: {fps:.1f})")

                    # Dispatch Telegram alert (non-blocking)
                    notifier.send_alert(event)

            # Step 4: Display GUI window if not in headless mode
            if not args.headless:
                display_frame = MotionLogger.annotate_frame(frame, detections, active_sections=active_sections)
                cv2.imshow("MotionDetector Feed", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or Esc
                    print("\n[INFO] User requested stop.")
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"\n[INFO] Reached max frames limit ({args.max_frames}).")
                break

            # Read next frame
            has_frame, frame = source.read()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        source.release()
        notifier.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("      MotionDetector Session Summary")
    print(f" Total Frames Processed: {frame_count}")
    print(f" Total Elapsed Time:     {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    print(f" Total Motion Events:    {total_events}")
    print(f" Humans Detected:        {category_totals.get('human', 0)}")
    print(f" Cars/Vehicles Detected: {category_totals.get('car', 0)}")
    print(f" Other Objects/Motion:   {category_totals.get('other', 0)}")
    print(f" Event Logs CSV:         {logger.csv_path}")
    print(f" Event Logs JSONL:       {logger.json_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline(parse_args()))
