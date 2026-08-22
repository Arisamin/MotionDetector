"""Interactive visual calibration tool to configure minimum motion threshold size."""
import sys
import os
import cv2
import numpy as np
from typing import Optional, Tuple
from src.capture import create_capture_source
from src.config import save_config, load_config, CONFIG_FILE


def run_calibration(
    source_str: str = "0",
    config_path: str = CONFIG_FILE,
    initial_percent: float = 1.0,
) -> Optional[dict]:
    """
    Launch an interactive calibration window with an adjustable reference square.

    :param source_str: Video/camera/screen source to grab background frame.
    :param config_path: Path to config.json to save result.
    :param initial_percent: Starting square size as % of screen area.
    :return: Dictionary with calibrated pixel area and percentage, or None if cancelled.
    """
    print("=" * 60)
    print("      Starting Visual Motion Size Calibrator")
    print("=" * 60)
    print("Controls:")
    print("  [+] or [UP Arrow] / [W]   : Increase square size by +10%")
    print("  [-] or [DOWN Arrow] / [S] : Decrease square size by -10%")
    print("  [Right Arrow] / []]       : Fine increase (+2%)")
    print("  [Left Arrow] / [[]        : Fine decrease (-2%)")
    print("  [Page Up] / [Page Down]   : +/-50%")
    print("  [T]                       : Type custom percentage change in console")
    print("  [Enter] or [Space]        : Accept and save minimum size")
    print("  [Esc] or [Q]              : Cancel without saving")
    print("=" * 60)

    # Grab a frame
    frame = None
    try:
        source = create_capture_source(source_str)
        for _ in range(5):  # Read a few frames in case camera needs auto-exposure
            ret, temp_frame = source.read()
            if ret and temp_frame is not None:
                frame = temp_frame
        source.release()
    except Exception as e:
        print(f"[WARN] Could not grab frame from source '{source_str}': {e}")

    # Fallback frame if source unavailable
    if frame is None or frame.size == 0:
        print("[INFO] Using default 1280x720 canvas for calibration.")
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    h_frame, w_frame = frame.shape[:2]
    total_frame_area = float(w_frame * h_frame)

    # Load existing config percent if present
    cfg = load_config(config_path)
    current_percent = cfg.get("min_area_percent", initial_percent)
    if current_percent <= 0:
        current_percent = initial_percent

    window_name = "Motion Size Calibration - [Enter]=Save, [Esc]=Cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # Calculate square side length from target percentage
        target_area = (current_percent / 100.0) * total_frame_area
        side_len = int(np.sqrt(max(10, target_area)))
        side_len = min(side_len, min(w_frame, h_frame) - 10)
        actual_area = side_len * side_len
        actual_percent = (actual_area / total_frame_area) * 100.0

        # Draw frame copy with square & HUD
        display = frame.copy()

        # Center coordinates
        cx, cy = w_frame // 2, h_frame // 2
        x1 = cx - side_len // 2
        y1 = cy - side_len // 2
        x2 = x1 + side_len
        y2 = y1 + side_len

        # Semi-transparent overlay inside square
        overlay = display.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

        # Bright border
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Crosshair center
        cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)

        # HUD background bar
        cv2.rectangle(display, (0, 0), (w_frame, 70), (20, 20, 20), -1)
        cv2.line(display, (0, 70), (w_frame, 70), (0, 255, 255), 1)

        # HUD Text
        hud_line1 = f"MINIMAL MOTION SIZE: {side_len}x{side_len} px | Area: {actual_area:,} px^2 ({actual_percent:.2f}% of screen)"
        hud_line2 = "[Up/W]: +10% | [Down/S]: -10% | [PgUp/PgDn]: +/-50% | [T]: Type % | [Enter]: Save | [Esc]: Cancel"

        cv2.putText(display, hud_line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, hud_line2, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Dimensions label next to square
        dim_label = f"{side_len} px ({actual_percent:.2f}%)"
        cv2.putText(display, dim_label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)

        # Wait for key with full extended keycode support
        key = cv2.waitKeyEx(0)

        # Key mappings:
        # Increase (+10%): Up Arrow (Windows: 2490368, Linux: 65362), '+', '=', 'w', 'W'
        if key in (2490368, 65362, ord("+"), ord("="), ord("w"), ord("W")):
            current_percent = max(0.01, current_percent * 1.10)
        # Decrease (-10%): Down Arrow (Windows: 2621440, Linux: 65364), '-', '_', 's', 'S'
        elif key in (2621440, 65364, ord("-"), ord("_"), ord("s"), ord("S")):
            current_percent = max(0.01, current_percent * 0.90)
        # Right Arrow / ']': Fine increase (+2%)
        elif key in (2555904, 65363, ord("]")):
            current_percent = max(0.01, current_percent * 1.02)
        # Left Arrow / '[': Fine decrease (-2%)
        elif key in (2424832, 65361, ord("[")):
            current_percent = max(0.01, current_percent * 0.98)
        # Page Up: +50%
        elif key in (2162688, 65365):
            current_percent = max(0.01, current_percent * 1.50)
        # Page Down: -50%
        elif key in (2228224, 65366):
            current_percent = max(0.01, current_percent * 0.50)
        # 't' or 'T': type custom change
        elif key in (ord("t"), ord("T")):
            try:
                cv2.destroyAllWindows()
                user_input = input(f"\nCurrent size is {actual_percent:.2f}% ({actual_area} px). Enter change percentage (e.g. '+25', '-15', or target '2.5'): ").strip()
                if user_input:
                    if user_input.startswith("+"):
                        delta = float(user_input[1:].replace("%", ""))
                        current_percent = current_percent * (1.0 + delta / 100.0)
                    elif user_input.startswith("-"):
                        delta = float(user_input[1:].replace("%", ""))
                        current_percent = current_percent * (1.0 - delta / 100.0)
                    else:
                        target = float(user_input.replace("%", ""))
                        current_percent = target
            except Exception as e:
                print(f"[ERROR] Invalid input: {e}")
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # Enter (13) or Space (32): Accept and save
        elif key in (13, 10, 32):
            cv2.destroyAllWindows()
            result_data = {
                "min_area_pixels": int(actual_area),
                "min_area_percent": round(actual_percent, 4),
            }
            save_config(result_data, config_path)
            print("\n" + "=" * 60)
            print(" [SUCCESS] Calibration Saved to config.json:")
            print(f"   Minimum Motion Area: {actual_area:,} pixels")
            print(f"   Screen Percentage:   {actual_percent:.2f}% ({side_len}x{side_len} px square)")
            print("=" * 60)
            return result_data
        # Esc (27) or 'q'
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyAllWindows()
            print("\n[INFO] Calibration cancelled by user.")
            return None


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    run_calibration(source_str=src)
