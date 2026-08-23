"""Reolink stream keeper and freeze watchdog module.

Monitors the Reolink on-screen timestamp strip (clock/seconds).
If the clock freezes for more than the timeout (default 10s), automatically
dispatches a simulated mouse click to resume the live stream.
"""
import os
import sys
import time
import ctypes
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class ReolinkWatchdog:
    """Monitors on-screen camera clock and auto-clicks to reactivate stalled Reolink streams."""

    def __init__(
        self,
        enabled: bool = True,
        freeze_timeout: float = 10.0,
        click_cooldown: float = 5.0,
        roi: Optional[Tuple[float, float, float, float]] = None,
        click_target: Optional[Tuple[int, int]] = None,
    ):
        """
        :param enabled: Whether the watchdog is active.
        :param freeze_timeout: Seconds of frozen clock before triggering a click (default: 10.0s).
        :param click_cooldown: Minimum seconds between click attempts (default: 5.0s).
        :param roi: Optional relative (x_rel, y_rel, w_rel, h_rel) for the timestamp strip.
                    Defaults to top 15% strip of frame: (0.1, 0.0, 0.8, 0.15).
        :param click_target: Optional screen (x, y) coordinates for the click. If None, uses frame center.
        """
        self.enabled = enabled
        self.freeze_timeout = freeze_timeout
        self.click_cooldown = click_cooldown
        # Default timestamp strip: upper portion of the frame where timestamp and reolink logo sit
        self.roi_rel = roi if roi is not None else (0.10, 0.0, 0.80, 0.15)
        self.click_target = click_target

        self.last_change_time = time.time()
        self.last_click_time = 0.0
        self.last_roi_gray: Optional[np.ndarray] = None
        self.freeze_count = 0

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the timestamp strip region of interest from the frame."""
        h, w = frame.shape[:2]
        xr, yr, wr, hr = self.roi_rel
        x1 = max(0, int(xr * w))
        y1 = max(0, int(yr * h))
        x2 = min(w, int((xr + wr) * w))
        y2 = min(h, int((yr + hr) * h))
        return frame[y1:y2, x1:x2]

    def check_frame(self, frame: np.ndarray, screen_pos: Optional[Tuple[int, int]] = None) -> bool:
        """
        Check if the timestamp has updated in this frame. If frozen, triggers a click.

        :param frame: Current BGR video/screen frame.
        :param screen_pos: Optional (x, y) top-left screen position of the frame (for screen clicks).
        :return: True if a click was dispatched to reactivate, False otherwise.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return False

        now = time.time()
        roi = self.get_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if self.last_roi_gray is None or self.last_roi_gray.shape != gray.shape:
            self.last_roi_gray = gray.copy()
            self.last_change_time = now
            return False

        # Compute absolute difference in the timestamp strip
        diff = cv2.absdiff(self.last_roi_gray, gray)
        # Count pixels that changed significantly (digits ticking)
        changed_pixels = np.count_nonzero(diff > 25)

        # A change in seconds digits typically modifies 20-500 pixels in the ROI
        if changed_pixels >= 15:
            self.last_change_time = now
            self.last_roi_gray = gray.copy()
            return False

        # If unchanged, check if freeze duration exceeded timeout
        frozen_duration = now - self.last_change_time

        if frozen_duration >= self.freeze_timeout and (now - self.last_click_time >= self.click_cooldown):
            self.freeze_count += 1
            print(f"\n[WARN] [Reolink Watchdog] Camera clock frozen for {frozen_duration:.1f}s (> {self.freeze_timeout:.0f}s threshold)!")
            print(f"[INFO] [Reolink Watchdog] Dispatching reactivation click #{self.freeze_count} on Reolink app screen...")

            # Determine click coordinates
            h, w = frame.shape[:2]
            if self.click_target is not None:
                click_x, click_y = self.click_target
            elif screen_pos is not None:
                click_x = screen_pos[0] + w // 2
                click_y = screen_pos[1] + h // 2
            else:
                # Default: center of primary screen / window
                click_x = w // 2
                click_y = h // 2

            self._send_mouse_click(click_x, click_y)
            self.last_click_time = now
            self.last_change_time = now  # Reset timer to give stream time to buffer
            self.last_roi_gray = gray.copy()
            return True

        return False

    def _send_mouse_click(self, x: int, y: int):
        """Simulate a mouse click at screen coordinates (x, y)."""
        if sys.platform == "win32":
            try:
                # Save current cursor position
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                orig_pt = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(orig_pt))

                # Move to target and click
                ctypes.windll.user32.SetCursorPos(int(x), int(y))
                time.sleep(0.05)
                # MOUSEEVENTF_LEFTDOWN = 0x0002, MOUSEEVENTF_LEFTUP = 0x0004
                ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.05)
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.05)

                # Restore cursor position
                ctypes.windll.user32.SetCursorPos(orig_pt.x, orig_pt.y)
                print(f"[SUCCESS] [Reolink Watchdog] Simulated click sent at ({x}, {y}) and cursor restored.")
            except Exception as e:
                print(f"[ERROR] [Reolink Watchdog] Win32 click simulation failed: {e}", file=sys.stderr)
        else:
            # Fallback for Linux / X11 / Headless ADB
            print(f"[INFO] [Reolink Watchdog] Non-Windows platform: Simulating tap at ({x}, {y}).")
            try:
                os.system(f"adb shell input tap {x} {y} > /dev/null 2>&1")
            except Exception:
                pass
