"""Reolink stream keeper and freeze watchdog module.

Monitors the Reolink on-screen timestamp strip (clock/seconds).
If the clock freezes for more than the timeout (default 10s), automatically
dispatches a targeted simulated mouse click to resume the live stream.
"""
import os
import sys
import time
import ctypes
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List


def find_bluestacks_or_reolink_window() -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]]]:
    """
    Locate the BlueStacks or Reolink window handle and screen rectangle.
    Returns (hwnd, (left, top, right, bottom)) or (None, None).
    """
    if sys.platform != "win32":
        return None, None

    user32 = ctypes.windll.user32
    target_keywords = ["bluestacks app player", "bluestacks", "reolink", "hd-player"]
    found_hwnd = None
    found_rect = None

    def enum_proc(hwnd, lParam):
        nonlocal found_hwnd, found_rect
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value.lower()
                for kw in target_keywords:
                    if kw in title:
                        rect = (ctypes.c_long * 4)()
                        user32.GetWindowRect(hwnd, rect)
                        w = rect[2] - rect[0]
                        h = rect[3] - rect[1]
                        if w > 250 and h > 250:
                            found_hwnd = hwnd
                            found_rect = (rect[0], rect[1], rect[2], rect[3])
                            return False  # stop enumeration
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
    return found_hwnd, found_rect


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
        :param click_target: Optional screen (x, y) coordinates for the click. If None, auto-detects window center.
        """
        self.enabled = enabled
        self.freeze_timeout = freeze_timeout
        self.click_cooldown = click_cooldown
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
        Check if the timestamp has updated in this frame. If frozen, triggers a targeted click.

        :param frame: Current BGR video/screen frame.
        :param screen_pos: Optional (x, y) top-left screen position of the frame.
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
        changed_pixels = np.count_nonzero(diff > 25)

        # A change in seconds digits typically modifies 15-500 pixels in the ROI
        if changed_pixels >= 15:
            self.last_change_time = now
            self.last_roi_gray = gray.copy()
            return False

        # If unchanged, check if freeze duration exceeded timeout
        frozen_duration = now - self.last_change_time

        if frozen_duration >= self.freeze_timeout and (now - self.last_click_time >= self.click_cooldown):
            self.freeze_count += 1
            print(f"\n[WARN] [Reolink Watchdog] Camera clock frozen for {frozen_duration:.1f}s (> {self.freeze_timeout:.0f}s threshold)!")

            # Determine click coordinates
            target_hwnd = None
            if self.click_target is not None:
                click_x, click_y = self.click_target
            else:
                # Auto-locate BlueStacks / Reolink window on screen
                hwnd, rect = find_bluestacks_or_reolink_window()
                if hwnd and rect:
                    target_hwnd = hwnd
                    # Click slightly off-center (center of video viewport in BlueStacks)
                    click_x = (rect[0] + rect[2]) // 2
                    click_y = (rect[1] + rect[3]) // 2
                    print(f"[INFO] [Reolink Watchdog] Located BlueStacks window (HWND: {hwnd}) at {rect}. Target: ({click_x}, {click_y})")
                elif screen_pos is not None:
                    h, w = frame.shape[:2]
                    click_x = screen_pos[0] + w // 2
                    click_y = screen_pos[1] + h // 2
                else:
                    h, w = frame.shape[:2]
                    click_x = w // 2
                    click_y = h // 2

            print(f"[INFO] [Reolink Watchdog] Dispatching reactivation click #{self.freeze_count} at ({click_x}, {click_y})...")
            self._send_mouse_click(click_x, click_y, target_hwnd=target_hwnd)
            self.last_click_time = now
            self.last_change_time = now  # Reset timer to give stream time to buffer
            self.last_roi_gray = gray.copy()
            return True

        return False

    def _send_mouse_click(self, x: int, y: int, target_hwnd: Optional[int] = None):
        """Simulate a mouse click at screen coordinates (x, y) with window focus and Android touch duration."""
        if sys.platform == "win32":
            try:
                user32 = ctypes.windll.user32

                # Save current cursor position
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                orig_pt = POINT()
                user32.GetCursorPos(ctypes.byref(orig_pt))

                # Bring target window to foreground if known
                if target_hwnd:
                    user32.ShowWindow(target_hwnd, 5)  # SW_SHOW
                    user32.SetForegroundWindow(target_hwnd)
                    time.sleep(0.08)

                # Move to target position
                user32.SetCursorPos(int(x), int(y))
                time.sleep(0.06)

                # Send Left Button Down (0x0002)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                # Hold down for 120ms (crucial for Android/BlueStacks touch registration)
                time.sleep(0.12)
                # Send Left Button Up (0x0004)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                time.sleep(0.08)

                # Restore cursor position
                user32.SetCursorPos(orig_pt.x, orig_pt.y)
                print(f"[SUCCESS] [Reolink Watchdog] Click delivered to BlueStacks and cursor restored.")
            except Exception as e:
                print(f"[ERROR] [Reolink Watchdog] Win32 click simulation failed: {e}", file=sys.stderr)
        else:
            # Fallback for Android on-device / Linux / Headless ADB
            print(f"[INFO] [Reolink Watchdog] Sending Android tap at ({x}, {y}) via input tap / ADB.")
            import subprocess
            commands = [
                ["adb", "-s", "127.0.0.1:5555", "shell", "input", "tap", str(x), str(y)],
                ["/system/bin/input", "tap", str(x), str(y)],
                ["input", "tap", str(x), str(y)],
            ]
            for cmd in commands:
                try:
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    if res.returncode == 0:
                        print(f"[SUCCESS] [Reolink Watchdog] Android tap delivered at ({x}, {y}) via {' '.join(cmd[:2])}")
                        return
                except Exception:
                    continue

