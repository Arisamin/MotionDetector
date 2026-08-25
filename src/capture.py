"""Frame capture source abstraction supporting video files, RTSP/camera streams, and screen capture."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np


class BaseCaptureSource(ABC):
    """Abstract base class for video/screen frame sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame_bgr)."""
        pass

    @abstractmethod
    def release(self):
        """Release any open resources or handles."""
        pass


class OpenCVStreamSource(BaseCaptureSource):
    """Captures frames from a video file, webcam index, or RTSP/HTTP URL via OpenCV."""

    def __init__(self, source: str):
        # If source is an integer string, convert to int for webcam
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open OpenCV video source: {source}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap:
            self.cap.release()


class ScreenCaptureSource(BaseCaptureSource):
    """Captures frames from a monitor region or full screen using mss."""

    def __init__(self, monitor_idx: int = 1, region: Optional[dict] = None):
        """
        :param monitor_idx: Monitor index (1 = primary monitor, 0 = all monitors).
        :param region: Optional dict with {'top', 'left', 'width', 'height'}.
        """
        import mss
        self.sct = mss.mss()
        if region is not None:
            self.monitor = region
        else:
            monitors = self.sct.monitors
            if monitor_idx < len(monitors):
                self.monitor = monitors[monitor_idx]
            else:
                self.monitor = monitors[1] if len(monitors) > 1 else monitors[0]

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            sct_img = self.sct.grab(self.monitor)
            # Convert BGRA to BGR
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception:
            return False, None

    def release(self):
        if self.sct:
            self.sct.close()


def create_capture_source(source_str: str) -> BaseCaptureSource:
    """Factory helper to instantiate the appropriate capture source."""
    s_lower = source_str.lower().strip()
    if s_lower.startswith("screen"):
        # e.g., "screen" or "screen:1"
        parts = source_str.split(":")
        idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        return ScreenCaptureSource(monitor_idx=idx)
    elif s_lower == "android":
        from src.android_capture import AndroidNativeCaptureSource
        return AndroidNativeCaptureSource()
    return OpenCVStreamSource(source_str)
