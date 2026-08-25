"""Native Android screen capture source for running inside Termux / Android environment."""
import subprocess
import cv2
import numpy as np
from typing import Optional, Tuple
from src.capture import BaseCaptureSource


class AndroidNativeCaptureSource(BaseCaptureSource):
    """Captures frames directly from Android framebuffer using native screencap or ADB."""

    def __init__(self, downscale_factor: float = 1.0, adb_device: str = "127.0.0.1:5555"):
        self.downscale_factor = downscale_factor
        self.adb_device = adb_device
        # Test connecting ADB once
        try:
            subprocess.run(["adb", "connect", self.adb_device], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        data = None

        # 1. Try native screencap binary
        for screencap_bin in ["screencap", "/system/bin/screencap"]:
            try:
                proc = subprocess.run(
                    [screencap_bin, "-p"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout and len(proc.stdout) > 100:
                    data = proc.stdout
                    break
            except Exception:
                pass

        # 2. Fallback to local ADB exec-out
        if not data:
            for dev in [self.adb_device, "127.0.0.1:5555", "emulator-5554"]:
                try:
                    proc = subprocess.run(
                        ["adb", "-s", dev, "exec-out", "screencap", "-p"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if proc.returncode == 0 and proc.stdout and len(proc.stdout) > 100:
                        data = proc.stdout
                        break
                except Exception:
                    pass

        if not data:
            return False, None

        try:
            # Decode PNG bytes directly in memory
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                return False, None

            if self.downscale_factor != 1.0:
                h, w = frame.shape[:2]
                new_w = int(w * self.downscale_factor)
                new_h = int(h * self.downscale_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return True, frame
        except Exception as e:
            print(f"[ERROR in AndroidNativeCaptureSource decode]: {e}")
            return False, None

    def release(self):
        pass
