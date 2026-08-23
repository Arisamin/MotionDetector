"""Autonomous Orchestrator for MotionDetector.

Manages the complete lifecycle:
1. Launches the emulator (BlueStacks on Windows / ReDroid on Linux).
2. Connects and launches the Reolink app via ADB/Win32.
3. Taps into the camera live stream.
4. Runs the MotionDetector pipeline with clock freeze watchdog and Telegram alerts.
"""
import os
import sys
import time
import subprocess
import argparse
from typing import Optional, Tuple, Dict, Any

from src.config import load_config
from src.main import run_pipeline, parse_args
from src.reolink_keeper import find_bluestacks_or_reolink_window


BLUESTACKS_DEFAULT_PATHS = [
    r"C:\Program Files\BlueStacks_nxt\HD-Player.exe",
    r"C:\Program Files (x86)\BlueStacks_nxt\HD-Player.exe",
]

BLUESTACKS_ADB_PATHS = [
    r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
    r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
]

REOLINK_PACKAGE = "com.mcu.reolink"


def get_bluestacks_instance_name() -> str:
    """Read the active BlueStacks instance name from bluestacks.conf."""
    conf_path = r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf"
    if os.path.exists(conf_path):
        try:
            with open(conf_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "bst.instance." in line and (".status.adb_port" in line or ".adb_port" in line):
                        parts = line.split(".")
                        if len(parts) >= 3:
                            return parts[2]
        except Exception:
            pass
    return "Nougat32"


class Orchestrator:
    """Orchestrates emulator lifecycle, Reolink app automation, and MotionDetector monitoring."""

    def __init__(
        self,
        mode: str = "auto",
        bluestacks_path: Optional[str] = None,
        adb_path: Optional[str] = None,
        adb_target: str = "127.0.0.1:5555",
        camera_tap_coords: Tuple[int, int] = (640, 360),
    ):
        """
        :param mode: 'local' (BlueStacks/Windows), 'redroid' (Docker/Linux), or 'auto'.
        :param bluestacks_path: Explicit path to HD-Player.exe if not in default location.
        :param adb_path: Explicit path to adb/HD-Adb executable.
        :param adb_target: ADB device address (e.g. 127.0.0.1:5555 or emulator-5554).
        :param camera_tap_coords: Coordinates to tap the camera card in the Reolink app.
        """
        if mode == "auto":
            self.mode = "local" if sys.platform == "win32" else "redroid"
        else:
            self.mode = mode

        self.bluestacks_path = bluestacks_path or self._find_bluestacks_exe()
        self.adb_path = adb_path or self._find_adb_exe()
        self.adb_target = adb_target
        self.camera_tap_coords = camera_tap_coords

    def _find_bluestacks_exe(self) -> Optional[str]:
        for p in BLUESTACKS_DEFAULT_PATHS:
            if os.path.exists(p):
                return p
        return None

    def _find_adb_exe(self) -> Optional[str]:
        for p in BLUESTACKS_ADB_PATHS:
            if os.path.exists(p):
                return p
        return "adb"

    def is_emulator_running(self) -> bool:
        """Check if BlueStacks window or ReDroid container is active."""
        if self.mode == "local":
            hwnd, _ = find_bluestacks_or_reolink_window()
            return hwnd is not None
        else:
            # ReDroid docker check
            try:
                res = subprocess.run(
                    ["docker", "ps", "-q", "-f", "name=redroid"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return bool(res.stdout.strip())
            except Exception:
                return False

    def launch_emulator(self, wait_seconds: int = 15) -> bool:
        """Start the emulator if it is not already running."""
        if self.is_emulator_running():
            print(f"[INFO] [Orchestrator] Emulator ({self.mode}) is already running.")
            return True

        print(f"[INFO] [Orchestrator] Starting emulator in {self.mode} mode...")
        if self.mode == "local":
            if not self.bluestacks_path or not os.path.exists(self.bluestacks_path):
                print(f"[ERROR] [Orchestrator] BlueStacks executable not found at {self.bluestacks_path}", file=sys.stderr)
                return False

            # Launch BlueStacks
            subprocess.Popen([self.bluestacks_path], shell=False)
            print(f"[INFO] [Orchestrator] Launched BlueStacks ({self.bluestacks_path}). Waiting {wait_seconds}s for boot...")

            # Poll for window appearance
            start = time.time()
            while time.time() - start < wait_seconds:
                time.sleep(2.0)
                if self.is_emulator_running():
                    print("[SUCCESS] [Orchestrator] BlueStacks window detected!")
                    time.sleep(3.0)  # Brief settling time
                    return True
            return False

        else:
            # Launch ReDroid container
            try:
                print("[INFO] [Orchestrator] Starting ReDroid Docker container...")
                subprocess.run(["docker", "start", "redroid"], check=True)
                time.sleep(5.0)
                return True
            except Exception as e:
                print(f"[ERROR] [Orchestrator] Failed to start ReDroid container: {e}", file=sys.stderr)
                return False

    def start_reolink_app(self) -> bool:
        """Launch the Reolink app and open the camera stream."""
        print("[INFO] [Orchestrator] Launching Reolink app...")

        launched = False
        if self.mode == "local" and self.bluestacks_path and os.path.exists(self.bluestacks_path):
            launched = self._launch_via_bluestacks_cmd()

        if not launched:
            launched = self._launch_via_adb()

        if launched:
            print("[SUCCESS] [Orchestrator] Reolink app launch command dispatched. Waiting 5s for camera list...")
            time.sleep(5.0)

        # Focus window and tap camera card to start live stream
        if self.mode == "local":
            hwnd, rect = find_bluestacks_or_reolink_window()
            if hwnd and rect:
                print(f"[INFO] [Orchestrator] BlueStacks window active at {rect}. Activating camera stream...")
                import ctypes
                user32 = ctypes.windll.user32
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
                user32.SetForegroundWindow(hwnd)
                time.sleep(1.0)
                # Tap center of the camera card inside BlueStacks
                tap_x = (rect[0] + rect[2]) // 2
                tap_y = (rect[1] + rect[3]) // 2
                self._send_local_tap(tap_x, tap_y)
                time.sleep(2.0)
                return True
        else:
            self._tap_camera_stream_adb()
            return True

        return launched

    def _launch_via_bluestacks_cmd(self) -> bool:
        """Launch Reolink using BlueStacks HD-Player command line."""
        instance_name = get_bluestacks_instance_name()
        print(f"[INFO] [Orchestrator] Launching {REOLINK_PACKAGE} via HD-Player on instance '{instance_name}'...")
        try:
            cmd = [
                self.bluestacks_path,
                "--instance",
                instance_name,
                "--cmd",
                "launchApp",
                "--package",
                REOLINK_PACKAGE,
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            return res.returncode == 0
        except Exception as e:
            print(f"[WARN] [Orchestrator] HD-Player app launch failed: {e}", file=sys.stderr)
            return False

    def _send_local_tap(self, x: int, y: int):
        """Simulate a click/tap on the camera stream card."""
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.SetCursorPos(int(x), int(y))
                time.sleep(0.06)
                user32.mouse_event(0x0002, 0, 0, 0, 0)
                time.sleep(0.12)
                user32.mouse_event(0x0004, 0, 0, 0, 0)
                print(f"[SUCCESS] [Orchestrator] Tapped camera card at ({x}, {y}).")
            except Exception:
                pass

    def _launch_via_adb(self) -> bool:
        if not self.adb_path:
            return False
        try:
            # Connect
            subprocess.run([self.adb_path, "connect", self.adb_target], capture_output=True, timeout=5)
            # Launch activity via monkey launcher
            cmd = [
                self.adb_path,
                "-s",
                self.adb_target,
                "shell",
                "monkey",
                "-p",
                REOLINK_PACKAGE,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return res.returncode == 0
        except Exception:
            return False

    def _tap_camera_stream_adb(self):
        """Tap on the camera preview tile to open full live view."""
        if not self.adb_path:
            return
        x, y = self.camera_tap_coords
        print(f"[INFO] [Orchestrator] Tapping camera stream tile at ({x}, {y}) via ADB...")
        try:
            subprocess.run(
                [self.adb_path, "-s", self.adb_target, "shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def run(self, pipeline_args):
        """Execute the full autonomous orchestration flow."""
        print("=" * 60)
        print("       Autonomous MotionDetector Orchestrator")
        print(f" Environment: {self.mode.upper()}")
        print("=" * 60)

        # Step 1: Ensure Emulator is running
        if not self.launch_emulator():
            print("[WARN] [Orchestrator] Could not confirm emulator status. Proceeding with detection.")

        # Step 2: Ensure Reolink app is active
        self.start_reolink_app()

        # Step 3: Run the MotionDetector pipeline with watchdog & Telegram alerts
        print("\n[INFO] [Orchestrator] Starting MotionDetector monitoring pipeline...")
        # Ensure reolink mode is enabled in pipeline args
        pipeline_args.reolink = True
        return run_pipeline(pipeline_args)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Orchestrator for BlueStacks/ReDroid & MotionDetector.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=["local", "redroid", "auto"],
        default="auto",
        help="Orchestration mode: 'local' (BlueStacks/Windows) or 'redroid' (Docker/Linux). Default: auto.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="screen",
        help="Video capture source for MotionDetector (default: screen).",
    )
    parser.add_argument(
        "--sections",
        "--zones",
        type=str,
        default=None,
        dest="sections",
        help="Underscore-separated screen sections to monitor: 1=TL, 2=TR, 3=BL, 4=BR.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run MotionDetector in headless mode without preview GUI.",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        default=True,
        help="Enable Telegram alert notifications (default: True).",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_false",
        dest="telegram",
        help="Disable Telegram alert notifications.",
    )
    parser.add_argument(
        "--freeze-timeout",
        type=float,
        default=10.0,
        help="Seconds of frozen clock before auto-reactivating stream (default: 10.0).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between motion events (default: 1.0).",
    )
    parser.add_argument(
        "--min-area",
        type=str,
        default=None,
        help="Minimum motion area in pixels or percentage (e.g. '1.0%%').",
    )

    args = parser.parse_args()

    # Pass full parsed args to main pipeline with clean defaults
    pipeline_args = parse_args(args_list=[])
    pipeline_args.source = args.source
    if args.sections is not None:
        pipeline_args.sections = args.sections
    pipeline_args.headless = args.headless
    pipeline_args.telegram = args.telegram
    pipeline_args.freeze_timeout = args.freeze_timeout
    pipeline_args.cooldown = args.cooldown
    if args.min_area is not None:
        pipeline_args.min_area = args.min_area

    orchestrator = Orchestrator(mode=args.mode)
    sys.exit(orchestrator.run(pipeline_args))


if __name__ == "__main__":
    main()
