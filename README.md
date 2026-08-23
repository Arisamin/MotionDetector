# MotionDetector

Autonomous motion detection, object classification (Human / Car / Other), and event logging agent for camera feeds and emulator apps.

## Features
- **Background Motion Detection:** MOG2 background subtractor with morphological filtering.
- **Lightweight AI Classifier:** YOLOv8 nano model classifying motion into **Human**, **Car/Vehicle**, or **Other**.
- **Screen Quadrant Filtering (`--sections`):** Monitor specific screen zones (`1`=TL, `2`=TR, `3`=BL, `4`=BR, e.g. `--sections 1_2`).
- **Visual Calibration (`--calibrate`):** Interactive resizable square to set resolution-independent motion size thresholds.
- **Telegram Notifications:** Instant async delivery of alert summaries with annotated snapshot photos.
- **Reolink Stream Watchdog (`--reolink`):** Auto-detects frozen video clock and dispatches simulated touch clicks to BlueStacks to keep streams live 24/7.

## Quick Start

### 1. Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Interactive Size Calibration (Optional)
```powershell
.\.venv\Scripts\python -m src.main --source screen --calibrate
```

### 3. Run Motion Detector with Reolink Watchdog & Telegram
```powershell
.\.venv\Scripts\python -m src.main --source screen --reolink --telegram
```

## CLI Reference
| Flag | Description | Default |
|---|---|---|
| `--source` | Screen (`screen`), webcam (`0`), RTSP URL, or video file path | `0` |
| `--reolink` | Enables clock freeze detection & auto-reactivation click | `False` |
| `--freeze-timeout` | Seconds of stopped clock before sending wakeup click | `10.0` |
| `--sections` / `--zones` | Screen sections to monitor (e.g. `1_2`, `4`) | `1_2_3_4` (All) |
| `--telegram` | Enable Telegram alert notifications and photos | Config value |
| `--calibrate` | Launch interactive square size calibration tool | - |
| `--headless` | Run in background without preview GUI window | `False` |
| `--cooldown` | Seconds between reported motion events | `1.0` |

## Project Structure
- `src/`: Core engine (`motion_detector.py`, `classifier.py`, `reolink_keeper.py`, `logger.py`, `notifier.py`, `calibrator.py`)
- `logs/`: Timestamped motion detection event logs (CSV/JSONL)
- `snapshots/`: Captured motion frame snapshots
- `docs/`: Architecture and roadmap specifications

