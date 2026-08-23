# MotionDetector Architecture & Roadmap

## Headless Linux (Hetzner) Deployment Plan

### Architecture
1. **Containerized Android (ReDroid):**
   - Runs Android 11 in Docker on headless Linux without X11 GUI.
   - Hosts the Reolink APK with hardware acceleration or Mesa software rendering.
2. **Orchestrator (`src/orchestrator.py`):**
   - Controls the Reolink app via ADB (`adb shell am start`, tap coords).
   - Captures screen frames from ReDroid via ADB framebuffer or virtual display pipe.
3. **MotionDetector Engine (`src/main.py`):**
   - Runs in `--headless` mode.
   - Evaluates motion (MOG2), classifies (YOLOv8n: human, car, other), respects active zones (`--sections`), and sends real-time Telegram alerts with snapshots.
4. **Daemonization:**
   - Managed via `docker-compose.yml` or `systemd` service for 24/7 reliability.

## Milestones Status
- [x] **Milestone 1:** Core motion detection & YOLOv8 classification pipeline.
- [x] **Milestone 2:** Stream/screen capture, visual size calibration (`--calibrate`), and quadrant filtering (`--sections`).
- [x] **Milestone 3:** Async Telegram notifications with snapshot photos to `ReolinkMotionDetector`.
- [x] **Milestone 3.5 (Reolink Watchdog & Fixes):**
  - Reolink on-screen time-strip monitor (checks seconds ticking in upper HUD).
  - Stream freeze detection (>10s inactivity threshold).
  - BlueStacks auto-window handle & viewport targeting with foreground activation.
  - Android touch registration fix (120ms press-hold duration).
- [ ] **Milestone 4:** Headless ReDroid Docker deployment & ADB orchestrator on Hetzner.

## Component Reference

### Reolink Watchdog (`src/reolink_keeper.py`)
- **Time Strip Monitoring:** Continuously extracts the upper camera timestamp strip (`MM/DD/YYYY HH:MM:SS DOW`).
- **Freeze Detection:** Calculates pixel diff on the seconds digits. If unchanged for $\ge 10$ seconds (`--freeze-timeout`), triggers reactivation.
- **Window Targeting:** Automatically enumerates visible Windows handles matching `BlueStacks App Player` / `Reolink`, computes video center coordinates, brings window to foreground, and dispatches a 120ms touch click.
