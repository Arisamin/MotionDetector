# MotionDetector — System Design & Technical Architecture

## 1. Executive Summary & Problem Statement

### The Problem
Commercial IP security cameras (like Reolink) provide live streaming through their mobile apps, but:
1. **Stream Timeout / Freeze**: Inactive streams pause automatically after a few minutes to conserve bandwidth, requiring physical or manual user interaction to resume.
2. **Proprietary Enclosures**: Direct RTSP/ONVIF streams might be locked or firewalled on cellular/cloud-only camera models.
3. **High False Alarms**: Built-in camera motion sensors often trigger on wind, bugs, and lighting changes without semantic object classification.

### The Solution
**MotionDetector** is an autonomous computer vision and orchestration agent that:
- Runs either on a host machine (Windows/Linux) or directly **on-device inside Android** (BlueStacks / ReDroid).
- Ingests video frames in real-time from the camera app.
- Detects motion using **MOG2 background subtraction**.
- Classifies moving entities into **Human**, **Vehicle (Car)**, or **Other** via **YOLOv8 deep learning**.
- Filters motion by user-defined **quadrants (sections 1–4)** and calibrated size thresholds.
- Monitors the camera's on-screen timestamp to detect stream freezes ($>10\text{s}$) and automatically dispatches simulated touch/click events to keep the stream alive 24/7.
- Sends instant **Telegram alerts with annotated snapshot images** via a non-blocking asynchronous queue.

---

## 2. High-Level Architecture Diagram

```
 +-----------------------------------------------------------------------------------+
 |                             VIDEO CAPTURE SOURCE                                  |
 |   (Android screencap / Screen MSS / OpenCV RTSP / Video Stream)                  |
 +------------------------------------------+----------------------------------------+
                                            |
                                            v (Raw BGR Frame)
 +-----------------------------------------------------------------------------------+
 |                         PARALLEL WATCHDOG & DETECTION PIPELINE                    |
 |                                                                                   |
 |   +---------------------------------------+   +-------------------------------+   |
 |   |         ReolinkWatchdog               |   |        MotionDetector         |   |
 |   | - Extracts HUD Time Strip (top 15%)   |   | - Top 8% HUD Mask (Anti-Noise)|   |
 |   | - Calculates pixel delta on seconds   |   | - MOG2 Background Subtractor  |   |
 |   | - If frozen >10s -> Auto Touch Click  |   | - Gaussian Blur + Dilation    |   |
 |   +---------------------------------------+   | - Min Area Filter (>2500 px)  |   |
 |                                               | - Quadrant Zoning (S1-S4)     |   |
 |                                               +---------------+---------------+   |
 +---------------------------------------------------------------|-------------------+
                                                                 v (Motion Bounding Boxes)
 +-----------------------------------------------------------------------------------+
 |                           SEMANTIC CLASSIFIER                                     |
 |   +---------------------------------------------------------------------------+   |
 |   |  ObjectClassifier (YOLOv8n / Lightweight Fallback)                        |   |
 |   |  - Infers on motion regions                                               |   |
 |   |  - Maps COCO labels -> 'human', 'car', 'other'                            |   |
 |   +-----------------------------------+---------------------------------------+   |
 +---------------------------------------|-------------------------------------------+
                                         v (Classified Detections)
 +-----------------------------------------------------------------------------------+
 |                        LOGGING & NOTIFICATION PIPELINE                            |
 |   +------------------------------------+   +----------------------------------+   |
 |   |          MotionLogger              |   |        TelegramNotifier          |   |
 |   |  - CSV & JSONL event logs          |   |  - Async Background Worker Queue |   |
 |   |  - Cooldown throttling (3.0s)      |   |  - Photo + Formatted HTML alert  |   |
 |   |  - Frame snapshot annotation       |   |  - Telegram Bot API              |   |
 |   +------------------------------------+   +----------------------------------+   |
 +-----------------------------------------------------------------------------------+
```

---

## 3. Technology Stack & Architectural Decisions (Interview Talking Points)

### Programming Language: **Python 3**
- **Why Python?**
  - **Rich Ecosystem**: Python is the lingua franca for computer vision, AI/ML (OpenCV, PyTorch, YOLO), and system automation.
  - **Cross-Platform Compatibility**: The identical codebase runs seamlessly across Windows, Linux servers (Docker/ReDroid), and native Android environments (Termux).
  - **Rapid Prototyping & Extensibility**: High-level abstractions allow clean separation of concerns across modules (`capture`, `detector`, `classifier`, `notifier`, `orchestrator`).

---

### Core Libraries & Tools

| Library / Tool | Role in Project | Why This Specific Tool Was Chosen |
| :--- | :--- | :--- |
| **OpenCV (`cv2`)** | Core Computer Vision | Industry standard C++ backend with Python bindings. Provides hardware-accelerated `createBackgroundSubtractorMOG2`, morphological filters (dilation/erosion), contour finding, and image transformation with sub-millisecond execution. |
| **Ultralytics YOLOv8n** | Deep Learning Classifier | State-of-the-art real-time object detector. The **Nano (n)** variant provides high accuracy with a tiny footprint (~6MB weights, <30ms CPU inference), making it ideal for edge devices and servers without dedicated GPUs. |
| **NumPy (`numpy`)** | Vectorized Array Computing | OpenCV frames are represented as multi-dimensional NumPy arrays (`H x W x C`), enabling fast zero-copy slicing, mask zeroing, and pixel difference computations in memory. |
| **Pillow (`PIL`)** | Image Handling | High-performance image encoding, manipulation, and format conversion. |
| **Requests + Threading Queue** | Asynchronous Telegram Delivery | HTTP requests over public internet can suffer 100–500ms network latency. Using a background **Producer-Consumer Queue (`queue.Queue`)** with a dedicated worker thread ensures network I/O **never blocks** the video capture loop. |
| **ADB (`android-tools`)** | Android System Orchestration | Android Debug Bridge provides low-level control for automated app launching (`am start`), touch event dispatch (`input tap`), and raw framebuffer streaming (`screencap`). |
| **Termux** | On-Device Linux Environment | Lightweight POSIX environment running directly inside Android OS, providing Python runtime, package management (`pkg`/`apt`), and hardware access. |

---

## 4. Key Engineering Challenges & Solutions

### Challenge 1: The Reolink Inactivity Stream Freeze
- **Problem**: The Reolink mobile app pauses video streaming after a few minutes of inactivity.
- **Solution**: Built `ReolinkWatchdog` (`src/reolink_keeper.py`):
  - Continuously crops the upper 15% HUD timestamp strip (`MM/DD/YYYY HH:MM:SS`).
  - Computes the absolute pixel difference between successive frames (`cv2.absdiff`).
  - When the seconds stop ticking for $>10\text{s}$, the watchdog dispatches a targeted touch click (`input tap 800 450`) to reactivate the stream.

### Challenge 2: Clock Ticks Triggering False Motion Alarms
- **Problem**: When seconds change on the on-screen clock, the pixel change would trigger motion detection every single second.
- **Solution**:
  1. **Dual-Zone Separation**: The watchdog inspects the unmasked frame for clock changes, while the motion detector applies a `mask_top_percent=0.08` mask over the HUD.
  2. **Calibrated Motion Threshold**: Measured single-digit HUD size ($24\times35\text{ px} \approx 840\text{ px}^2$). Set minimum contour area threshold to **$2500\text{ px}$** (> $2\times$ digit size), ensuring text changes are completely ignored.

### Challenge 3: 32-bit Architecture Source-Compilation Lockup
- **Problem**: On 32-bit `x86/i686` Android instances (BlueStacks Nougat 32), running `pip install torch ultralytics` triggered full C++ source compilation via `g++`, freezing the system.
- **Solution**:
  - Engineered a **Lightweight Fallback Architecture**: Replaced heavy source builds with pre-compiled Termux binary packages (`tur-repo`, `opencv-python`, `python-numpy`).
  - Made YOLO optional: On resource-constrained edge devices, `ObjectClassifier` gracefully falls back to fast MOG2 motion bounding boxes without crashing.

### Challenge 4: Zero-Disk Frame Ingestion
- **Problem**: Writing temporary screenshot files to disk for OpenCV consumption produces excessive flash storage wear and severe I/O latency.
- **Solution**: Engineered `AndroidNativeCaptureSource` (`src/android_capture.py`) using in-memory stream pipes:
  - Executes `adb exec-out screencap -p` directly to standard output.
  - Decodes raw PNG bytes in memory via `cv2.imdecode(np.frombuffer(stdout), cv2.IMREAD_COLOR)` in <15ms.

---

## 5. Next Steps According to the Master Plan

### Milestone 4: Headless Linux Server Deployment (Hetzner Cloud)
1. **Dockerized ReDroid Setup**:
   - Deploy **ReDroid** (Android 11 in Docker) on a headless Ubuntu Hetzner server.
   - Run ReDroid with software GPU rendering (`Mesa / SwiftShader`) or GPU passthrough.
2. **Container Orchestrator**:
   - Create `docker-compose.yml` linking the ReDroid Android container with the Python MotionDetector service.
3. **Automated ADB Pairing**:
   - ReDroid exposes ADB on port 5555 within Docker internal network.
   - The host Python daemon connects to `localhost:5555`, installs/starts Reolink, and monitors 24/7.
4. **Daemonization & System Health**:
   - Setup `systemd` auto-restart service and watchdog health monitoring.
