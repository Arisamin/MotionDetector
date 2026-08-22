# MotionDetector

Autonomous motion detection, object classification (Human / Car / Other), and event logging agent for camera apps running locally or on headless server environments.

## Features
- Background motion detection (MOG2 / Frame differencing)
- Real-time object classification with lightweight AI (YOLOv8n)
- Headless stream capture (Media player / RTSP / Screen capture / Android emulator)
- Event logging (JSON/CSV) with snapshot recording

## Project Structure
- `src/`: Motion detection, capture, and classification engine
- `logs/`: Timestamped motion detection event logs
- `snapshots/`: Captured motion frame snapshots
