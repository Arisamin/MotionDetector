# Running MotionDetector & Orchestrator on Android (BlueStacks / ReDroid)

This module allows running the complete Orchestrator and Motion Detection pipeline **natively inside Android**.

---

## 1. Prerequisites inside BlueStacks / Android
1. Install **Termux** (from F-Droid or GitHub Releases).
2. Open Termux on BlueStacks.

---

## 2. Setup (One-Time)
Run in Termux:
```bash
pkg update && pkg install git -y
git clone https://github.com/<your-repo>/MotionDetector.git
cd MotionDetector
bash android/install.sh
```

Ensure `config.json` is configured with your Telegram Bot Token and Chat ID:
```bash
cp config.example.json config.json
nano config.json
```

---

## 3. Execution
Launch the orchestrator directly from Termux:
```bash
bash android/run.sh
```

The orchestrator will:
1. Launch the Reolink app via Android's internal intent system.
2. Tap the camera stream card to activate live video.
3. Continuously capture screen frames using native `screencap`.
4. Run MOG2 + YOLOv8 object detection on-device.
5. Send instant Telegram notifications with annotated snapshot photos.
