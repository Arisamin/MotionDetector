#!/data/data/com.termux/files/usr/bin/bash
echo "=== Syncing latest code from /sdcard/MotionDetector ==="
cp -r /sdcard/MotionDetector/src/* ~/MotionDetector/src/ 2>/dev/null || true
cp /sdcard/MotionDetector/config.json ~/MotionDetector/ 2>/dev/null || true
DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"
cd "$DIR"
echo "=== Starting MotionDetector Android Orchestrator ==="
python3 src/android_orchestrator.py "$@" > /sdcard/MotionDetector/orchestrator.log 2>&1