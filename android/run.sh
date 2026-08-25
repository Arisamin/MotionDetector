#!/data/data/com.termux/files/usr/bin/bash
DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"
cd "$DIR"
echo "=== Starting MotionDetector Android Orchestrator ==="
python3 src/android_orchestrator.py "$@"