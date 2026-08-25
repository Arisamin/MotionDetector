#!/data/data/com.termux/files/usr/bin/bash
echo "=== 1. Updating Termux package list ==="
pkg update -y
echo "=== 2. Enabling TUR repo for pre-compiled python-opencv ==="
pkg install -y tur-repo
echo "=== 3. Installing pre-compiled packages ==="
pkg install -y python python-numpy python-pillow python-requests python-opencv
echo "=== Setup complete! ==="