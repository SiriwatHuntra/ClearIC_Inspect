#!/usr/bin/env bash
# One-time environment setup for ClearIC_Inspect — see requirements.txt / CLAUDE.md.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

echo "=== ClearIC_Inspect setup ==="

echo "--- apt packages (PyQt5 / GPIO / numpy / opencv) ---"
sudo apt update
sudo apt install -y \
    python3-pyqt5 \
    python3-rpi.gpio \
    python3-numpy \
    python3-opencv

echo "--- Basler Pylon SDK check ---"
if ! dpkg -l 2>/dev/null | grep -qi pylon; then
    echo "WARNING: Basler Pylon SDK not detected."
    echo "  Download the arm64/armhf .deb from https://www.baslerweb.com/en/downloads/software-downloads/"
    echo "  then install it with: sudo dpkg -i pylon_*.deb"
    echo "  (pypylon will fail to import without it)"
fi

echo "--- Python venv (.venv, system-site-packages) ---"
if [ ! -d .venv ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install openvino pypylon

echo
echo "=== Setup complete ==="
echo "Run manually with:"
echo "  source .venv/bin/activate && python CLearIC.py"
echo "Or use the desktop shortcut (WSOF_insp.desktop -> run.sh)."
