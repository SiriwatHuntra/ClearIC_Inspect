# Installation Guide — ClearIC Inspect

## Prerequisites

Raspberry Pi OS (64-bit, Bookworm or later) with internet access.

---

## 1. System Packages (apt)

Install native dependencies that are best sourced from the system package manager.
These will be shared into the venv via `--system-site-packages`.

```bash
sudo apt update && sudo apt install -y \
    python3-pyqt5 \
    python3-rpi.gpio \
    python3-gpiozero \
    python3-numpy \
    python3-opencv \
    python3-venv
```

> `python3-opencv` and `python3-pyqt5` carry heavy native dependencies — installing them via apt
> avoids compilation from source and is faster and more reliable on the Pi.

---

## 2. Basler Pylon SDK

`pypylon` (the Python binding) requires the Pylon runtime to be installed first.

1. Download the Pylon Camera Software Suite for **Linux ARM 64-bit** from the Basler website.
2. Install the runtime:

```bash
# Unpack and run the installer (filename will differ by version)
tar -xzf pylon_*.tar.gz
sudo tar -C /opt -xzf pylon_*_aarch64.tar.gz
```

3. Add Pylon to the library path (add to `~/.bashrc` to persist):

```bash
export PYLON_ROOT=/opt/pylon
export LD_LIBRARY_PATH=$PYLON_ROOT/lib:$LD_LIBRARY_PATH
```

4. Apply immediately:

```bash
source ~/.bashrc
```

---

## 3. Create the Virtual Environment

Create a venv that inherits the system-site packages installed in step 1:

```bash
cd /home/pi/ClearIC_Inspect
python3 -m venv --system-site-packages .venv
```

Activate it:

```bash
source .venv/bin/activate
```

> Verify system packages are visible: `python -c "import cv2, PyQt5, RPi.GPIO; print('OK')`

---

## 4. Install pip Packages

With the venv active, install the remaining packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Verify Installation

```bash
python -c "
import cv2
import PyQt5
import RPi.GPIO
import openvino
import pypylon.pylon
print('All imports OK')
"
```

---

## Running the Application

```bash
source .venv/bin/activate
python main.py
```

---

## Deactivate / Reactivate

```bash
# Deactivate
deactivate

# Reactivate next time
source /home/pi/ClearIC_Inspect/.venv/bin/activate
```
