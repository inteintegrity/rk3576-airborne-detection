# Airborne Object Detection - RK3576 NPU (ai-lab package)
# Dockerfile for Seeed ai-lab platform

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RKNN Toolkit Lite2
COPY rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl .
RUN pip install --no-cache-dir rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# Copy RKNN runtime library
COPY lib/librknnrt.so /usr/lib/
RUN chmod 755 /usr/lib/librknnrt.so

# Copy application code
COPY . .

# Expose web service port
EXPOSE 8000

# Default: run web detection with VisDrone model
# - camera_id -1 = video analysis mode (no camera, web UI only)
CMD ["python", "web_detection.py", "--model_path", "model/best.rknn", "--camera_id", "-1"]