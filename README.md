# RK3576 Airborne Object Detection

VisDrone YOLOv8s based airborne object detection system running on RK3576 NPU.

## Overview

Real-time drone-view object detection using YOLOv8s (VisDrone fine-tuned) with RKNN NPU acceleration on RK3576. Detects 11 VisDrone classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others.

## Directory Structure

```
├── airborne_detection_rknn.py   # Main detection script (NPU)
├── rknn_detector.py              # RKNN inference engine
├── multi_scene_demo.py           # 4-video real-time grid demo
├── web_detection.py              # FastAPI web service (ai-lab)
├── Dockerfile                    # Docker for ai-lab deployment
├── requirements.txt              # Web service dependencies
├── requirements_rknn.txt         # NPU detection dependencies
├── model/
│   └── best.rknn                 # VisDrone YOLOv8s RKNN model
├── lib/
│   └── librknnrt.so              # RKNN runtime library
├── py_utils/
│   ├── __init__.py
│   └── coco_utils.py             # Letterbox & coordinate utilities
├── rknn-toolkit-lite2-packages/
│   └── rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
└── video/
    ├── test1.mp4
    ├── test2.mp4
    ├── test3.mp4
    └── test4.mp4
```

## VisDrone Classes (11)

| ID | Class | ID | Class |
|----|-------|-----|--------|
| 0 | pedestrian | 6 | tricycle |
| 1 | people | 7 | awning-tricycle |
| 2 | bicycle | 8 | bus |
| 3 | car | 9 | motor |
| 4 | van | 10 | others |
| 5 | truck | | |

## Quick Start (Direct on RK3576)

### 1. Install runtime library

```bash
sudo cp lib/librknnrt.so /usr/lib/
sudo ldconfig
```

### 2. Install Python dependencies

```bash
pip install opencv-python numpy
pip install rknn-toolkit-lite2-packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 3. Run detection

```bash
# Basic detection
python airborne_detection_rknn.py --model model/best.rknn --video video/test1.mp4

# Headless mode (no display)
python airborne_detection_rknn.py --no-display

# Filter vehicle classes only (car, van, truck, bus)
python airborne_detection_rknn.py --classes 3,4,5,8

# Adjust confidence threshold
python airborne_detection_rknn.py --conf 0.15

# Multi-scene demo (4 videos in 2x2 grid)
python multi_scene_demo.py \
  --videos video/test1.mp4 video/test2.mp4 video/test3.mp4 video/test4.mp4 \
  --names "Scene1" "Scene2" "Scene3" "Scene4"
```

## Docker Deployment (ai-lab)

### Build

```bash
docker build -t airborne-detection-rk3576 .
```

### Run

```bash
# Video analysis mode (web UI)
docker run -d -p 8000:8000 airborne-detection-rk3576

# With camera
docker run -d -p 8000:8000 --device /dev/video1 airborne-detection-rk3576 \
  python web_detection.py --model_path model/best.rknn --camera_id 1
```

Access web UI at `http://<RK3576_IP>:8000`

## Model Conversion (Optional)

If you need to regenerate the RKNN model from PyTorch:

```
best.pt → best.onnx → best.rknn
  ultralytics    rknn-toolkit2
  (PC)           (PC/WSL2 Linux)
```

### Step 1: pt → onnx (PC)

```bash
pip install ultralytics onnx onnxruntime onnxslim
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=640, simplify=True)"
```

### Step 2: onnx → rknn (WSL2 Linux)

```bash
pip install rknn-toolkit2 onnx==1.14.1
```

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3576')
rknn.load_onnx('best.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('best.rknn')
rknn.release()
```

> **Note**: onnx must be 1.14.1 (not latest). rknn-toolkit2 2.3.2 uses `onnx.mapping` which was removed in onnx 1.15+.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `model/best.rknn` | RKNN model path |
| `CONF_THRESHOLD` | 0.25 | Confidence threshold |
| `IOU_THRESHOLD` | 0.45 | NMS IoU threshold |
| `IMG_SIZE` | 640 | Inference image size |
| `NPU_CORE` | 3 | NPU core (0/1/2/3=auto) |

## Tech Stack

- **Model**: YOLOv8s (VisDrone fine-tuned, from `dronefreak/visdrone-yolov8s`)
- **NPU Inference**: RKNN Toolkit Lite2 v2.3.2
- **Runtime**: librknnrt.so v2.3.2
- **Image Processing**: OpenCV
- **Web Service**: FastAPI + Uvicorn
- **Platform**: RK3576 (Rockchip NPU, 6 TOPS)

## License

MIT License
