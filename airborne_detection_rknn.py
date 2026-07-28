"""
Airborne Object Detection (RK3576 NPU)
======================================
RKNN NPU-accelerated airborne object detection system for RK3576.

Features:
  - RKNN NPU accelerated inference (YOLOv8 VisDrone fine-tuned model)
  - Real-time video detection and visualization
  - Per-class object counting
  - FPS performance monitoring
  - Output video saving

Prerequisites:
  1. best.pt -> ONNX -> RKNN (model conversion done by user)
  2. rknn-toolkit-lite2 installed
  3. Video file uploaded to device

Author: ZCode AI Assistant
Date: 2026-07-27
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import time
import sys

# Import RKNN detector
from rknn_detector import RKNNDetector, VISDRONE_CLASSES, CLASS_COLORS


# ============================================================
# Configuration
# ============================================================

class Config:
    """Global configuration"""

    # --- Model ---
    MODEL_PATH: str = "best.rknn"           # RKNN model path
    IMG_SIZE: int = 640                      # Inference image size
    CONF_THRESHOLD: float = 0.25            # Confidence threshold
    IOU_THRESHOLD: float = 0.45             # NMS IoU threshold
    NUM_CLASSES: int = 11                   # VisDrone 11 classes
    NPU_CORE: int = 3                       # NPU core (0/1/2/3=auto)

    # --- Video ---
    VIDEO_PATH: str = "video/video5.mp4"    # Input video path
    OUTPUT_DIR: str = "output"              # Output directory
    SAVE_OUTPUT: bool = True                # Whether to save output video
    SHOW_DISPLAY: bool = True               # Whether to show real-time display

    # --- Visualization ---
    LINE_THICKNESS: int = 2
    FONT_SCALE: float = 0.5

    # --- Class filter (empty list = no filter) ---
    # VisDrone: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
    #           5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
    CLASS_FILTER: list = []


# ============================================================
# Utility Functions
# ============================================================

def get_video_files(video_path: str) -> list:
    """Get list of video files"""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {video_path}")

    if path.is_file():
        return [str(path)]

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    videos = sorted([
        str(p) for p in path.rglob("*")
        if p.suffix.lower() in video_exts
    ])
    if not videos:
        raise FileNotFoundError(f"No video files found in: {video_path}")
    return videos


def draw_text_with_bg(img, text, pos, font_scale=0.5, thickness=1,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - baseline - 2), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


# ============================================================
# Video Processor
# ============================================================

class VideoProcessor:
    """Video detection processor"""

    def __init__(self, detector: RKNNDetector, config: Config):
        self.detector = detector
        self.cfg = config

        # Statistics
        self.frame_count = 0
        self.total_time = 0.0
        self.class_counts = defaultdict(int)

    def process_video(self, video_path: str):
        """Process a single video"""
        print(f"\n{'='*60}")
        print(f"[INFO] Processing video: {video_path}")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")

        # Output video
        writer = None
        if self.cfg.SAVE_OUTPUT:
            out_dir = Path(self.cfg.OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"{Path(video_path).stem}_rknn_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            print(f"[INFO] Output video: {out_path}")

        # Reset statistics
        self.frame_count = 0
        self.total_time = 0.0
        self.class_counts.clear()

        # Process frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Detection
            t_start = time.perf_counter()
            detections = self.detector.detect(frame)
            t_end = time.perf_counter()
            self.total_time += (t_end - t_start)

            # Class filter
            if self.cfg.CLASS_FILTER:
                detections = [d for d in detections
                             if d["class_id"] in self.cfg.CLASS_FILTER]

            # Draw
            annotated = self.detector.draw_detections(frame, detections)

            # Statistics
            for det in detections:
                self.class_counts[det["class_id"]] += 1

            # Status bar
            self._draw_stats(annotated, width, height)

            # Display
            if self.cfg.SHOW_DISPLAY:
                display = annotated
                if width > 1280:
                    scale = 1280 / width
                    display = cv2.resize(annotated, (int(width * scale), int(height * scale)))
                cv2.imshow("RKNN Airborne Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[INFO] Interrupted by user")
                    break
                elif key == ord("p"):
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord("p") or k2 == 27:
                            break

            # Save
            if writer:
                writer.write(annotated)

            # Progress
            if self.frame_count % 50 == 0:
                elapsed = self.total_time
                avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {self.frame_count}/{total_frames}, "
                      f"Avg FPS: {avg_fps:.1f}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Summary
        self._print_summary()

    def _draw_stats(self, frame, width, height):
        """Draw statistics overlay"""
        # Semi-transparent status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 70), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        elapsed = self.total_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Line 1: FPS and frame count
        draw_text_with_bg(
            frame,
            f"RKNN NPU | Frame: {self.frame_count} | FPS: {avg_fps:.1f} | "
            f"Detections: {sum(self.class_counts.values())}",
            (10, 22), font_scale=0.5, bg_color=(40, 40, 40)
        )

        # Line 2: Top-5 classes
        if self.class_counts:
            top5 = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            parts = []
            for cls_id, cnt in top5:
                name = VISDRONE_CLASSES.get(cls_id, f"cls_{cls_id}")
                parts.append(f"{name}:{cnt}")
            draw_text_with_bg(
                frame, " | ".join(parts),
                (10, 48), font_scale=0.4, bg_color=(40, 40, 40)
            )

    def _print_summary(self):
        """Print summary report"""
        print(f"\n{'='*60}")
        print(f"[SUMMARY] RKNN NPU inference complete")
        print(f"{'='*60}")
        print(f"  Total frames:    {self.frame_count}")
        print(f"  Total time:      {self.total_time:.2f}s")
        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
        print(f"  Average FPS:     {avg_fps:.1f}")
        print(f"  Total detections:{sum(self.class_counts.values())}")
        print(f"  Detection stats:")
        if self.class_counts:
            for cls_id, count in sorted(self.class_counts.items(),
                                        key=lambda x: x[1], reverse=True):
                name = VISDRONE_CLASSES.get(cls_id, f"cls_{cls_id}")
                print(f"    {name:<20s}: {count:>6d}")
        print(f"{'='*60}\n")


# ============================================================
# CLI Interface
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Airborne Object Detection (RK3576 NPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run
  python airborne_detection_rknn.py

  # Specify video
  python airborne_detection_rknn.py --video path/to/video.mp4

  # Specify RKNN model
  python airborne_detection_rknn.py --model best.rknn

  # Headless mode (no display)
  python airborne_detection_rknn.py --no-display

  # Detect vehicles only
  python airborne_detection_rknn.py --classes 3,4,5,8
        """,
    )
    parser.add_argument("--model", "-m", type=str, default=Config.MODEL_PATH,
                        help=f"RKNN model path (default: {Config.MODEL_PATH})")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Video path (default: Config.VIDEO_PATH)")
    parser.add_argument("--conf", type=float, default=Config.CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {Config.CONF_THRESHOLD})")
    parser.add_argument("--iou", type=float, default=Config.IOU_THRESHOLD,
                        help=f"NMS IoU threshold (default: {Config.IOU_THRESHOLD})")
    parser.add_argument("--imgsz", type=int, default=Config.IMG_SIZE,
                        help=f"Inference image size (default: {Config.IMG_SIZE})")
    parser.add_argument("--npu-core", type=int, default=Config.NPU_CORE,
                        help="NPU core: 0/1/2/3=auto (default: 3)")
    parser.add_argument("--classes", "-c", type=str, default=None,
                        help="Filter classes, comma-separated (e.g.: 3,4,5,8)")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    parser.add_argument("--no-display", action="store_true", help="Do not show real-time window")
    parser.add_argument("--output", "-o", type=str, default=Config.OUTPUT_DIR,
                        help=f"Output directory (default: {Config.OUTPUT_DIR})")

    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    cfg = Config()
    cfg.MODEL_PATH = args.model
    cfg.CONF_THRESHOLD = args.conf
    cfg.IOU_THRESHOLD = args.iou
    cfg.IMG_SIZE = args.imgsz
    cfg.NPU_CORE = args.npu_core
    cfg.SAVE_OUTPUT = not args.no_save
    cfg.SHOW_DISPLAY = not args.no_display
    cfg.OUTPUT_DIR = args.output

    if args.video:
        cfg.VIDEO_PATH = args.video

    if args.classes:
        cfg.CLASS_FILTER = [int(c.strip()) for c in args.classes.split(",") if c.strip()]

    # Print configuration
    print(f"\n{'='*60}")
    print(f"  Airborne Object Detection (RK3576 NPU)")
    print(f"{'='*60}")
    print(f"  RKNN Model:  {cfg.MODEL_PATH}")
    print(f"  NPU Core:    {cfg.NPU_CORE}")
    print(f"  Confidence:  {cfg.CONF_THRESHOLD}")
    print(f"  IoU Thresh:  {cfg.IOU_THRESHOLD}")
    print(f"  Image Size:  {cfg.IMG_SIZE}")
    print(f"  Video Path:  {cfg.VIDEO_PATH}")
    print(f"  Save Output: {'Yes' if cfg.SAVE_OUTPUT else 'No'}")
    print(f"  Display:     {'Yes' if cfg.SHOW_DISPLAY else 'No'}")
    if cfg.CLASS_FILTER:
        names = [VISDRONE_CLASSES.get(c, str(c)) for c in cfg.CLASS_FILTER]
        print(f"  Class Filter:{names}")
    print(f"{'='*60}\n")

    # Check model file
    if not Path(cfg.MODEL_PATH).exists():
        print(f"[ERROR] RKNN model not found: {cfg.MODEL_PATH}")
        print(f"[INFO] Please convert best.pt to RKNN format first:")
        print(f"  1. On PC: python onnx_inference.py --export --model best.pt --onnx best.onnx")
        print(f"  2. On PC: Use RKNN Toolkit2 to convert best.onnx -> best.rknn")
        print(f"  3. Upload best.rknn to RK3576 device")
        sys.exit(1)

    # Get video list
    try:
        video_files = get_video_files(cfg.VIDEO_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[INFO] Found {len(video_files)} video file(s):")
    for vf in video_files:
        print(f"  - {Path(vf).name}")

    # Create detector
    try:
        detector = RKNNDetector(
            model_path=cfg.MODEL_PATH,
            img_size=cfg.IMG_SIZE,
            conf_threshold=cfg.CONF_THRESHOLD,
            iou_threshold=cfg.IOU_THRESHOLD,
            num_classes=cfg.NUM_CLASSES,
            npu_core=cfg.NPU_CORE,
        )
    except ImportError:
        print("[ERROR] rknn-toolkit-lite2 not installed")
        print("[INFO] Install with: pip install rknn-toolkit-lite2")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}")
        sys.exit(1)

    # Process videos
    processor = VideoProcessor(detector, cfg)

    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Starting...")
        processor.process_video(video_path)

    # Release resources
    detector.release()
    print(f"\n[INFO] All done! Processed {len(video_files)} video(s).")


if __name__ == "__main__":
    main()
