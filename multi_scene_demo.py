"""
Multi-Scene Airborne Object Detection Demo
=============================================
Displays 4 video detection results in a 2x2 grid with a title bar.
Designed for RK3576 NPU inference demonstration.

Usage:
  python3 multi_scene_demo.py --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4
  python3 multi_scene_demo.py --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4 --no-display

Layout:
  ┌──────────────────────────────────────┐
  │     RK3576多场景空中目标检测          │  ← Title bar
  ├──────────────┬───────────────────────┤
  │  Scene 1     │  Scene 2              │
  │  (video1)    │  (video2)             │
  ├──────────────┼───────────────────────┤
  │  Scene 3     │  Scene 4              │
  │  (video3)    │  (video4)             │
  └──────────────┴───────────────────────┘
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys

from rknn_detector import RKNNDetector, VISDRONE_CLASSES, CLASS_COLORS


# ============================================================
# Configuration
# ============================================================

class Config:
    # Model
    MODEL_PATH: str = "best.rknn"
    IMG_SIZE: int = 640
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    NPU_CORE: int = 3  # 0/1/2/3=auto

    # Layout
    TITLE_TEXT = "RK3576 Multi-Scene Airborne Object Detection"
    TITLE_TEXT_CN = "RK3576 多场景空中目标检测"
    TITLE_HEIGHT: int = 70           # Title bar height (pixels)
    CELL_W: int = 640                # Each video cell width
    CELL_H: int = 360                # Each video cell height
    PADDING: int = 4                # Gap between cells
    BG_COLOR = (10, 10, 10)         # Background color (BGR)

    # Video
    OUTPUT_FPS: int = 15
    SAVE_OUTPUT: bool = True
    SHOW_DISPLAY: bool = True
    OUTPUT_DIR: str = "output"


# ============================================================
# Grid Layout Helper
# ============================================================

class GridLayout:
    """Calculate positions for 2x2 grid layout."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.title_h = cfg.TITLE_HEIGHT
        self.cell_w = cfg.CELL_W
        self.cell_h = cfg.CELL_H
        self.pad = cfg.PADDING

        # Total composite size
        self.total_w = self.cell_w * 2 + self.pad * 3
        self.total_h = self.title_h + self.cell_h * 2 + self.pad * 3

        # Cell positions (top-left corner)
        self.positions = [
            (self.pad, self.title_h + self.pad),                              # top-left
            (self.cell_w + self.pad * 2, self.title_h + self.pad),            # top-right
            (self.pad, self.title_h + self.cell_h + self.pad * 2),            # bottom-left
            (self.cell_w + self.pad * 2, self.title_h + self.cell_h + self.pad * 2),  # bottom-right
        ]

    def draw_title(self, canvas: np.ndarray):
        """Draw title bar at the top."""
        # Title background
        cv2.rectangle(canvas, (0, 0), (self.total_w, self.title_h),
                       (30, 30, 60), -1)

        # Title text (Chinese)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = self.cfg.TITLE_TEXT_CN
        font_scale = 1.2
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (self.total_w - tw) // 2
        y = (self.title_h + th) // 2
        cv2.putText(canvas, text, (x, y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # Subtitle (English)
        sub_text = self.cfg.TITLE_TEXT
        sub_scale = 0.5
        (stw, sth), _ = cv2.getTextSize(sub_text, font, sub_scale, 1)
        sx = (self.total_w - stw) // 2
        sy = y + th + 8
        cv2.putText(canvas, sub_text, (sx, sy), font, sub_scale,
                    (180, 180, 180), 1, cv2.LINE_AA)

    def draw_cell(self, canvas: np.ndarray, frame: np.ndarray,
                  pos_idx: int, label: str, stats: str = ""):
        """Draw a video frame into a grid cell."""
        x, y = self.positions[pos_idx]

        # Resize frame to cell size
        cell = cv2.resize(frame, (self.cell_w, self.cell_h))

        # Draw cell background
        cv2.rectangle(canvas, (x - 1, y - 1),
                       (x + self.cell_w + 1, y + self.cell_h + 1),
                       (50, 50, 50), 1)

        # Place frame
        canvas[y:y + self.cell_h, x:x + self.cell_w] = cell

        # Scene label (top-left of cell)
        label_bg_h = 28
        cv2.rectangle(canvas, (x, y), (x + 200, y + label_bg_h),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (x + 8, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1, cv2.LINE_AA)

        # Stats (top-right of cell)
        if stats:
            (stw, sth), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas,
                          (x + self.cell_w - stw - 16, y),
                          (x + self.cell_w, y + label_bg_h),
                          (0, 0, 0), -1)
            cv2.putText(canvas, stats,
                        (x + self.cell_w - stw - 8, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

    def create_canvas(self) -> np.ndarray:
        """Create blank canvas with background."""
        canvas = np.full((self.total_h, self.total_w, 3),
                         self.cfg.BG_COLOR, dtype=np.uint8)
        self.draw_title(canvas)
        return canvas


# ============================================================
# Multi-Scene Processor
# ============================================================

class MultiSceneProcessor:
    """Process 4 videos simultaneously and create 2x2 grid output."""

    def __init__(self, detector: RKNNDetector, cfg: Config):
        self.detector = detector
        self.cfg = cfg
        self.layout = GridLayout(cfg)

    def process_videos(self, video_paths: list, scene_names: list = None):
        """Process 4 videos and create composite output."""
        num_videos = len(video_paths)
        if num_videos != 4:
            print(f"[WARNING] Expected 4 videos, got {num_videos}. Using first 4.")
            video_paths = video_paths[:4]

        if scene_names is None:
            scene_names = [f"Scene {i+1}" for i in range(4)]

        # Open all video captures
        caps = []
        for i, vp in enumerate(video_paths):
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video {i+1}: {vp}")
                for c in caps:
                    c.release()
                return
            caps.append(cap)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[INFO] Video {i+1}: {Path(vp).name} | FPS: {fps:.1f} | Frames: {total}")

        # Output video writer
        writer = None
        if self.cfg.SAVE_OUTPUT:
            out_dir = Path(self.cfg.OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"multi_scene_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, self.cfg.OUTPUT_FPS,
                (self.layout.total_w, self.layout.total_h)
            )
            print(f"[INFO] Output: {out_path}")
            print(f"[INFO] Composite size: {self.layout.total_w}x{self.layout.total_h}")

        # Per-video statistics
        frame_counts = [0] * 4
        total_detections = [0] * 4
        total_inference_time = 0.0
        total_frames = 0

        print(f"\n[INFO] Starting multi-scene detection...\n")

        while True:
            frames = []
            all_done = True

            # Read one frame from each video
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    all_done = False
                else:
                    # Video ended, use last frame or black frame
                    frames.append(None)

            if all_done:
                break

            # Create canvas
            canvas = self.layout.create_canvas()

            # Process each video frame
            for i, frame in enumerate(frames):
                if frame is None:
                    # Draw "Video Ended" placeholder
                    x, y = self.layout.positions[i]
                    cv2.rectangle(canvas, (x, y),
                                   (x + self.cfg.CELL_W, y + self.cfg.CELL_H),
                                   (20, 20, 20), -1)
                    text = "Ended"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    cv2.putText(canvas, text,
                                (x + (self.cfg.CELL_W - tw) // 2,
                                 y + (self.cfg.CELL_H + th) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (100, 100, 100), 2, cv2.LINE_AA)
                    self.layout.draw_cell(canvas, np.zeros((self.cfg.CELL_H, self.cfg.CELL_W, 3), dtype=np.uint8),
                                          i, scene_names[i], "Ended")
                    continue

                frame_counts[i] += 1
                total_frames += 1

                # NPU inference
                t_start = time.perf_counter()
                detections = self.detector.detect(frame)
                t_end = time.perf_counter()
                total_inference_time += (t_end - t_start)

                # Draw detections
                annotated = self.detector.draw_detections(frame, detections)

                # Count detections
                det_count = len(detections)
                total_detections[i] += det_count

                # Class summary for this frame
                class_counts = {}
                for det in detections:
                    name = det["class_name"]
                    class_counts[name] = class_counts.get(name, 0) + 1

                # Build stats string
                top_classes = sorted(class_counts.items(),
                                     key=lambda x: x[1], reverse=True)[:3]
                stats_parts = [f"{n}:{c}" for n, c in top_classes]
                stats = f"Det:{det_count} | " + " ".join(stats_parts) if stats_parts else f"Det:{det_count}"

                # Draw into grid cell
                self.layout.draw_cell(canvas, annotated, i,
                                      scene_names[i], stats)

            # Draw overall FPS info on title bar
            if total_frames > 0:
                avg_inf = total_inference_time / total_frames * 1000
                fps_text = f"NPU Inference: {avg_inf:.1f}ms/frame | Total Frames: {total_frames}"
                cv2.putText(canvas, fps_text,
                            (10, self.cfg.TITLE_HEIGHT - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (120, 255, 120), 1, cv2.LINE_AA)

            # Write frame
            if writer:
                writer.write(canvas)

            # Display
            if self.cfg.SHOW_DISPLAY:
                # Scale down for display if too large
                display = canvas
                if self.layout.total_w > 1280:
                    scale = 1280 / self.layout.total_w
                    display = cv2.resize(canvas,
                                         (int(self.layout.total_w * scale),
                                          int(self.layout.total_h * scale)))
                cv2.imshow("Multi-Scene Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[INFO] Interrupted by user")
                    break

            # Progress
            if total_frames % 30 == 0:
                avg_inf = total_inference_time / total_frames * 1000 if total_frames > 0 else 0
                print(f"  Progress: {total_frames} frames | "
                      f"Avg inference: {avg_inf:.1f}ms | "
                      f"Detections: {total_detections}")

        # Cleanup
        for cap in caps:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Summary
        self._print_summary(frame_counts, total_detections, total_inference_time, total_frames)

    def _print_summary(self, frame_counts, total_detections, total_time, total_frames):
        print(f"\n{'='*60}")
        print(f"  Multi-Scene Detection Summary")
        print(f"{'='*60}")
        for i in range(4):
            print(f"  Scene {i+1}: {frame_counts[i]} frames, "
                  f"{total_detections[i]} detections")
        print(f"  {'-'*40}")
        print(f"  Total frames:      {total_frames}")
        print(f"  Total inference:   {total_time:.2f}s")
        if total_frames > 0:
            avg = total_time / total_frames * 1000
            print(f"  Avg per frame:     {avg:.1f}ms")
            print(f"  Avg per scene:     {avg/4:.1f}ms")
        print(f"  Total detections:  {sum(total_detections)}")
        print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Scene Airborne Object Detection Demo (2x2 grid)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 4 videos
  python3 multi_scene_demo.py --videos video/video1.mp4 video/video3.mp4 video/video5.mp4 video/video6.mp4

  # With custom scene names
  python3 multi_scene_demo.py --videos v1.mp4 v2.mp4 v3.mp4 v4.mp4 --names "Highway" "Urban" "Park" "Square"

  # Headless mode
  python3 multi_scene_demo.py --videos v1.mp4 v2.mp4 v3.mp4 v4.mp4 --no-display
        """,
    )
    parser.add_argument("--model", "-m", type=str, default=Config.MODEL_PATH,
                        help=f"RKNN model path (default: {Config.MODEL_PATH})")
    parser.add_argument("--videos", "-v", nargs=4, required=True,
                        help="4 video file paths")
    parser.add_argument("--names", nargs=4, default=None,
                        help="4 scene names (default: Scene 1-4)")
    parser.add_argument("--conf", type=float, default=Config.CONF_THRESHOLD,
                        help=f"Confidence threshold (default: {Config.CONF_THRESHOLD})")
    parser.add_argument("--iou", type=float, default=Config.IOU_THRESHOLD,
                        help=f"NMS IoU threshold (default: {Config.IOU_THRESHOLD})")
    parser.add_argument("--npu-core", type=int, default=Config.NPU_CORE,
                        help="NPU core: 0/1/2/3=auto (default: 3)")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    parser.add_argument("--no-display", action="store_true", help="Do not show real-time window")
    parser.add_argument("--output", "-o", type=str, default=Config.OUTPUT_DIR,
                        help=f"Output directory (default: {Config.OUTPUT_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.MODEL_PATH = args.model
    cfg.CONF_THRESHOLD = args.conf
    cfg.IOU_THRESHOLD = args.iou
    cfg.NPU_CORE = args.npu_core
    cfg.SAVE_OUTPUT = not args.no_save
    cfg.SHOW_DISPLAY = not args.no_display
    cfg.OUTPUT_DIR = args.output

    # Print config
    print(f"\n{'='*60}")
    print(f"  RK3576 Multi-Scene Airborne Object Detection")
    print(f"{'='*60}")
    print(f"  Model:       {cfg.MODEL_PATH}")
    print(f"  NPU Core:    {cfg.NPU_CORE}")
    print(f"  Confidence:  {cfg.CONF_THRESHOLD}")
    print(f"  IoU Thresh:  {cfg.IOU_THRESHOLD}")
    print(f"  Grid:        2x2 ({cfg.CELL_W}x{cfg.CELL_H} per cell)")
    print(f"  Composite:   {GridLayout(cfg).total_w}x{GridLayout(cfg).total_h}")
    print(f"  Videos:")
    for i, v in enumerate(args.videos):
        name = args.names[i] if args.names else f"Scene {i+1}"
        print(f"    [{i+1}] {name}: {v}")
    print(f"{'='*60}\n")

    # Check model
    if not Path(cfg.MODEL_PATH).exists():
        print(f"[ERROR] RKNN model not found: {cfg.MODEL_PATH}")
        sys.exit(1)

    # Check videos
    for i, v in enumerate(args.videos):
        if not Path(v).exists():
            print(f"[ERROR] Video {i+1} not found: {v}")
            sys.exit(1)

    # Create detector
    try:
        detector = RKNNDetector(
            model_path=cfg.MODEL_PATH,
            img_size=cfg.IMG_SIZE,
            conf_threshold=cfg.CONF_THRESHOLD,
            iou_threshold=cfg.IOU_THRESHOLD,
            num_classes=11,
            npu_core=cfg.NPU_CORE,
        )
    except Exception as e:
        print(f"[ERROR] Detector init failed: {e}")
        sys.exit(1)

    # Process
    processor = MultiSceneProcessor(detector, cfg)
    processor.process_videos(args.videos, args.names)

    detector.release()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
