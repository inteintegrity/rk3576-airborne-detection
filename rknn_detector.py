"""
RKNN Detector - RK3576 NPU Inference Engine
============================================
YOLOv8 inference engine based on RKNNLite for RK3576 NPU acceleration.

Features:
  - RKNN model loading (multi-core NPU support)
  - Image preprocessing (Letterbox + uint8)
  - NPU inference
  - YOLOv8 3-branch DFL output decoding + NMS post-processing
  - Detection result visualization

Dependencies:
  pip install rknn-toolkit-lite2 opencv-python numpy

Date: 2026-07-27
"""

import cv2
import numpy as np
from pathlib import Path
import time


# ============================================================
# VisDrone Classes (11 categories)
# ============================================================

VISDRONE_CLASSES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car",
    4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle",
    8: "bus", 9: "motor", 10: "others"
}

# Class colors (BGR)
def _generate_colors(n: int = 11) -> dict:
    np.random.seed(42)
    return {i: tuple(map(int, np.random.randint(80, 255, 3))) for i in range(n)}

CLASS_COLORS = _generate_colors(11)

# Image size (width, height)
IMG_SIZE = (640, 640)


# ============================================================
# RKNN Detector
# ============================================================

class RKNNDetector:
    """
    RK3576 NPU Inference Detector

    Encapsulates the full pipeline: model loading, preprocessing,
    inference, and post-processing. Supports YOLOv8 RKNN models
    with 3-branch DFL output format.
    """

    def __init__(self, model_path: str, img_size: int = 640,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 num_classes: int = 11, npu_core: int = 0):
        """
        Args:
            model_path: Path to RKNN model file (.rknn)
            img_size: Inference image size
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            num_classes: Number of classes (VisDrone=11, COCO=80)
            npu_core: NPU core (0/1/2/3=auto)
        """
        self.model_path = model_path
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.npu_core = npu_core
        self.class_names = VISDRONE_CLASSES if num_classes == 11 else {}

        # RKNN runtime
        self.rknn = None
        self._load_model()

    def _load_model(self):
        """Load RKNN model and initialize NPU runtime"""
        from rknnlite.api import RKNNLite

        print(f"[INFO] Loading RKNN model: {self.model_path}")

        self.rknn = RKNNLite()

        # Load model
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {ret}")

        # Select NPU core
        core_map = {
            0: RKNNLite.NPU_CORE_0,
            1: RKNNLite.NPU_CORE_1,
            2: RKNNLite.NPU_CORE_2,
            3: RKNNLite.NPU_CORE_0_1_2,  # Auto (all 3 cores)
        }
        core = core_map.get(self.npu_core, RKNNLite.NPU_CORE_0_1_2)

        # Initialize runtime
        ret = self.rknn.init_runtime(core_mask=core)
        if ret != 0:
            raise RuntimeError(f"Failed to initialize RKNN runtime: {ret}")

        core_name = {0: "NPU_CORE_0", 1: "NPU_CORE_1", 2: "NPU_CORE_2", 3: "NPU_CORE_0_1_2"}
        print(f"[INFO] RKNN model loaded, NPU core: {core_name.get(self.npu_core, 'AUTO')}")

    # ========================================
    # Preprocessing (Letterbox + uint8)
    # ========================================

    def preprocess(self, frame: np.ndarray) -> tuple:
        """
        Preprocessing: Letterbox resize + BGR->RGB + uint8

        RKNN model expects uint8 NHWC input (1, H, W, 3),
        because mean/std is configured in the model itself.

        Returns:
            (input_tensor, letterbox_info)
            input_tensor: (1, H, W, 3) uint8
            letterbox_info: (scale, pad_w, pad_h, orig_h, orig_w)
        """
        h, w = frame.shape[:2]
        new_w, new_h = self.img_size, self.img_size

        # Compute scale ratio (preserve aspect ratio)
        scale = min(new_w / w, new_h / h)
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))

        # Resize
        img = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = (new_w - resize_w) // 2
        pad_h = (new_h - resize_h) // 2

        # Letterbox padding
        img = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Add batch dimension -> (1, H, W, 3) NHWC format, keep uint8
        img = np.expand_dims(img, axis=0)

        return img, (scale, pad_w, pad_h, h, w)

    # ========================================
    # Inference
    # ========================================

    def infer(self, input_tensor: np.ndarray) -> list:
        """Run NPU inference"""
        outputs = self.rknn.inference(inputs=[input_tensor])
        return outputs

    # ========================================
    # Post-processing (3-branch DFL decode + NMS)
    # ========================================

    def postprocess(self, outputs: list, letterbox_info: tuple) -> list:
        """
        Post-processing: YOLOv8 3-branch DFL output decoding + NMS

        RKNN output format: 3 branches, each with 2 tensors (box + class)
        Total: 6 tensors (or 3 if combined)

        Returns:
            list of dicts: [{"box": [x1,y1,x2,y2], "class_id": int, "conf": float}, ...]
        """
        if not outputs or outputs[0] is None:
            return []

        scale, pad_w, pad_h, orig_h, orig_w = letterbox_info

        # Decode 3-branch DFL output
        boxes, classes, scores = self._decode_dfl_output(outputs)

        if len(boxes) == 0:
            return []

        # Restore coordinates to original image
        detections = []
        for box, cls_id, conf in zip(boxes, classes, scores):
            # Remove letterbox padding and scale back
            x1 = (box[0] - pad_w) / scale
            y1 = (box[1] - pad_h) / scale
            x2 = (box[2] - pad_w) / scale
            y2 = (box[3] - pad_h) / scale

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": int(cls_id),
                "class_name": self.class_names.get(int(cls_id), f"cls_{cls_id}"),
                "confidence": float(conf),
            })

        return detections

    def _decode_dfl_output(self, outputs: list) -> tuple:
        """
        Decode YOLOv8 DFL output. Handles different output formats:
        - 6 outputs: 3 branches x 2 (box + class)
        - 3 outputs: 3 branches, each combined (box+class)
        - 1 output: single combined tensor (1, 4+nc, 8400)
        """
        num_outputs = len(outputs)

        # Debug: print output structure (first frame only)
        if not hasattr(self, '_debug_printed'):
            for i, o in enumerate(outputs):
                print(f"  [DEBUG] output[{i}]: shape={o.shape}, dtype={o.dtype}")
            self._debug_printed = True

        all_boxes = []
        all_classes_conf = []
        all_scores = []

        if num_outputs == 1:
            # Single output: (1, 4+nc, 8400) or (1, 8400, 4+nc)
            output = outputs[0]
            if output.shape[0] == 1:
                output = output[0]

            # Determine orientation: (15, 8400) or (8400, 15)
            if output.shape[0] < output.shape[1]:
                output = output.T  # (8400, 15)

            boxes = output[:, :4]       # cx, cy, w, h
            classes_conf = output[:, 4:]  # class scores

            # cx,cy,w,h -> x1,y1,x2,y2
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

            all_boxes = [boxes_xyxy]
            all_classes_conf = [classes_conf]
            all_scores = [np.ones((len(classes_conf), 1), dtype=np.float32)]

        elif num_outputs == 3:
            # 3 branches, each with combined box+class
            for i in range(3):
                tensor = outputs[i]
                # Could be (1, 4*mc+nc, H, W) or separate
                # Try DFL box decode on first 4*mc channels
                c = tensor.shape[1]
                # Heuristic: if channels > 4*16, split into box (4*mc) + class
                # For 11 classes: box=64, cls=11, total=75
                # For DFL with mc=16: box=4*16=64
                mc = 16  # default DFL bins
                box_ch = 4 * mc
                if c > box_ch + self.num_classes:
                    # Combined: box (DFL) + class
                    box_tensor = tensor[:, :box_ch, :, :]
                    cls_tensor = tensor[:, box_ch:, :, :]
                    decoded_boxes = self._box_process(box_tensor)
                    all_boxes.append(decoded_boxes)
                    all_classes_conf.append(self._sp_flatten(cls_tensor))
                    all_scores.append(np.ones_like(
                        self._sp_flatten(cls_tensor)[:, :1], dtype=np.float32))
                else:
                    # Assume it's box only (DFL encoded)
                    decoded_boxes = self._box_process(tensor)
                    all_boxes.append(decoded_boxes)

        elif num_outputs == 6:
            # 3 branches x 2 tensors (box + class)
            for i in range(3):
                box_tensor = outputs[2 * i]
                cls_tensor = outputs[2 * i + 1]

                decoded_boxes = self._box_process(box_tensor)
                all_boxes.append(decoded_boxes)
                all_classes_conf.append(self._sp_flatten(cls_tensor))
                all_scores.append(np.ones_like(
                    self._sp_flatten(cls_tensor)[:, :1], dtype=np.float32))

        else:
            print(f"[WARNING] Unexpected number of outputs: {num_outputs}")
            return np.array([]), np.array([]), np.array([])

        # Concatenate all branches
        boxes = np.concatenate(all_boxes)
        classes_conf = np.concatenate(all_classes_conf)
        scores = np.concatenate(all_scores)

        # Filter by confidence threshold
        scores_flat = scores.reshape(-1)
        class_max_score = np.max(classes_conf, axis=-1)
        class_ids = np.argmax(classes_conf, axis=-1)

        # YOLOv8: score = class_conf (no separate objectness)
        confidences = class_max_score * scores_flat
        mask = confidences >= self.conf_threshold

        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # NMS per class
        keep_boxes = []
        keep_classes = []
        keep_scores = []

        for c in set(class_ids):
            inds = np.where(class_ids == c)[0]
            b = boxes[inds]
            s = confidences[inds]

            keep = self._nms_xyxy(b, s, self.iou_threshold)

            if len(keep) > 0:
                keep_boxes.append(b[keep])
                keep_classes.append(np.full(len(keep), c))
                keep_scores.append(s[keep])

        if not keep_boxes:
            return np.array([]), np.array([]), np.array([])

        return (np.concatenate(keep_boxes),
                np.concatenate(keep_classes),
                np.concatenate(keep_scores))

    def _sp_flatten(self, tensor: np.ndarray) -> np.ndarray:
        """Flatten spatial dimensions: (1, C, H, W) -> (H*W, C)"""
        ch = tensor.shape[1]
        tensor = tensor.transpose(0, 2, 3, 1)  # (1, H, W, C)
        return tensor.reshape(-1, ch)  # (H*W, C)

    def _dfl(self, position: np.ndarray) -> np.ndarray:
        """
        Distribution Focal Loss decoding.

        Input: (n, 4*mc, h, w) where mc = channels / 4
        Output: (n, 4, h, w) decoded box offsets
        """
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num

        y = position.reshape(n, p_num, mc, h, w)

        # Softmax over mc dimension
        y_max = np.max(y, axis=2, keepdims=True)
        y_exp = np.exp(y - y_max)
        y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)

        # Weighted sum with arange(0, mc)
        acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        y = np.sum(y_softmax * acc_matrix, axis=2)

        return y

    def _box_process(self, position: np.ndarray) -> np.ndarray:
        """
        Decode box predictions from DFL output.

        Input: (1, 4*mc, grid_h, grid_w)
        Output: (grid_h*grid_w, 4) in xyxy format (on letterboxed image)
        """
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)

        stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

        position = self._dfl(position)

        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        # Flatten to (grid_h*grid_w, 4)
        ch = xyxy.shape[1]
        xyxy = xyxy.transpose(0, 2, 3, 1)
        return xyxy.reshape(-1, ch)

    def _nms_xyxy(self, boxes: np.ndarray, scores: np.ndarray,
                  iou_threshold: float) -> np.ndarray:
        """NMS for xyxy boxes."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=int)

    # ========================================
    # Visualization
    # ========================================

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection results on image"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_id = det["class_id"]
            label = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    # ========================================
    # Full Detection Pipeline
    # ========================================

    def detect(self, frame: np.ndarray) -> list:
        """
        Full detection pipeline: preprocess -> inference -> postprocess

        Args:
            frame: BGR image

        Returns:
            List of detection results
        """
        # Preprocess
        input_tensor, letterbox_info = self.preprocess(frame)

        # Inference
        outputs = self.infer(input_tensor)

        # Post-process
        detections = self.postprocess(outputs, letterbox_info)

        return detections

    def release(self):
        """Release RKNN resources"""
        if self.rknn is not None:
            self.rknn.release()
            print("[INFO] RKNN resources released")
