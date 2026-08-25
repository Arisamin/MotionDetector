"""Object classification module using YOLOv8 to categorize detections into human, car, or other."""
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("ObjectClassifier")

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("Ultralytics/YOLO not found. Running in lightweight motion-only mode.")


class ObjectClassifier:
    """Classifies objects in frames or ROIs into human, car, or other categories."""

    HUMAN_CLASSES = {"person"}
    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}

    CATEGORY_HUMAN = "human"
    CATEGORY_CAR = "car"
    CATEGORY_OTHER = "other"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        """
        Initialize the YOLO-based classifier.

        :param model_name: YOLO model file or weights name (default: yolov8n.pt).
        :param confidence_threshold: Minimum confidence to consider a detection.
        :param device: Inference device ('cpu', 'cuda', etc. Default: auto-selected).
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        if HAS_ULTRALYTICS:
            try:
                self.model = YOLO(model_name)
            except Exception as e:
                logger.warning(f"Could not load YOLO model {model_name}: {e}. Falling back to motion-only mode.")

    def map_class_to_category(self, class_name: str) -> str:
        """Map YOLO COCO class names to simplified category: human, car, other."""
        lower_name = class_name.lower()
        if lower_name in self.HUMAN_CLASSES:
            return self.CATEGORY_HUMAN
        if lower_name in self.VEHICLE_CLASSES:
            return self.CATEGORY_CAR
        return self.CATEGORY_OTHER

    def classify_frame(self, frame: np.ndarray, motion_boxes: Optional[List[tuple]] = None) -> List[Dict[str, Any]]:
        """
        Run inference on the frame and associate detections with motion regions.

        :param frame: BGR image numpy array.
        :param motion_boxes: Optional list of (x, y, w, h) bounding boxes with motion.
        :return: List of detections with category, original class, confidence, and bbox (x1, y1, x2, y2).
        """
        if frame is None or frame.size == 0:
            return []

        # If no YOLO model loaded, return motion regions as 'other'
        if self.model is None:
            detections = []
            if motion_boxes:
                for (x, y, w, h) in motion_boxes:
                    detections.append({
                        "category": self.CATEGORY_OTHER,
                        "raw_label": "motion_blob",
                        "confidence": 1.0,
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    })
            return detections

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            # If motion was detected but no COCO object was recognized, categorize as 'other'
            if motion_boxes:
                for (x, y, w, h) in motion_boxes:
                    detections.append({
                        "category": self.CATEGORY_OTHER,
                        "raw_label": "motion_blob",
                        "confidence": 1.0,
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    })
            return detections

        names = res.names
        for box in boxes:
            cls_id = int(box.cls[0])
            raw_label = names.get(cls_id, "unknown")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            bbox = [int(v) for v in xyxy]

            category = self.map_class_to_category(raw_label)

            # If motion boxes were provided, check if this detection overlaps with any motion box
            is_moving = True
            if motion_boxes:
                is_moving = self._has_overlap(bbox, motion_boxes)

            if is_moving:
                detections.append({
                    "category": category,
                    "raw_label": raw_label,
                    "confidence": round(conf, 4),
                    "bbox": bbox,
                })

        # If there are motion boxes that did not overlap with any YOLO detection, add them as 'other'
        if motion_boxes:
            for (mx, my, mw, mh) in motion_boxes:
                mbox = [int(mx), int(my), int(mx + mw), int(my + mh)]
                has_detected_object = any(self._box_intersection_over_area(mbox, d["bbox"]) > 0.2 for d in detections)
                if not has_detected_object:
                    detections.append({
                        "category": self.CATEGORY_OTHER,
                        "raw_label": "motion_blob",
                        "confidence": 1.0,
                        "bbox": mbox,
                    })

        return detections

    @staticmethod
    def _has_overlap(det_box: List[int], motion_boxes: List[tuple], min_overlap: float = 0.1) -> bool:
        """Check if detected bounding box overlaps with any motion box."""
        dx1, dy1, dx2, dy2 = det_box
        for (mx, my, mw, mh) in motion_boxes:
            mx2, my2 = mx + mw, my + mh
            # Calculate intersection
            ix1 = max(dx1, mx)
            iy1 = max(dy1, my)
            ix2 = min(dx2, mx2)
            iy2 = min(dy2, my2)

            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = (dx2 - dx1) * (dy2 - dy1)
                if box_area > 0 and (inter_area / box_area) >= min_overlap:
                    return True
        return False

    @staticmethod
    def _box_intersection_over_area(boxA: List[int], boxB: List[int]) -> float:
        """Calculate intersection area over boxA area."""
        ix1 = max(boxA[0], boxB[0])
        iy1 = max(boxA[1], boxB[1])
        ix2 = min(boxA[2], boxB[2])
        iy2 = min(boxA[3], boxB[3])

        if ix1 < ix2 and iy1 < iy2:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            return inter_area / areaA if areaA > 0 else 0.0
        return 0.0
