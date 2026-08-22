"""Event logging and snapshot recording module for motion detections."""
import os
import csv
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import cv2
import numpy as np


class MotionLogger:
    """Logs motion events to CSV/JSON files and captures snapshot images."""

    def __init__(
        self,
        log_dir: str = "logs",
        snapshot_dir: str = "snapshots",
        cooldown_seconds: float = 1.0,
        draw_annotations: bool = True,
        active_sections: Optional[Set[int]] = None,
    ):
        """
        Initialize the motion logger.

        :param log_dir: Directory to store log files.
        :param snapshot_dir: Directory to save snapshot images.
        :param cooldown_seconds: Minimum seconds between reporting consecutive motion events.
        :param draw_annotations: Whether to draw bounding boxes and labels on saved snapshots.
        :param active_sections: Monitored screen sections ({1, 2, 3, 4}).
        """
        self.log_dir = log_dir
        self.snapshot_dir = snapshot_dir
        self.cooldown_seconds = cooldown_seconds
        self.draw_annotations = draw_annotations
        self.active_sections = active_sections if active_sections is not None else {1, 2, 3, 4}

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # File paths
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = os.path.join(self.log_dir, f"motion_events_{date_str}.csv")
        self.json_path = os.path.join(self.log_dir, f"motion_events_{date_str}.jsonl")

        self.last_event_time: float = 0.0
        self.last_snapshot_time: Dict[str, float] = {}
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV header if file is new."""
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event_id",
                    "categories_detected",
                    "human_count",
                    "car_count",
                    "other_count",
                    "details_json",
                    "snapshot_path"
                ])

    def log_event(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_idx: int = 0,
        fps: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Log a detection event and optionally save a snapshot.

        :param frame: Current BGR frame.
        :param detections: List of detection dictionaries from ObjectClassifier.
        :param frame_idx: Frame index in the stream/video.
        :param fps: Estimated FPS of processing.
        :return: Event summary dictionary or None if no detections.
        """
        if not detections:
            return None

        # Check event reporting cooldown
        current_time = time.time()
        if self.last_event_time > 0 and (current_time - self.last_event_time < self.cooldown_seconds):
            return None

        self.last_event_time = current_time

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        event_id = f"evt_{now.strftime('%Y%m%d_%H%M%S_%f')}"

        # Group counts by category
        categories = [d["category"] for d in detections]
        human_count = categories.count("human")
        car_count = categories.count("car")
        other_count = categories.count("other")
        unique_categories = sorted(list(set(categories)))

        # Snapshot saving
        snapshot_path = ""
        if frame is not None and frame.size > 0:
            primary_category = unique_categories[0] if unique_categories else "other"
            snap_filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{primary_category}_{event_id[-4:]}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, snap_filename)

            snap_frame = frame.copy()
            if self.draw_annotations:
                snap_frame = self.annotate_frame(snap_frame, detections, active_sections=self.active_sections)

            cv2.imwrite(snapshot_path, snap_frame)

        event_data = {
            "timestamp": timestamp_str,
            "event_id": event_id,
            "frame_index": frame_idx,
            "fps": round(fps, 1),
            "categories": unique_categories,
            "counts": {
                "human": human_count,
                "car": car_count,
                "other": other_count,
            },
            "detections": detections,
            "snapshot": snapshot_path,
        }

        # Write to JSONL
        with open(self.json_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")

        # Write to CSV
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                event_id,
                "|".join(unique_categories),
                human_count,
                car_count,
                other_count,
                json.dumps(detections, ensure_ascii=False),
                snapshot_path,
            ])

        return event_data

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        active_sections: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, category labels, and section grid on a copy of the frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        x_mid, y_mid = int(w / 2), int(h / 2)

        # Draw 4-section division grid lines
        grid_color = (60, 60, 60)
        cv2.line(annotated, (x_mid, 0), (x_mid, h), grid_color, 1)
        cv2.line(annotated, (0, y_mid), (w, y_mid), grid_color, 1)

        # Draw section corner tags
        sections_info = [
            (1, 10, 25),              # Top Left
            (2, x_mid + 10, 25),      # Top Right
            (3, 10, y_mid + 25),      # Bottom Left
            (4, x_mid + 10, y_mid + 25) # Bottom Right
        ]
        active = active_sections if active_sections is not None else {1, 2, 3, 4}
        for sec_num, sx, sy in sections_info:
            is_active = sec_num in active
            tag_color = (0, 255, 255) if is_active else (80, 80, 80)
            tag_text = f"S{sec_num}" + ("" if is_active else " (OFF)")
            cv2.putText(annotated, tag_text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

        # Color mapping: BGR
        colors = {
            "human": (0, 0, 255),    # Red
            "car": (255, 128, 0),    # Blue / Cyan
            "other": (0, 255, 0),    # Green
        }

        for d in detections:
            cat = d.get("category", "other")
            raw = d.get("raw_label", "")
            conf = d.get("confidence", 0.0)
            bbox = d.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox

            color = colors.get(cat, (0, 255, 255))
            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label
            label_text = f"{cat.upper()} ({raw} {conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
