"""Motion detector module using background subtraction and contour analysis."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set


def parse_sections(sections_str: Optional[str]) -> Set[int]:
    """
    Parse an underscore-separated sections string (e.g. '1_2', '1_3_4') into a set of integers.
    Sections are:
      1 = Top Left
      2 = Top Right
      3 = Bottom Left
      4 = Bottom Right
    If None or empty, returns all sections {1, 2, 3, 4}.
    """
    if not sections_str:
        return {1, 2, 3, 4}

    parsed = set()
    for item in sections_str.split("_"):
        item = item.strip()
        if item.isdigit():
            val = int(item)
            if val in {1, 2, 3, 4}:
                parsed.add(val)

    return parsed if parsed else {1, 2, 3, 4}


def get_box_sections(box: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Set[int]:
    """
    Determine which screen section(s) [1, 2, 3, 4] a bounding box intersects.
    """
    x, y, w, h = box
    x2, y2 = x + w, y + h
    x_mid = frame_width / 2.0
    y_mid = frame_height / 2.0

    sections = set()
    # Section 1: Top Left [0, 0, x_mid, y_mid]
    if x < x_mid and y < y_mid:
        sections.add(1)
    # Section 2: Top Right [x_mid, 0, width, y_mid]
    if x2 > x_mid and y < y_mid:
        sections.add(2)
    # Section 3: Bottom Left [0, y_mid, x_mid, height]
    if x < x_mid and y2 > y_mid:
        sections.add(3)
    # Section 4: Bottom Right [x_mid, y_mid, width, height]
    if x2 > x_mid and y2 > y_mid:
        sections.add(4)

    return sections


class MotionDetector:
    """Detects motion in video frames using MOG2 background subtraction."""

    def __init__(
        self,
        min_area: int = 500,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = False,
        blur_kernel_size: Tuple[int, int] = (21, 21),
        dilation_iterations: int = 2,
        active_sections: Optional[Set[int]] = None,
    ):
        """
        Initialize the motion detector.

        :param min_area: Minimum contour area in pixels to consider as motion.
        :param history: Length of history for MOG2 background subtractor.
        :param var_threshold: Threshold on squared Mahalanobis distance.
        :param detect_shadows: Whether to detect and mark shadows.
        :param blur_kernel_size: Gaussian blur kernel size to reduce noise.
        :param dilation_iterations: Number of dilation iterations to bridge fragmented contours.
        :param active_sections: Set of allowed section numbers ({1, 2, 3, 4}) from which to report motion.
        """
        self.min_area = min_area
        self.blur_kernel_size = blur_kernel_size
        self.dilation_iterations = dilation_iterations
        self.active_sections = active_sections if active_sections is not None else {1, 2, 3, 4}

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def detect(self, frame: np.ndarray, learning_rate: float = -1) -> Dict[str, Any]:
        """
        Process a single frame and detect motion bounding boxes filtered by active sections.

        :param frame: BGR input frame.
        :param learning_rate: Learning rate for background subtractor (-1 for auto).
        :return: Dictionary containing 'has_motion' (bool), 'boxes' (list of (x,y,w,h)), 'mask' (fg mask).
        """
        if frame is None or frame.size == 0:
            return {"has_motion": False, "boxes": [], "mask": None}

        h_frame, w_frame = frame.shape[:2]

        # Apply Gaussian blur to reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)

        # Foreground mask
        fg_mask = self.subtractor.apply(blurred, learningRate=learning_rate)

        # Threshold to remove gray shadows if any
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological dilation to merge nearby contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=self.dilation_iterations)

        # Find contours of moving objects
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                # Check if box intersects with any active section
                box_sections = get_box_sections((x, y, w, h), w_frame, h_frame)
                if any(s in self.active_sections for s in box_sections):
                    boxes.append((x, y, w, h))

        return {
            "has_motion": len(boxes) > 0,
            "boxes": boxes,
            "mask": fg_mask,
        }

