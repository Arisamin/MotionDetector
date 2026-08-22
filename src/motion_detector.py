"""Motion detector module using background subtraction and contour analysis."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


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
    ):
        """
        Initialize the motion detector.

        :param min_area: Minimum contour area in pixels to consider as motion.
        :param history: Length of history for MOG2 background subtractor.
        :param var_threshold: Threshold on squared Mahalanobis distance.
        :param detect_shadows: Whether to detect and mark shadows.
        :param blur_kernel_size: Gaussian blur kernel size to reduce noise.
        :param dilation_iterations: Number of dilation iterations to bridge fragmented contours.
        """
        self.min_area = min_area
        self.blur_kernel_size = blur_kernel_size
        self.dilation_iterations = dilation_iterations

        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def detect(self, frame: np.ndarray, learning_rate: float = -1) -> Dict[str, Any]:
        """
        Process a single frame and detect motion bounding boxes.

        :param frame: BGR input frame.
        :param learning_rate: Learning rate for background subtractor (-1 for auto).
        :return: Dictionary containing 'has_motion' (bool), 'boxes' (list of (x,y,w,h)), 'mask' (fg mask).
        """
        if frame is None or frame.size == 0:
            return {"has_motion": False, "boxes": [], "mask": None}

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
                boxes.append((x, y, w, h))

        return {
            "has_motion": len(boxes) > 0,
            "boxes": boxes,
            "mask": fg_mask,
        }
