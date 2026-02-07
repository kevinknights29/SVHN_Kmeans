"""
Character segmentation for multi-digit house numbers.
"""

import numpy as np
from .utils import rgb_to_gray, compute_vertical_projection
from typing import List, Dict, Tuple


class CharacterSegmenter:
    """
    Segments multi-digit house numbers into individual character candidates.

    Uses vertical projection analysis to find potential character boundaries.
    """

    def __init__(self, min_char_width: int = 8, max_char_width: int = 24,
                 min_gap_threshold: float = 0.3):
        """
        Initialize segmenter.

        Args:
            min_char_width: Minimum width for a character
            max_char_width: Maximum width for a character
            min_gap_threshold: Minimum relative depth for detecting valleys
        """
        self.min_char_width = min_char_width
        self.max_char_width = max_char_width
        self.min_gap_threshold = min_gap_threshold

    def find_breakpoints(self, image: np.ndarray) -> List[int]:
        """
        Find candidate vertical boundaries between characters.

        Args:
            image: Input image (house number patch)

        Returns:
            List of x-coordinates representing potential breakpoints
        """
        # Convert to grayscale
        gray = rgb_to_gray(image)

        # Compute vertical projection profile
        projection = compute_vertical_projection(gray)

        # Normalize projection
        proj_max = np.max(projection)
        if proj_max > 0:
            projection = projection / proj_max

        # Find local minima as candidate breakpoints
        breakpoints = [0]  # Always start at left edge

        # Compute gradient to find valleys
        gradient = np.gradient(projection)

        # Find valleys (local minima)
        for i in range(1, len(projection) - 1):
            # Check if it's a local minimum
            if projection[i] < projection[i - 1] and projection[i] < projection[i + 1]:
                # Check if the valley is deep enough
                if projection[i] < (1.0 - self.min_gap_threshold):
                    # Check minimum spacing from last breakpoint
                    if i - breakpoints[-1] >= self.min_char_width:
                        breakpoints.append(i)

        # Always end at right edge
        if breakpoints[-1] != gray.shape[1]:
            breakpoints.append(gray.shape[1])

        return breakpoints

    def generate_candidate_segments(self, image: np.ndarray,
                                   breakpoints: List[int] = None) -> List[Dict]:
        """
        Generate all valid pairs of breakpoints as candidate segments.

        Args:
            image: Input image
            breakpoints: List of breakpoint positions (if None, will be computed)

        Returns:
            List of candidate segments, each as a dictionary with:
                - 'segment': The image segment
                - 'left': Left boundary
                - 'right': Right boundary
                - 'position': Position index in sequence
        """
        if breakpoints is None:
            breakpoints = self.find_breakpoints(image)

        candidates = []

        for i in range(len(breakpoints)):
            for j in range(i + 1, len(breakpoints)):
                left = breakpoints[i]
                right = breakpoints[j]
                width = right - left

                # Check if width is valid
                if self.min_char_width <= width <= self.max_char_width:
                    segment = image[:, left:right]
                    candidates.append({
                        'segment': segment,
                        'left': left,
                        'right': right,
                        'width': width,
                        'position': i
                    })

        return candidates

    def segment_greedy(self, image: np.ndarray) -> List[Dict]:
        """
        Simple greedy segmentation (baseline method).

        Args:
            image: Input image

        Returns:
            List of segments in left-to-right order
        """
        breakpoints = self.find_breakpoints(image)

        # Create segments from consecutive breakpoints
        segments = []
        for i in range(len(breakpoints) - 1):
            left = breakpoints[i]
            right = breakpoints[i + 1]
            width = right - left

            if width >= self.min_char_width:
                segment = image[:, left:right]
                segments.append({
                    'segment': segment,
                    'left': left,
                    'right': right,
                    'width': width,
                    'position': i
                })

        return segments

    def visualize_breakpoints(self, image: np.ndarray,
                            breakpoints: List[int] = None) -> np.ndarray:
        """
        Visualize breakpoints on the image.

        Args:
            image: Input image
            breakpoints: List of breakpoint positions (if None, will be computed)

        Returns:
            Image with breakpoints drawn
        """
        import cv2

        if breakpoints is None:
            breakpoints = self.find_breakpoints(image)

        result = image.copy()

        # Draw vertical lines at breakpoints
        for bp in breakpoints:
            cv2.line(result, (bp, 0), (bp, image.shape[0]), (0, 255, 0), 2)

        return result


class GeometryModel:
    """
    Models the expected geometry of digits for scoring segmentations.
    """

    def __init__(self):
        """Initialize with default reference sizes."""
        # These are typical values, could be learned from training data
        self.mean_width = 16
        self.std_width = 4
        self.mean_height = 28
        self.std_height = 4
        self.mean_aspect_ratio = 0.65
        self.std_aspect_ratio = 0.15

    def learn_from_data(self, bboxes: List[Dict]):
        """
        Learn geometry parameters from training data.

        Args:
            bboxes: List of bounding box dictionaries with 'width' and 'height'
        """
        widths = [bbox['width'] for bbox in bboxes]
        heights = [bbox['height'] for bbox in bboxes]
        aspect_ratios = [bbox['width'] / max(bbox['height'], 1) for bbox in bboxes]

        self.mean_width = np.mean(widths)
        self.std_width = np.std(widths)
        self.mean_height = np.mean(heights)
        self.std_height = np.std(heights)
        self.mean_aspect_ratio = np.mean(aspect_ratios)
        self.std_aspect_ratio = np.std(aspect_ratios)

    def score(self, bbox: Dict) -> float:
        """
        Score a bounding box based on how well it matches expected geometry.

        Args:
            bbox: Dictionary with 'width' and optionally 'height'

        Returns:
            Geometry score (higher is better)
        """
        width = bbox['width']
        height = bbox.get('height', self.mean_height)

        aspect_ratio = width / max(height, 1)

        # Gaussian penalty for deviation from reference
        width_score = -((width - self.mean_width) ** 2) / (2 * self.std_width ** 2)

        aspect_score = -((aspect_ratio - self.mean_aspect_ratio) ** 2) / \
                      (2 * self.std_aspect_ratio ** 2)

        # Combined score
        return width_score + aspect_score
