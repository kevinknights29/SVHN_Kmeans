"""
Utility functions for image processing and preprocessing.
"""

import numpy as np
import cv2
from typing import Tuple


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        Grayscale image (H, W)
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_to_32x32(image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image to 32x32 pixels.

    Args:
        image: Input image
        maintain_aspect: If True, pad to square before resizing to avoid distortion

    Returns:
        32x32 image
    """
    if maintain_aspect:
        # Extend to square in appropriate dimension
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if len(image.shape) == 3:
            square = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
        else:
            square = np.zeros((max_dim, max_dim), dtype=image.dtype)

        # Center the image
        h_offset = (max_dim - h) // 2
        w_offset = (max_dim - w) // 2
        square[h_offset:h_offset + h, w_offset:w_offset + w] = image

        image = square

    # Resize to 32x32
    return cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)


def extract_patches(image: np.ndarray, patch_size: Tuple[int, int] = (8, 8),
                   stride: int = 1, normalize: bool = True) -> np.ndarray:
    """
    Extract patches from an image.

    Args:
        image: Input image (H, W) or (H, W, C)
        patch_size: Size of patches to extract (height, width)
        stride: Stride for patch extraction
        normalize: If True, normalize each patch to zero mean and unit variance

    Returns:
        Array of patches, shape (N, patch_h * patch_w) or (N, patch_h * patch_w * C)
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        patch_h, patch_w = patch_size
        patch_dim = patch_h * patch_w * c
    else:
        h, w = image.shape
        c = 1
        patch_h, patch_w = patch_size
        patch_dim = patch_h * patch_w

    patches = []

    for i in range(0, h - patch_h + 1, stride):
        for j in range(0, w - patch_w + 1, stride):
            patch = image[i:i + patch_h, j:j + patch_w]
            patch_flat = patch.flatten()

            if normalize:
                mean = patch_flat.mean()
                std = patch_flat.std()
                if std > 1e-5:  # Avoid division by zero
                    patch_flat = (patch_flat - mean) / std
                else:
                    patch_flat = patch_flat - mean

            patches.append(patch_flat)

    return np.array(patches)


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalize a patch to zero mean and unit variance.

    Args:
        patch: Input patch (flattened or 2D)

    Returns:
        Normalized patch
    """
    patch = patch.astype(np.float32)
    mean = patch.mean()
    std = patch.std()

    if std > 1e-5:
        return (patch - mean) / std
    else:
        return patch - mean


def convolve2d(image: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """
    Perform 2D convolution.

    Args:
        image: Input image
        kernel: Convolution kernel
        mode: Convolution mode ('valid', 'same', 'full')

    Returns:
        Convolved image
    """
    from scipy.signal import convolve2d as scipy_convolve2d
    return scipy_convolve2d(image, kernel, mode=mode)


def draw_bounding_boxes(image: np.ndarray, bboxes: list, labels: list = None,
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input image (RGB)
        bboxes: List of bounding boxes, each as dict with 'left', 'top', 'width', 'height'
                or as tuple (left, top, width, height)
        labels: Optional list of labels to display
        color: Box color in RGB
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    image_copy = image.copy()

    for idx, bbox in enumerate(bboxes):
        if isinstance(bbox, dict):
            left = int(bbox['left'])
            top = int(bbox['top'])
            width = int(bbox['width'])
            height = int(bbox['height'])
        else:
            left, top, width, height = map(int, bbox)

        # Draw rectangle
        cv2.rectangle(image_copy, (left, top), (left + width, top + height),
                     color, thickness)

        # Draw label if provided
        if labels is not None and idx < len(labels):
            label = str(labels[idx])
            # Put text above the box
            text_pos = (left, max(top - 5, 10))
            cv2.putText(image_copy, label, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_copy


def visualize_predictions(image: np.ndarray, predictions: list,
                         bboxes: list = None, title: str = "Predictions") -> np.ndarray:
    """
    Visualize predictions on an image.

    Args:
        image: Input image
        predictions: List of predicted digit sequences or single sequence
        bboxes: Optional bounding boxes for each digit
        title: Title for display

    Returns:
        Annotated image
    """
    result = image.copy()

    # If predictions is a list of digits, concatenate them
    if isinstance(predictions, list):
        pred_text = ''.join(str(p) for p in predictions)
    else:
        pred_text = str(predictions)

    # Draw bounding boxes if provided
    if bboxes is not None:
        result = draw_bounding_boxes(result, bboxes, labels=predictions)

    # Add prediction text at the bottom
    h, w = result.shape[:2]
    cv2.putText(result, f"Predicted: {pred_text}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return result


def compute_vertical_projection(image: np.ndarray) -> np.ndarray:
    """
    Compute vertical projection profile of an image.

    Args:
        image: Grayscale image

    Returns:
        Vertical projection (sum of pixel intensities along each column)
    """
    if len(image.shape) == 3:
        image = rgb_to_gray(image)

    return np.sum(image, axis=0)
