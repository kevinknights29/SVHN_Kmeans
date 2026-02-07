"""
Data loader for SVHN dataset.
Handles loading of images and annotations from .mat files.
Supports both MATLAB v5-v7.2 (scipy) and v7.3 (h5py) formats.
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SVHNDataLoader:
    """Loads and processes SVHN dataset from .mat files."""

    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize data loader.

        Args:
            data_dir: Path to data directory containing train/test folders
            split: Either 'train' or 'test'
        """
        self.data_dir = os.path.join(data_dir, split)
        self.split = split
        self.digit_struct_path = os.path.join(self.data_dir, 'digitStruct.mat')

        # Load digit structure metadata
        self.digit_struct = self._load_digit_struct()

    def _load_digit_struct(self) -> List[Dict]:
        """
        Load digitStruct.mat file containing bounding box annotations.
        Supports both MATLAB v5-v7.2 (scipy) and v7.3 (h5py) formats.

        Returns:
            List of dictionaries containing image metadata and bbox info
        """
        # Try to determine format and load accordingly
        try:
            # First try with h5py (for MATLAB v7.3)
            if HAS_H5PY:
                return self._load_digit_struct_h5py()
        except:
            pass

        # Fall back to scipy (for MATLAB v5-v7.2)
        if HAS_SCIPY:
            try:
                return self._load_digit_struct_scipy()
            except NotImplementedError:
                # scipy can't handle v7.3, try h5py
                if HAS_H5PY:
                    return self._load_digit_struct_h5py()
                else:
                    raise ImportError("h5py is required for MATLAB v7.3 files. Install with: pip install h5py")

        raise ImportError("Either scipy or h5py is required to load .mat files")

    def _load_digit_struct_h5py(self) -> List[Dict]:
        """Load using h5py (for MATLAB v7.3 format)."""
        data = []

        with h5py.File(self.digit_struct_path, 'r') as f:
            digitStruct = f['digitStruct']
            n_images = digitStruct['name'].shape[0]

            for i in range(n_images):
                item = self._get_bbox_h5py(f, digitStruct, i)
                data.append(item)

        return data

    def _load_digit_struct_scipy(self) -> List[Dict]:
        """Load using scipy (for MATLAB v5-v7.2 format)."""
        mat = loadmat(self.digit_struct_path)
        digit_struct = mat['digitStruct']

        data = []
        for i in range(len(digit_struct[0])):
            item = self._get_bbox_scipy(digit_struct, i)
            data.append(item)

        return data

    def _get_bbox_h5py(self, f, digitStruct, index: int) -> Dict:
        """
        Extract bounding box information using h5py.

        Args:
            f: HDF5 file object
            digitStruct: digitStruct dataset
            index: Index of the image

        Returns:
            Dictionary with filename and bbox information
        """
        # Get filename
        name_ref = digitStruct['name'][index, 0]
        filename = ''.join(chr(c[0]) for c in f[name_ref][:])

        # Get bbox data
        bbox_ref = digitStruct['bbox'][index, 0]
        bbox = f[bbox_ref]

        # Helper to extract attribute
        def get_attr(attr_name):
            attr = bbox[attr_name]
            if attr.shape[0] == 1:
                # Single digit
                return [int(attr[0, 0])]
            else:
                # Multiple digits
                values = []
                for ref in attr[:, 0]:
                    values.append(int(f[ref][0, 0]))
                return values

        # Extract bounding boxes for all digits
        labels = get_attr('label')
        lefts = get_attr('left')
        tops = get_attr('top')
        widths = get_attr('width')
        heights = get_attr('height')

        bboxes = []
        for label, left, top, width, height in zip(labels, lefts, tops, widths, heights):
            bboxes.append({
                'label': int(label),  # 1-9 for digits 1-9, 10 for digit 0
                'left': int(left),
                'top': int(top),
                'width': int(width),
                'height': int(height)
            })

        return {
            'filename': filename,
            'bboxes': bboxes
        }

    def _get_bbox_scipy(self, digit_struct, index: int) -> Dict:
        """
        Extract bounding box information using scipy.

        Args:
            digit_struct: The loaded .mat structure
            index: Index of the image

        Returns:
            Dictionary with filename and bbox information
        """
        # Get filename
        filename_obj = digit_struct[0, index]['name'][0, 0]
        filename = ''.join(chr(c[0]) for c in filename_obj)

        # Get bbox data
        bbox_obj = digit_struct[0, index]['bbox'][0, 0]

        # Helper function to extract values
        def get_attr(obj, attr_name):
            attr = obj[attr_name]
            if attr.size == 1:
                return [int(attr[0, 0])]
            else:
                return [int(attr[i, 0][0, 0]) for i in range(attr.size)]

        # Extract bounding boxes for all digits in the image
        labels = get_attr(bbox_obj, 'label')
        lefts = get_attr(bbox_obj, 'left')
        tops = get_attr(bbox_obj, 'top')
        widths = get_attr(bbox_obj, 'width')
        heights = get_attr(bbox_obj, 'height')

        bboxes = []
        for label, left, top, width, height in zip(labels, lefts, tops, widths, heights):
            bboxes.append({
                'label': label,  # 1-9 for digits 1-9, 10 for digit 0
                'left': left,
                'top': top,
                'width': width,
                'height': height
            })

        return {
            'filename': filename,
            'bboxes': bboxes
        }

    def load_image(self, index: int) -> np.ndarray:
        """
        Load image at given index.

        Args:
            index: Index of the image to load

        Returns:
            Image as numpy array (RGB)
        """
        filename = self.digit_struct[index]['filename']
        img_path = os.path.join(self.data_dir, filename)
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def get_digit_crops(self, index: int) -> List[Tuple[np.ndarray, int]]:
        """
        Get individual digit crops from an image.

        Args:
            index: Index of the image

        Returns:
            List of tuples (cropped_image, label)
        """
        img = self.load_image(index)
        bboxes = self.digit_struct[index]['bboxes']

        crops = []
        for bbox in bboxes:
            left = max(0, bbox['left'])
            top = max(0, bbox['top'])
            right = min(img.shape[1], left + bbox['width'])
            bottom = min(img.shape[0], top + bbox['height'])

            crop = img[top:bottom, left:right]
            label = bbox['label']

            crops.append((crop, label))

        return crops

    def get_full_number_crop(self, index: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get the full house number region (bounding box around all digits).

        Args:
            index: Index of the image

        Returns:
            Tuple of (cropped image, list of bbox dicts with relative coordinates)
        """
        img = self.load_image(index)
        bboxes = self.digit_struct[index]['bboxes']

        if not bboxes:
            return img, []

        # Find bounding box that encompasses all digits
        min_left = min(bbox['left'] for bbox in bboxes)
        min_top = min(bbox['top'] for bbox in bboxes)
        max_right = max(bbox['left'] + bbox['width'] for bbox in bboxes)
        max_bottom = max(bbox['top'] + bbox['height'] for bbox in bboxes)

        # Clip to image boundaries
        min_left = max(0, min_left)
        min_top = max(0, min_top)
        max_right = min(img.shape[1], max_right)
        max_bottom = min(img.shape[0], max_bottom)

        # Crop the full number region
        crop = img[min_top:max_bottom, min_left:max_right]

        # Adjust bbox coordinates to be relative to the crop
        relative_bboxes = []
        for bbox in bboxes:
            relative_bboxes.append({
                'label': bbox['label'],
                'left': bbox['left'] - min_left,
                'top': bbox['top'] - min_top,
                'width': bbox['width'],
                'height': bbox['height']
            })

        return crop, relative_bboxes

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self.digit_struct)

    def __getitem__(self, index: int) -> Dict:
        """Get item at index."""
        return {
            'image': self.load_image(index),
            'metadata': self.digit_struct[index]
        }
