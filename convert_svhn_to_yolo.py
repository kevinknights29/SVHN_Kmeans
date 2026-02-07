#!/usr/bin/env python3
"""
Convert SVHN dataset to YOLO format for house number detection.

This script:
1. Loads SVHN digitStruct.mat annotations
2. Computes house number bounding boxes from individual digit bboxes
3. Converts to YOLO format (normalized [cx, cy, w, h])
4. Creates YOLO directory structure with images/ and labels/
5. Generates dataset.yaml configuration file

Usage:
    python convert_svhn_to_yolo.py --data_dir data --output_dir data_yolo
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import numpy as np
from PIL import Image

from src.svhn_kmeans import SVHNDataLoader


def compute_house_number_bbox(bboxes: List[Dict]) -> Tuple[int, int, int, int]:
    """
    Compute bounding box for entire house number from individual digit bboxes.

    Args:
        bboxes: List of digit bounding boxes with keys: left, top, width, height

    Returns:
        Tuple of (left, top, width, height) for house number bbox
    """
    if not bboxes:
        return (0, 0, 0, 0)

    # Find min/max coordinates across all digits
    left_min = min(bbox['left'] for bbox in bboxes)
    top_min = min(bbox['top'] for bbox in bboxes)
    right_max = max(bbox['left'] + bbox['width'] for bbox in bboxes)
    bottom_max = max(bbox['top'] + bbox['height'] for bbox in bboxes)

    # Compute house number bbox
    left = left_min
    top = top_min
    width = right_max - left_min
    height = bottom_max - top_min

    return (left, top, width, height)


def bbox_to_yolo_format(bbox: Tuple[int, int, int, int],
                        img_width: int,
                        img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bbox from (left, top, width, height) to YOLO format (cx, cy, w, h).

    All coordinates are normalized to [0, 1].

    Args:
        bbox: Bounding box (left, top, width, height)
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (cx, cy, w, h) normalized to [0, 1]
    """
    left, top, width, height = bbox

    # Convert to center coordinates
    cx = (left + width / 2) / img_width
    cy = (top + height / 2) / img_height
    w = width / img_width
    h = height / img_height

    # Clip to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (cx, cy, w, h)


def convert_split(data_loader: SVHNDataLoader,
                 output_dir: Path,
                 split_name: str,
                 max_samples: int = None,
                 verbose: bool = True) -> int:
    """
    Convert a single dataset split to YOLO format.

    Args:
        data_loader: SVHN data loader
        output_dir: Output directory for this split
        split_name: Name of split ('train', 'val', or 'test')
        max_samples: Maximum number of samples to convert (None for all)
        verbose: Print progress

    Returns:
        Number of samples converted
    """
    # Create subdirectories
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get number of samples
    n_samples = len(data_loader)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    if verbose:
        print(f"Converting {split_name} split: {n_samples} samples")

    converted = 0
    skipped = 0

    for idx in range(n_samples):
        try:
            # Get full house number crop with bounding boxes
            crop, bboxes = data_loader.get_full_number_crop(idx)

            if len(bboxes) == 0:
                if verbose and idx % 1000 == 0:
                    print(f"  Warning: Image {idx} has no bboxes, skipping")
                skipped += 1
                continue

            # Get image dimensions
            img_height, img_width = crop.shape[:2]

            # Compute house number bbox
            house_bbox = compute_house_number_bbox(bboxes)

            # Convert to YOLO format
            cx, cy, w, h = bbox_to_yolo_format(house_bbox, img_width, img_height)

            # Validate bbox
            if w <= 0 or h <= 0:
                if verbose and idx % 1000 == 0:
                    print(f"  Warning: Image {idx} has invalid bbox, skipping")
                skipped += 1
                continue

            # Save image
            image_filename = f"{idx:05d}.png"
            image_path = images_dir / image_filename
            Image.fromarray(crop).save(image_path)

            # Save label (class 0 for house number)
            label_filename = f"{idx:05d}.txt"
            label_path = labels_dir / label_filename
            with open(label_path, 'w') as f:
                # YOLO format: class cx cy w h
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            converted += 1

            if verbose and (converted % 1000 == 0 or converted == n_samples):
                print(f"  Converted: {converted}/{n_samples}")

        except Exception as e:
            if verbose:
                print(f"  Error processing image {idx}: {e}")
            skipped += 1
            continue

    if verbose:
        print(f"  Completed: {converted} converted, {skipped} skipped")

    return converted


def create_dataset_yaml(output_dir: Path,
                       train_count: int,
                       val_count: int,
                       test_count: int):
    """
    Create YOLO dataset.yaml configuration file.

    Args:
        output_dir: Root output directory
        train_count: Number of training samples
        val_count: Number of validation samples
        test_count: Number of test samples
    """
    dataset_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes
        'names': ['house_number'],
        'stats': {
            'train_samples': train_count,
            'val_samples': val_count,
            'test_samples': test_count,
        }
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"\nDataset configuration saved: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert SVHN to YOLO format')

    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to SVHN data directory')
    parser.add_argument('--output_dir', type=str, default='data_yolo',
                       help='Output directory for YOLO dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for quick testing)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Fraction of train data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for train/val split')

    args = parser.parse_args()

    print("=" * 70)
    print("SVHN to YOLO Format Converter")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Validation split: {args.val_split:.1%}")
    if args.max_samples:
        print(f"  Max samples per split: {args.max_samples}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SVHN data
    print(f"\nLoading SVHN train data...")
    train_loader = SVHNDataLoader(args.data_dir, split='train')
    print(f"  Found {len(train_loader)} training images")

    print(f"\nLoading SVHN test data...")
    test_loader = SVHNDataLoader(args.data_dir, split='test')
    print(f"  Found {len(test_loader)} test images")

    # Split train into train/val
    np.random.seed(args.seed)
    n_train = len(train_loader)
    indices = np.random.permutation(n_train)
    n_val = int(n_train * args.val_split)
    val_indices = set(indices[:n_val].tolist())

    print(f"\nSplit statistics:")
    print(f"  Train: {n_train - n_val} images")
    print(f"  Val: {n_val} images")
    print(f"  Test: {len(test_loader)} images")

    # Convert train split
    print(f"\n" + "=" * 70)
    print("Converting training data...")
    print("=" * 70)

    # Create temporary full train loader
    class FilteredDataLoader:
        """Wrapper to filter data loader by indices."""
        def __init__(self, loader, indices, exclude=False):
            self.loader = loader
            self.indices = set(indices)
            self.exclude = exclude
            # Build valid index mapping
            self.valid_indices = []
            for i in range(len(loader)):
                should_include = (i in self.indices) != self.exclude
                if should_include:
                    self.valid_indices.append(i)

        def __len__(self):
            return len(self.valid_indices)

        def get_full_number_crop(self, idx):
            real_idx = self.valid_indices[idx]
            return self.loader.get_full_number_crop(real_idx)

    train_filtered = FilteredDataLoader(train_loader, val_indices, exclude=True)
    max_train = args.max_samples if args.max_samples else len(train_filtered)
    train_count = convert_split(
        train_filtered,
        output_dir / 'train',
        'train',
        max_samples=max_train
    )

    # Convert val split
    print(f"\n" + "=" * 70)
    print("Converting validation data...")
    print("=" * 70)

    val_filtered = FilteredDataLoader(train_loader, val_indices, exclude=False)
    max_val = args.max_samples if args.max_samples else len(val_filtered)
    val_count = convert_split(
        val_filtered,
        output_dir / 'val',
        'val',
        max_samples=max_val
    )

    # Convert test split
    print(f"\n" + "=" * 70)
    print("Converting test data...")
    print("=" * 70)

    max_test = args.max_samples if args.max_samples else len(test_loader)
    test_count = convert_split(
        test_loader,
        output_dir / 'test',
        'test',
        max_samples=max_test
    )

    # Create dataset.yaml
    create_dataset_yaml(output_dir, train_count, val_count, test_count)

    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/ ({train_count} images)")
    print(f"    │   └── labels/ ({train_count} labels)")
    print(f"    ├── val/")
    print(f"    │   ├── images/ ({val_count} images)")
    print(f"    │   └── labels/ ({val_count} labels)")
    print(f"    ├── test/")
    print(f"    │   ├── images/ ({test_count} images)")
    print(f"    │   └── labels/ ({test_count} labels)")
    print(f"    └── dataset.yaml")
    print(f"\nYou can now train YOLO with:")
    print(f"  python train_yolo.py --data {output_dir / 'dataset.yaml'}")


if __name__ == '__main__':
    main()
