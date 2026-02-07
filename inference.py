#!/usr/bin/env python3
"""
Inference script for SVHN K-means recognition system.

This script takes an image as input, recognizes the house number,
and outputs an image with bounding boxes and predicted digits.
"""

import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.svhn_kmeans import SVHNRecognitionPipeline, SVHNDataLoader
from src.svhn_kmeans.utils import draw_bounding_boxes


def load_image(image_path: str) -> np.ndarray:
    """Load image from file."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def visualize_result(image: np.ndarray, digits: list, bboxes: list,
                    save_path: str = None, show: bool = True):
    """
    Visualize recognition result with bounding boxes.

    Args:
        image: Input image
        digits: Predicted digits
        bboxes: Bounding boxes for each digit
        save_path: Path to save output image
        show: Whether to display the image
    """
    # Draw bounding boxes
    result = draw_bounding_boxes(image, bboxes, labels=digits,
                                 color=(0, 255, 0), thickness=2)

    # Add prediction text
    pred_text = ''.join(str(d) for d in digits)
    import cv2
    h, w = result.shape[:2]
    cv2.putText(result, f"Predicted: {pred_text}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display and/or save
    if show or save_path:
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.axis('off')
        plt.title(f"Predicted House Number: {pred_text}", fontsize=16)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Result saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return result


def process_single_image(pipeline: SVHNRecognitionPipeline,
                        image_path: str,
                        output_path: str = None,
                        show: bool = True):
    """
    Process a single image.

    Args:
        pipeline: Trained pipeline
        image_path: Path to input image
        output_path: Path to save output (optional)
        show: Whether to display result
    """
    print(f"\nProcessing: {image_path}")

    # Load image
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")

    # Recognize
    result = pipeline.process_image(image)

    digits = result['digits']
    bboxes = result['bboxes']
    confidence = result['confidence']

    # Print results
    house_number = ''.join(str(d) for d in digits)
    print(f"Predicted house number: {house_number}")
    print(f"Number of digits: {len(digits)}")
    print(f"Confidence score: {confidence:.2f}")

    # Visualize
    if output_path or show:
        visualize_result(image, digits, bboxes, save_path=output_path, show=show)

    return digits, bboxes


def process_from_dataset(pipeline: SVHNRecognitionPipeline,
                        data_loader: SVHNDataLoader,
                        image_indices: list,
                        output_dir: str = None,
                        show: bool = True):
    """
    Process images from the dataset.

    Args:
        pipeline: Trained pipeline
        data_loader: Data loader
        image_indices: List of image indices to process
        output_dir: Directory to save outputs
        show: Whether to display results
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for idx in image_indices:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset image {idx}")
        print('=' * 60)

        # Get full house number crop
        crop, gt_bboxes = data_loader.get_full_number_crop(idx)

        # Ground truth
        gt_digits = [bbox['label'] if bbox['label'] < 10 else 0 for bbox in gt_bboxes]
        gt_number = ''.join(str(d) for d in gt_digits)
        print(f"Ground truth: {gt_number}")

        # Predict
        result = pipeline.process_image(crop)
        pred_digits = result['digits']
        pred_bboxes = result['bboxes']
        pred_number = ''.join(str(d) for d in pred_digits)

        print(f"Predicted: {pred_number}")
        print(f"Match: {pred_number == gt_number}")

        # Visualize
        output_path = None
        if output_dir:
            output_path = os.path.join(output_dir, f"result_{idx}.png")

        visualize_result(crop, pred_digits, pred_bboxes,
                        save_path=output_path, show=show)


def main():
    parser = argparse.ArgumentParser(
        description='SVHN digit recognition inference')

    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory containing trained model')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to SVHN dataset directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--indices', type=int, nargs='+', default=[1, 10, 100],
                       help='Image indices to process from dataset')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--no_display', action='store_true',
                       help='Don\'t display images (only save)')
    parser.add_argument('--beam_width', type=int, default=10,
                       help='Beam width for recognition')

    args = parser.parse_args()

    print("=" * 70)
    print("SVHN K-means Recognition System - Inference")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    pipeline = SVHNRecognitionPipeline(beam_width=args.beam_width)
    pipeline.load(args.model_dir)

    # Process image(s)
    if args.image:
        # Process single image file
        output_path = os.path.join(args.output, 'result.png')
        os.makedirs(args.output, exist_ok=True)

        process_single_image(
            pipeline,
            args.image,
            output_path=output_path,
            show=not args.no_display
        )

    else:
        # Process images from dataset
        print(f"\nLoading dataset from {args.data_dir} ({args.split} split)...")
        data_loader = SVHNDataLoader(args.data_dir, split=args.split)

        process_from_dataset(
            pipeline,
            data_loader,
            image_indices=args.indices,
            output_dir=args.output,
            show=not args.no_display
        )

    print("\n" + "=" * 70)
    print("Inference complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
