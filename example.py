#!/usr/bin/env python3
"""
Simple example demonstrating the SVHN recognition system.

This script shows how to:
1. Load a trained model
2. Process an image from the dataset
3. Visualize the results with bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
from src.svhn_kmeans import SVHNRecognitionPipeline, SVHNDataLoader
from src.svhn_kmeans.utils import draw_bounding_boxes


def main():
    print("SVHN Recognition System - Quick Example")
    print("=" * 60)

    # Configuration
    model_dir = 'models'
    data_dir = 'data'
    image_index = 1  # Try image 1 from test set

    # Load the trained pipeline
    print("\n1. Loading trained model...")
    try:
        pipeline = SVHNRecognitionPipeline()
        pipeline.load(model_dir)
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model: {e}")
        print("   Please train the model first using: python train.py")
        return

    # Load dataset
    print("\n2. Loading test dataset...")
    try:
        test_loader = SVHNDataLoader(data_dir, split='test')
        print(f"   Dataset loaded: {len(test_loader)} images")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("   Please ensure the data directory exists with SVHN dataset")
        return

    # Get a test image (full house number crop)
    print(f"\n3. Processing test image #{image_index}...")
    crop, gt_bboxes = test_loader.get_full_number_crop(image_index)

    # Get ground truth
    gt_digits = [bbox['label'] if bbox['label'] < 10 else 0 for bbox in gt_bboxes]
    gt_number = ''.join(str(d) for d in gt_digits)
    print(f"   Ground truth: {gt_number}")
    print(f"   Image shape: {crop.shape}")

    # Recognize the digits
    print("\n4. Running recognition...")
    result = pipeline.process_image(crop)

    pred_digits = result['digits']
    pred_bboxes = result['bboxes']
    pred_number = ''.join(str(d) for d in pred_digits)

    print(f"   Predicted: {pred_number}")
    print(f"   Number of digits detected: {len(pred_digits)}")
    print(f"   Confidence score: {result['confidence']:.2f}")
    print(f"   Match: {'✓' if pred_number == gt_number else '✗'}")

    # Visualize results
    print("\n5. Visualizing results...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original image with ground truth
    gt_img = draw_bounding_boxes(crop, gt_bboxes, labels=gt_digits,
                                 color=(255, 0, 0), thickness=2)
    axes[0].imshow(gt_img)
    axes[0].set_title(f"Ground Truth: {gt_number}", fontsize=14, color='red')
    axes[0].axis('off')

    # Prediction with bounding boxes
    pred_img = draw_bounding_boxes(crop, pred_bboxes, labels=pred_digits,
                                   color=(0, 255, 0), thickness=2)
    axes[1].imshow(pred_img)
    axes[1].set_title(f"Predicted: {pred_number}", fontsize=14, color='green')
    axes[1].axis('off')

    plt.suptitle("SVHN Recognition System - Example", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nTry different images by changing 'image_index' in the script.")
    print("=" * 60)


if __name__ == '__main__':
    main()
