#!/usr/bin/env python3
"""
End-to-end YOLO + CNN pipeline for SVHN digit recognition.

This script combines:
1. YOLOv8 for house number detection (bounding box localization)
2. CNN for digit sequence recognition (multi-digit classification)

Usage:
    # Process test images
    python inference_yolo.py --yolo_model models_yolo/train/weights/best.pt \\
                             --cnn_model models_cnn --indices 1 10 100

    # Process custom image
    python inference_yolo.py --yolo_model models_yolo/train/weights/best.pt \\
                             --cnn_model models_cnn --image path/to/image.png
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict
import time

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.svhn_kmeans import SVHNDataLoader


def check_ultralytics():
    """Check if ultralytics is installed."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        print("=" * 70)
        print("ERROR: ultralytics package not found")
        print("=" * 70)
        print("\nPlease install ultralytics:")
        print("  pip install ultralytics")
        print("\nOr with uv:")
        print("  uv pip install ultralytics")
        return False


def load_yolo_model(model_path: str):
    """Load trained YOLO model."""
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    print(f"  ✓ YOLO model loaded")
    return model


def load_cnn_model(model_dir: str, device: str):
    """Load trained CNN model."""
    from src.svhn_kmeans.cnn_model import create_model

    model_path = Path(model_dir) / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found: {model_path}")

    print(f"Loading CNN model from {model_dir}...")

    # Create model (using same architecture as training)
    model = create_model(
        model_type='enhanced',
        input_channels=3,
        use_stn=True
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  ✓ CNN model loaded")
    return model


def detect_house_numbers(yolo_model, image: np.ndarray,
                         conf_threshold: float = 0.25) -> List[Dict]:
    """
    Detect house numbers in image using YOLO.

    Args:
        yolo_model: Trained YOLO model
        image: Input image (H, W, 3) in RGB format
        conf_threshold: Confidence threshold for detections

    Returns:
        List of detections with keys: bbox, confidence
    """
    # Run YOLO inference
    results = yolo_model(image, conf=conf_threshold, verbose=False)

    detections = []

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bbox coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])

            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence
            })

    return detections


def recognize_digits(cnn_model, crop: np.ndarray, device: str) -> Tuple[List[int], np.ndarray]:
    """
    Recognize digit sequence using CNN.

    Args:
        cnn_model: Trained CNN model
        crop: Cropped house number image (H, W, 3) in RGB format
        device: Device for inference

    Returns:
        Tuple of (digit_list, confidences)
    """
    # Resize to CNN input size (32x32)
    from src.svhn_kmeans.utils import resize_to_32x32
    crop_resized = resize_to_32x32(crop, maintain_aspect=True)

    # Normalize to [0, 1] (CNN uses simple normalization, not ImageNet)
    crop_normalized = crop_resized.astype(np.float32) / 255.0
    image = torch.from_numpy(crop_normalized).permute(2, 0, 1).float()

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = cnn_model(image)

    # Parse outputs
    # outputs is a dict with keys: 'num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC'
    num_logits = outputs['num']  # [1, 5] - number of digits (0-4)
    dig1_logits = outputs['dig1']  # [1, 11] - first digit (0-9 + no-digit)
    dig2_logits = outputs['dig2']  # [1, 11] - second digit
    dig3_logits = outputs['dig3']  # [1, 11] - third digit
    dig4_logits = outputs['dig4']  # [1, 11] - fourth digit

    # Get number of digits
    num_digits = torch.argmax(num_logits, dim=1).item()

    # Get digits
    digits = []
    confidences = []

    if num_digits >= 1:
        dig1 = torch.argmax(dig1_logits, dim=1).item()
        dig1_conf = torch.softmax(dig1_logits, dim=1)[0, dig1].item()
        if dig1 < 10:  # Not no-digit
            digits.append(dig1)
            confidences.append(dig1_conf)

    if num_digits >= 2:
        dig2 = torch.argmax(dig2_logits, dim=1).item()
        dig2_conf = torch.softmax(dig2_logits, dim=1)[0, dig2].item()
        if dig2 < 10:
            digits.append(dig2)
            confidences.append(dig2_conf)

    if num_digits >= 3:
        dig3 = torch.argmax(dig3_logits, dim=1).item()
        dig3_conf = torch.softmax(dig3_logits, dim=1)[0, dig3].item()
        if dig3 < 10:
            digits.append(dig3)
            confidences.append(dig3_conf)

    if num_digits >= 4:
        dig4 = torch.argmax(dig4_logits, dim=1).item()
        dig4_conf = torch.softmax(dig4_logits, dim=1)[0, dig4].item()
        if dig4 < 10:
            digits.append(dig4)
            confidences.append(dig4_conf)

    return digits, np.array(confidences)


def process_image(yolo_model, cnn_model, image: np.ndarray,
                 device: str, conf_threshold: float = 0.25) -> Dict:
    """
    End-to-end processing: detection + recognition.

    Args:
        yolo_model: Trained YOLO model
        cnn_model: Trained CNN model
        image: Input image (H, W, 3) in RGB format
        device: Device for inference
        conf_threshold: Confidence threshold for YOLO

    Returns:
        Dictionary with keys: detections, image_with_boxes
    """
    # Step 1: Detect house numbers
    detections = detect_house_numbers(yolo_model, image, conf_threshold)

    # Step 2: Recognize digits for each detection
    results = []

    for det in detections:
        x1, y1, x2, y2 = det['bbox']

        # Crop detected region
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Recognize digits
        digits, confidences = recognize_digits(cnn_model, crop, device)

        results.append({
            'bbox': det['bbox'],
            'yolo_confidence': det['confidence'],
            'digits': digits,
            'digit_confidences': confidences,
            'sequence': ''.join(str(d) for d in digits)
        })

    # Step 3: Visualize
    image_vis = image.copy()

    for result in results:
        x1, y1, x2, y2 = result['bbox']

        # Draw bounding box
        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{result['sequence']} ({result['yolo_confidence']:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image_vis, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image_vis, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return {
        'results': results,
        'image_with_boxes': image_vis
    }


def main():
    parser = argparse.ArgumentParser(description='YOLO + CNN inference for SVHN')

    # Model arguments
    parser.add_argument('--yolo_model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--cnn_model', type=str, required=True,
                       help='Path to trained CNN model directory')

    # Input arguments
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to SVHN dataset (for processing test images)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--indices', type=int, nargs='+', default=[1, 10, 100],
                       help='Image indices to process from dataset')

    # Inference arguments
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for YOLO detection')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')

    # Output arguments
    parser.add_argument('--output', type=str, default='output_yolo',
                       help='Output directory for results')
    parser.add_argument('--no_display', action='store_true',
                       help='Don\'t display images, only save')

    args = parser.parse_args()

    # Check for ultralytics
    if not check_ultralytics():
        return

    print("=" * 70)
    print("YOLO + CNN Pipeline for SVHN Digit Recognition")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  YOLO model: {args.yolo_model}")
    print(f"  CNN model: {args.cnn_model}")
    print(f"  Device: {args.device}")
    print(f"  Confidence threshold: {args.conf_threshold}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    yolo_model = load_yolo_model(args.yolo_model)
    cnn_model = load_cnn_model(args.cnn_model, args.device)

    # Process images
    if args.image:
        # Process single custom image
        print(f"\nProcessing custom image: {args.image}")

        image_path = Path(args.image)
        if not image_path.exists():
            print(f"✗ Error: Image not found: {args.image}")
            return

        image = np.array(Image.open(image_path).convert('RGB'))

        start_time = time.time()
        result = process_image(yolo_model, cnn_model, image, args.device, args.conf_threshold)
        inference_time = time.time() - start_time

        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Detections: {len(result['results'])}")

        for i, res in enumerate(result['results']):
            print(f"    {i+1}. Sequence: {res['sequence']}, "
                  f"YOLO conf: {res['yolo_confidence']:.3f}, "
                  f"Avg digit conf: {res['digit_confidences'].mean():.3f}")

        # Save visualization
        output_path = output_dir / f"{image_path.stem}_result.png"
        Image.fromarray(result['image_with_boxes']).save(output_path)
        print(f"  Saved: {output_path}")

        # Display
        if not args.no_display:
            plt.figure(figsize=(12, 8))
            plt.imshow(result['image_with_boxes'])
            plt.axis('off')
            plt.title(f"Detections: {', '.join(r['sequence'] for r in result['results'])}")
            plt.tight_layout()
            plt.show()

    else:
        # Process images from dataset
        print(f"\nLoading SVHN {args.split} data from {args.data_dir}...")
        data_loader = SVHNDataLoader(args.data_dir, split=args.split)
        print(f"  Found {len(data_loader)} images")

        print(f"\nProcessing {len(args.indices)} images...")

        total_time = 0

        for idx in args.indices:
            if idx >= len(data_loader):
                print(f"\n  Warning: Index {idx} out of range, skipping")
                continue

            print(f"\nImage {idx}:")

            # Get image and ground truth
            crop, bboxes = data_loader.get_full_number_crop(idx)
            gt_digits = [bbox['label'] % 10 for bbox in bboxes]
            gt_sequence = ''.join(str(d) for d in gt_digits)

            print(f"  Ground truth: {gt_sequence}")

            # Process
            start_time = time.time()
            result = process_image(yolo_model, cnn_model, crop, args.device, args.conf_threshold)
            inference_time = time.time() - start_time
            total_time += inference_time

            print(f"  Inference time: {inference_time:.3f}s")
            print(f"  Detections: {len(result['results'])}")

            for i, res in enumerate(result['results']):
                match = "✓" if res['sequence'] == gt_sequence else "✗"
                print(f"    {match} {i+1}. Sequence: {res['sequence']}, "
                      f"YOLO conf: {res['yolo_confidence']:.3f}, "
                      f"Avg digit conf: {res['digit_confidences'].mean():.3f}")

            # Save visualization
            output_path = output_dir / f"test_{idx}_result.png"
            Image.fromarray(result['image_with_boxes']).save(output_path)
            print(f"  Saved: {output_path}")

        avg_time = total_time / len(args.indices)
        print(f"\nAverage inference time: {avg_time:.3f}s")
        print(f"Results saved to: {output_dir}")

        # Display last image
        if not args.no_display and len(args.indices) > 0:
            plt.figure(figsize=(12, 8))
            plt.imshow(result['image_with_boxes'])
            plt.axis('off')
            plt.title(f"GT: {gt_sequence} | Pred: {', '.join(r['sequence'] for r in result['results'])}")
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    main()
