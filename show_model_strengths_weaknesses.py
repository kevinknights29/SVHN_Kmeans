#!/usr/bin/env python3
"""
Demonstrate strengths and weaknesses of all trained models.

This script runs inference on the test dataset and finds one correct
and one incorrect prediction for each model to showcase their capabilities
and limitations.

Models evaluated:
- K-means + SVM
- CNN with STN
- Character Query Transformer (CQT)
- YOLO + CNN

Usage:
    python show_model_strengths_weaknesses.py --data_dir data --output_dir model_analysis
"""

import argparse
import os
from pathlib import Path
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Global model cache to avoid reloading models repeatedly
_MODEL_CACHE = {}


def get_ground_truth_sequence(bboxes: List[Dict]) -> str:
    """
    Extract ground truth digit sequence from bboxes.

    Args:
        bboxes: List of bbox dictionaries with 'label' and 'left' keys

    Returns:
        String of digits sorted left to right
    """
    # Sort by left coordinate (left to right)
    sorted_bboxes = sorted(bboxes, key=lambda b: b['left'])

    # Convert labels (1-9 for 1-9, 10 for 0)
    digits = []
    for bbox in sorted_bboxes:
        label = bbox['label']
        digit = 0 if label == 10 else label
        digits.append(digit)

    return ''.join(str(d) for d in digits)


def run_kmeans_on_sample(image: np.ndarray, bboxes: List[Dict], model_dir: str, device: str = 'cpu', debug: bool = False) -> Optional[str]:
    """Run K-means + SVM inference on a single sample using full image."""
    try:
        from src.svhn_kmeans import SVHNRecognitionPipeline
        import cv2

        # Use cached model
        cache_key = f'kmeans_{model_dir}'

        if cache_key not in _MODEL_CACHE:
            pipeline = SVHNRecognitionPipeline()
            pipeline.load(model_dir)
            _MODEL_CACHE[cache_key] = pipeline
        else:
            pipeline = _MODEL_CACHE[cache_key]

        # K-means needs larger images to work properly
        # Resize small test images to a reasonable size
        min_size = 100
        h, w = image.shape[:2]

        if h < min_size or w < min_size:
            # Calculate scale to make the smaller dimension at least min_size
            scale = max(min_size / h, min_size / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            if debug:
                print(f"    K-means: Resizing from {image.shape} to {resized_image.shape}")
        else:
            resized_image = image

        if debug:
            print(f"    K-means: Processing image shape: {resized_image.shape}")

        # Run inference on full image (K-means does its own detection)
        result = pipeline.process_image(resized_image)
        digits = result['digits']
        sequence = ''.join(str(d) for d in digits)

        if debug:
            print(f"    K-means: Detected {len(digits)} digits: {digits}")
            print(f"    K-means: Sequence: '{sequence}'")

        return sequence if sequence else None
    except Exception as e:
        if debug:
            print(f"    K-means error: {e}")
            import traceback
            traceback.print_exc()
        return None


def run_cnn_on_sample(image: np.ndarray, model_dir: str, device: str = 'cuda') -> Optional[str]:
    """Run CNN inference on a single sample."""
    try:
        from src.svhn_kmeans.cnn_model import create_model
        from src.svhn_kmeans.utils import resize_to_32x32

        # Use cached model
        cache_key = f'cnn_{model_dir}'

        if cache_key not in _MODEL_CACHE:
            model_path = Path(model_dir) / 'best_model.pth'

            model = create_model(
                model_type='enhanced',
                input_channels=3,
                use_stn=True
            )

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            _MODEL_CACHE[cache_key] = model
        else:
            model = _MODEL_CACHE[cache_key]

        # Preprocess
        image_resized = resize_to_32x32(image, maintain_aspect=True)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Parse outputs
        num_digits = torch.argmax(outputs['num'], dim=1).item()

        digits = []
        if num_digits >= 1:
            dig1 = torch.argmax(outputs['dig1'], dim=1).item()
            if dig1 < 10:
                digits.append(dig1)
        if num_digits >= 2:
            dig2 = torch.argmax(outputs['dig2'], dim=1).item()
            if dig2 < 10:
                digits.append(dig2)
        if num_digits >= 3:
            dig3 = torch.argmax(outputs['dig3'], dim=1).item()
            if dig3 < 10:
                digits.append(dig3)
        if num_digits >= 4:
            dig4 = torch.argmax(outputs['dig4'], dim=1).item()
            if dig4 < 10:
                digits.append(dig4)

        sequence = ''.join(str(d) for d in digits)
        return sequence
    except Exception as e:
        return None


def run_cqt_on_sample(image: np.ndarray, model_dir: str, device: str = 'cuda') -> Optional[str]:
    """Run CQT inference on a single sample."""
    try:
        from src.svhn_kmeans.cqt_model import create_cqt_model

        # Use cached model
        cache_key = f'cqt_{model_dir}'

        if cache_key not in _MODEL_CACHE:
            model_path = Path(model_dir) / 'best_model.pth'

            model = create_cqt_model(
                num_queries=6,
                num_classes=11,
                d_model=256,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=1024,
                dropout=0.1,
                backbone='resnet50',
                pretrained_backbone=False
            )

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            _MODEL_CACHE[cache_key] = model
        else:
            model = _MODEL_CACHE[cache_key]

        # Preprocess
        image_resized = cv2.resize(image, (224, 224))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Parse outputs
        pred_logits = outputs['class_logits']
        pred_boxes = outputs['bbox_coords']

        probs = torch.softmax(pred_logits, dim=-1)[0]
        boxes = pred_boxes[0]

        # Filter predictions
        detections = []
        for i in range(probs.shape[0]):
            class_id = torch.argmax(probs[i]).item()
            confidence = probs[i, class_id].item()

            if class_id < 10 and confidence > 0.5:
                cx = boxes[i][0].item()
                detections.append({
                    'digit': class_id,
                    'cx': cx
                })

        # Sort by x-coordinate
        detections.sort(key=lambda d: d['cx'])

        digits = [d['digit'] for d in detections]
        sequence = ''.join(str(d) for d in digits)

        return sequence
    except Exception as e:
        return None


def run_yolo_on_sample(image: np.ndarray, yolo_model_path: str, cnn_model_dir: str,
                       device: str = 'cuda') -> Optional[str]:
    """Run YOLO + CNN inference on a single sample."""
    try:
        from ultralytics import YOLO
        from src.svhn_kmeans.cnn_model import create_model
        from src.svhn_kmeans.utils import resize_to_32x32

        # Use cached models to avoid reloading on every sample
        cache_key = f'yolo_{yolo_model_path}_{cnn_model_dir}'

        if cache_key not in _MODEL_CACHE:
            # Load YOLO
            yolo_model = YOLO(yolo_model_path, task='detect')

            # Load CNN
            cnn_model_path = Path(cnn_model_dir) / 'best_model.pth'
            cnn_model = create_model(
                model_type='enhanced',
                input_channels=3,
                use_stn=True
            )
            checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)
            cnn_model.load_state_dict(checkpoint['model_state_dict'])
            cnn_model = cnn_model.to(device)
            cnn_model.eval()

            _MODEL_CACHE[cache_key] = {
                'yolo': yolo_model,
                'cnn': cnn_model
            }
        else:
            yolo_model = _MODEL_CACHE[cache_key]['yolo']
            cnn_model = _MODEL_CACHE[cache_key]['cnn']

        # YOLO detection
        yolo_results = yolo_model(image, conf=0.25, verbose=False)

        # Process detections
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'x1': x1
                })

        # Recognize digits
        results = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_resized = resize_to_32x32(crop, maintain_aspect=True)
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).float()
            crop_tensor = crop_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = cnn_model(crop_tensor)

            num_digits = torch.argmax(outputs['num'], dim=1).item()

            digits = []
            if num_digits >= 1:
                dig1 = torch.argmax(outputs['dig1'], dim=1).item()
                if dig1 < 10:
                    digits.append(dig1)
            if num_digits >= 2:
                dig2 = torch.argmax(outputs['dig2'], dim=1).item()
                if dig2 < 10:
                    digits.append(dig2)
            if num_digits >= 3:
                dig3 = torch.argmax(outputs['dig3'], dim=1).item()
                if dig3 < 10:
                    digits.append(dig3)
            if num_digits >= 4:
                dig4 = torch.argmax(outputs['dig4'], dim=1).item()
                if dig4 < 10:
                    digits.append(dig4)

            sequence = ''.join(str(d) for d in digits)

            results.append({
                'digits': digits,
                'x1': det['x1']
            })

        # Sort by x-coordinate and combine
        results.sort(key=lambda r: r['x1'])
        all_digits = []
        for r in results:
            all_digits.extend(r['digits'])

        combined_sequence = ''.join(str(d) for d in all_digits)
        return combined_sequence if combined_sequence else None
    except Exception as e:
        return None


def find_examples_for_model(model_name: str, data_loader, model_args: Dict,
                            max_samples: int = 500, exclude_single_digit: bool = False,
                            used_correct_indices: set = None) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Find one correct and one incorrect prediction for a model.

    Args:
        model_name: Name of the model
        data_loader: SVHNDataLoader instance
        model_args: Dictionary with model paths and device
        max_samples: Maximum number of samples to check
        exclude_single_digit: If True, skip samples with single digit sequences
        used_correct_indices: Set of indices already used for correct predictions by other models

    Returns:
        Tuple of (correct_example, incorrect_example), each is a dict or None
    """
    print(f"\nSearching for examples for {model_name}...")

    if used_correct_indices is None:
        used_correct_indices = set()

    correct_example = None
    incorrect_example = None

    num_samples = min(len(data_loader), max_samples)

    for i in range(num_samples):
        if correct_example is not None and incorrect_example is not None:
            break

        try:
            # Load sample
            sample = data_loader[i]
            image = sample['image']
            metadata = sample['metadata']
            bboxes = metadata['bboxes']

            # Get ground truth
            gt_sequence = get_ground_truth_sequence(bboxes)

            # Skip single digit examples if requested
            if exclude_single_digit and len(gt_sequence) == 1:
                continue

            # Run inference
            if model_name == 'kmeans':
                debug = (i < 3)  # Debug first 3 samples
                if debug:
                    print(f"  Sample {i}: GT = {gt_sequence}")
                pred_sequence = run_kmeans_on_sample(
                    image, bboxes, model_args['kmeans_model'], model_args['device'], debug=debug
                )
            elif model_name == 'cnn':
                pred_sequence = run_cnn_on_sample(
                    image, model_args['cnn_model'], model_args['device']
                )
            elif model_name == 'cqt':
                pred_sequence = run_cqt_on_sample(
                    image, model_args['cqt_model'], model_args['device']
                )
            elif model_name == 'yolo':
                pred_sequence = run_yolo_on_sample(
                    image, model_args['yolo_model'], model_args['cnn_model'],
                    model_args['device']
                )
            else:
                continue

            if pred_sequence is None or pred_sequence == '':
                continue

            # Check if prediction is correct
            is_correct = pred_sequence == gt_sequence

            if is_correct and correct_example is None and i not in used_correct_indices:
                correct_example = {
                    'image': image,
                    'metadata': metadata,
                    'gt_sequence': gt_sequence,
                    'pred_sequence': pred_sequence,
                    'index': i
                }
                used_correct_indices.add(i)
                print(f"  Found correct prediction at index {i}: {gt_sequence} == {pred_sequence}")

            elif not is_correct and incorrect_example is None:
                incorrect_example = {
                    'image': image,
                    'metadata': metadata,
                    'gt_sequence': gt_sequence,
                    'pred_sequence': pred_sequence,
                    'index': i
                }
                print(f"  Found incorrect prediction at index {i}: {gt_sequence} != {pred_sequence}")

        except Exception as e:
            continue

    if correct_example is None:
        print(f"  WARNING: No correct prediction found for {model_name}")
    if incorrect_example is None:
        print(f"  WARNING: No incorrect prediction found for {model_name}")

    return correct_example, incorrect_example


def visualize_examples(model_name: str, correct_example: Optional[Dict],
                       incorrect_example: Optional[Dict], output_path: str):
    """
    Visualize correct and incorrect examples for a model.

    Args:
        model_name: Name of the model
        correct_example: Dictionary with correct prediction info
        incorrect_example: Dictionary with incorrect prediction info
        output_path: Path to save the visualization
    """
    model_labels = {
        'kmeans': 'K-means + SVM',
        'cnn': 'CNN with STN',
        'cqt': 'Character Query Transformer',
        'yolo': 'YOLO + CNN'
    }

    model_label = model_labels.get(model_name, model_name)

    # Count how many examples we have
    num_examples = sum([correct_example is not None, incorrect_example is not None])

    if num_examples == 0:
        print(f"  No examples to visualize for {model_name}")
        return

    fig, axes = plt.subplots(1, num_examples, figsize=(8 * num_examples, 8))
    if num_examples == 1:
        axes = [axes]

    idx = 0

    # Plot correct example
    if correct_example is not None:
        ax = axes[idx]
        image = correct_example['image']
        bboxes = correct_example['metadata']['bboxes']
        gt_seq = correct_example['gt_sequence']
        pred_seq = correct_example['pred_sequence']

        ax.imshow(image)

        # Draw ground truth bboxes
        for bbox in bboxes:
            x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        ax.set_title(f"✓ STRENGTH: Correct Prediction\nGT: {gt_seq} | Pred: {pred_seq}",
                    fontsize=14, fontweight='bold', color='green')
        idx += 1

    # Plot incorrect example
    if incorrect_example is not None:
        ax = axes[idx]
        image = incorrect_example['image']
        bboxes = incorrect_example['metadata']['bboxes']
        gt_seq = incorrect_example['gt_sequence']
        pred_seq = incorrect_example['pred_sequence']

        ax.imshow(image)

        # Draw ground truth bboxes
        for bbox in bboxes:
            x, y, w, h = bbox['left'], bbox['top'], bbox['width'], bbox['height']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.axis('off')
        ax.set_title(f"✗ WEAKNESS: Incorrect Prediction\nGT: {gt_seq} | Pred: {pred_seq}",
                    fontsize=14, fontweight='bold', color='red')

    plt.suptitle(f"{model_label}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Demonstrate strengths and weaknesses of all trained models'
    )

    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='model_analysis',
                       help='Output directory for visualizations')

    # Model directories
    parser.add_argument('--kmeans_model', type=str, default='models',
                       help='K-means + SVM model directory')
    parser.add_argument('--cnn_model', type=str, default='models_cnn',
                       help='CNN model directory')
    parser.add_argument('--cqt_model', type=str, default='models_cqt',
                       help='CQT model directory')
    parser.add_argument('--yolo_model', type=str, default='models_yolo/train/weights/best.pt',
                       help='YOLO model path')

    # Device
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference')

    # Options
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of test samples to check')
    parser.add_argument('--exclude_single_digit', action='store_true',
                       help='Exclude single digit examples from analysis')
    parser.add_argument('--skip_kmeans', action='store_true',
                       help='Skip K-means + SVM')
    parser.add_argument('--skip_cnn', action='store_true',
                       help='Skip CNN')
    parser.add_argument('--skip_cqt', action='store_true',
                       help='Skip CQT')
    parser.add_argument('--skip_yolo', action='store_true',
                       help='Skip YOLO + CNN')

    args = parser.parse_args()

    print("=" * 70)
    print("Model Strengths & Weaknesses Analysis")
    print("=" * 70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples to check: {args.max_samples}")
    print(f"Exclude single digit examples: {args.exclude_single_digit}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("\nLoading test dataset...")
    try:
        from src.svhn_kmeans.data_loader import SVHNDataLoader
        data_loader = SVHNDataLoader(args.data_dir, split='test')
        print(f"  Loaded {len(data_loader)} test samples")
    except Exception as e:
        print(f"  ✗ Error loading test data: {e}")
        print("\nMake sure the SVHN test data is available at:")
        print(f"  {args.data_dir}/test/")
        return

    # Check which models to run
    models_to_run = []

    if not args.skip_kmeans:
        kmeans_path = Path(args.kmeans_model) / 'feature_extractor.pkl'
        if kmeans_path.exists():
            models_to_run.append('kmeans')
            print(f"  ✓ K-means + SVM model found")
        else:
            print(f"  ✗ K-means + SVM model not found")

    if not args.skip_cnn:
        cnn_path = Path(args.cnn_model) / 'best_model.pth'
        if cnn_path.exists():
            models_to_run.append('cnn')
            print(f"  ✓ CNN model found")
        else:
            print(f"  ✗ CNN model not found")

    if not args.skip_cqt:
        cqt_path = Path(args.cqt_model) / 'best_model.pth'
        if cqt_path.exists():
            models_to_run.append('cqt')
            print(f"  ✓ CQT model found")
        else:
            print(f"  ✗ CQT model not found")

    if not args.skip_yolo:
        if Path(args.yolo_model).exists():
            cnn_path = Path(args.cnn_model) / 'best_model.pth'
            if cnn_path.exists():
                models_to_run.append('yolo')
                print(f"  ✓ YOLO + CNN models found")
            else:
                print(f"  ✗ CNN model not found (required for YOLO)")
        else:
            print(f"  ✗ YOLO model not found")

    if not models_to_run:
        print("\n✗ No trained models found!")
        return

    print(f"\nAnalyzing {len(models_to_run)} model(s)...")

    # Model arguments
    model_args = {
        'kmeans_model': args.kmeans_model,
        'cnn_model': args.cnn_model,
        'cqt_model': args.cqt_model,
        'yolo_model': args.yolo_model,
        'device': args.device
    }

    # Track indices used for correct predictions to avoid duplicates across models
    used_correct_indices = set()

    # Process each model
    for model_name in models_to_run:
        print("\n" + "=" * 70)
        print(f"Processing {model_name.upper()}")
        print("=" * 70)

        # Find examples
        correct_example, incorrect_example = find_examples_for_model(
            model_name, data_loader, model_args, args.max_samples,
            exclude_single_digit=args.exclude_single_digit,
            used_correct_indices=used_correct_indices
        )

        # Visualize
        output_path = output_dir / f'{model_name}_analysis.png'
        visualize_examples(model_name, correct_example, incorrect_example, str(output_path))

    # Print summary
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    for model_name in models_to_run:
        print(f"  - {model_name}_analysis.png")

    print("\nEach visualization shows:")
    print("  LEFT:  Strength - A correct prediction (green)")
    print("  RIGHT: Weakness - An incorrect prediction (red)")


if __name__ == '__main__':
    main()
