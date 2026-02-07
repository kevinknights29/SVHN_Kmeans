#!/usr/bin/env python3
"""
Evaluate end-to-end YOLO + CNN pipeline on SVHN test set.

This script measures:
- Sequence accuracy (all digits correct)
- Detection success rate
- Per-digit accuracy
- Average inference time

Usage:
    python evaluate_yolo_cnn.py --yolo_model models/yolo/train/weights/best.pt \
                                 --cnn_model models/cnn

    # Evaluate on subset for quick test
    python evaluate_yolo_cnn.py --yolo_model models/yolo/train/weights/best.pt \
                                 --cnn_model models/cnn --max_samples 1000
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.svhn_kmeans import SVHNDataLoader
from src.svhn_kmeans.cnn_model import create_model
from src.svhn_kmeans.utils import resize_to_32x32


def check_ultralytics():
    """Check if ultralytics is installed."""
    try:
        from ultralytics import YOLO
        return True
    except ImportError:
        print("ERROR: ultralytics package not found")
        print("Install with: pip install ultralytics")
        return False


def load_models(yolo_model_path: str, cnn_model_dir: str, device: str):
    """Load YOLO and CNN models."""
    from ultralytics import YOLO

    print("Loading models...")

    # Load YOLO
    yolo_model = YOLO(yolo_model_path)
    print(f"  ✓ YOLO model loaded")

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
    print(f"  ✓ CNN model loaded")

    return yolo_model, cnn_model


def predict_single_image(yolo_model, cnn_model, image: np.ndarray,
                         device: str, conf_threshold: float = 0.25) -> Dict:
    """
    Run end-to-end prediction on a single image.

    Returns:
        Dictionary with:
            - 'detected': bool (whether YOLO found a detection)
            - 'digits': list of predicted digits (empty if no detection)
            - 'confidence': average digit confidence
            - 'yolo_conf': YOLO detection confidence
    """
    # YOLO detection
    results = yolo_model(image, conf=conf_threshold, verbose=False)

    detected = False
    digits = []
    digit_conf = 0.0
    yolo_conf = 0.0

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            # Take first (highest confidence) detection
            box = boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            yolo_conf = float(box.conf[0])

            # Crop detected region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                detected = True

                # Resize to 32x32 and normalize
                crop_resized = resize_to_32x32(crop, maintain_aspect=True)
                crop_normalized = crop_resized.astype(np.float32) / 255.0
                crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).float()
                crop_tensor = crop_tensor.unsqueeze(0).to(device)

                # CNN recognition
                with torch.no_grad():
                    outputs = cnn_model(crop_tensor)

                # Parse outputs
                num_digits = torch.argmax(outputs['num'], dim=1).item()

                confidences = []
                if num_digits >= 1:
                    dig1 = torch.argmax(outputs['dig1'], dim=1).item()
                    if dig1 < 10:
                        digits.append(dig1)
                        confidences.append(torch.softmax(outputs['dig1'], dim=1)[0, dig1].item())
                if num_digits >= 2:
                    dig2 = torch.argmax(outputs['dig2'], dim=1).item()
                    if dig2 < 10:
                        digits.append(dig2)
                        confidences.append(torch.softmax(outputs['dig2'], dim=1)[0, dig2].item())
                if num_digits >= 3:
                    dig3 = torch.argmax(outputs['dig3'], dim=1).item()
                    if dig3 < 10:
                        digits.append(dig3)
                        confidences.append(torch.softmax(outputs['dig3'], dim=1)[0, dig3].item())
                if num_digits >= 4:
                    dig4 = torch.argmax(outputs['dig4'], dim=1).item()
                    if dig4 < 10:
                        digits.append(dig4)
                        confidences.append(torch.softmax(outputs['dig4'], dim=1)[0, dig4].item())

                if confidences:
                    digit_conf = np.mean(confidences)

            break  # Only use first detection

    return {
        'detected': detected,
        'digits': digits,
        'confidence': digit_conf,
        'yolo_conf': yolo_conf
    }


def evaluate(yolo_model, cnn_model, data_loader: SVHNDataLoader,
            device: str, max_samples: int = None, conf_threshold: float = 0.25):
    """
    Evaluate YOLO + CNN pipeline on dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    n_samples = len(data_loader)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    print(f"\nEvaluating on {n_samples} samples...")

    # Metrics
    total = 0
    detected = 0
    sequence_correct = 0
    total_digits = 0
    correct_digits = 0
    total_time = 0.0

    results_list = []

    for idx in tqdm(range(n_samples), desc="Evaluating"):
        try:
            # Get image and ground truth
            crop, bboxes = data_loader.get_full_number_crop(idx)
            gt_digits = [bbox['label'] % 10 for bbox in bboxes]

            # Run prediction
            start_time = time.time()
            pred = predict_single_image(yolo_model, cnn_model, crop, device, conf_threshold)
            inference_time = time.time() - start_time

            total_time += inference_time

            # Check if detected
            if pred['detected']:
                detected += 1

                # Check sequence accuracy
                if pred['digits'] == gt_digits:
                    sequence_correct += 1

                # Check per-digit accuracy
                for i, (pred_digit, gt_digit) in enumerate(zip(pred['digits'], gt_digits)):
                    total_digits += 1
                    if pred_digit == gt_digit:
                        correct_digits += 1

                # Count remaining unmatched ground truth digits
                if len(pred['digits']) < len(gt_digits):
                    total_digits += len(gt_digits) - len(pred['digits'])

            else:
                # No detection - all ground truth digits are wrong
                total_digits += len(gt_digits)

            total += 1

            results_list.append({
                'idx': idx,
                'gt_digits': gt_digits,
                'pred_digits': pred['digits'],
                'detected': pred['detected'],
                'correct': pred['digits'] == gt_digits,
                'confidence': pred['confidence'],
                'yolo_conf': pred['yolo_conf'],
                'inference_time': inference_time
            })

        except Exception as e:
            print(f"\nError processing image {idx}: {e}")
            total += 1
            total_digits += len(gt_digits) if 'gt_digits' in locals() else 0
            continue

    # Calculate metrics
    detection_rate = detected / total if total > 0 else 0.0
    sequence_accuracy = sequence_correct / total if total > 0 else 0.0
    digit_accuracy = correct_digits / total_digits if total_digits > 0 else 0.0
    avg_inference_time = total_time / total if total > 0 else 0.0

    metrics = {
        'total_samples': total,
        'detected': detected,
        'detection_rate': detection_rate,
        'sequence_correct': sequence_correct,
        'sequence_accuracy': sequence_accuracy,
        'total_digits': total_digits,
        'correct_digits': correct_digits,
        'digit_accuracy': digit_accuracy,
        'avg_inference_time': avg_inference_time,
        'total_time': total_time,
        'results': results_list
    }

    return metrics


def print_metrics(metrics: Dict):
    """Print evaluation metrics."""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    print(f"\nDataset:")
    print(f"  Total samples: {metrics['total_samples']}")

    print(f"\nDetection Performance:")
    print(f"  Detected: {metrics['detected']}/{metrics['total_samples']}")
    print(f"  Detection rate: {metrics['detection_rate']:.2%}")

    print(f"\nRecognition Performance:")
    print(f"  Sequence correct: {metrics['sequence_correct']}/{metrics['total_samples']}")
    print(f"  Sequence accuracy: {metrics['sequence_accuracy']:.2%}")
    print(f"  Digit correct: {metrics['correct_digits']}/{metrics['total_digits']}")
    print(f"  Digit accuracy: {metrics['digit_accuracy']:.2%}")

    print(f"\nInference Speed:")
    print(f"  Average time: {metrics['avg_inference_time']*1000:.1f}ms")
    print(f"  Total time: {metrics['total_time']:.1f}s")

    # Show some examples
    print(f"\nExample Predictions:")
    results = metrics['results']

    # Show first 5 correct
    correct = [r for r in results if r['correct']][:5]
    if correct:
        print(f"  Correct predictions:")
        for r in correct:
            gt_str = ''.join(str(d) for d in r['gt_digits'])
            pred_str = ''.join(str(d) for d in r['pred_digits'])
            print(f"    Image {r['idx']:5d}: GT={gt_str:6s} Pred={pred_str:6s} "
                  f"(conf={r['confidence']:.3f}, yolo={r['yolo_conf']:.3f})")

    # Show first 5 incorrect
    incorrect = [r for r in results if not r['correct']][:5]
    if incorrect:
        print(f"  Incorrect predictions:")
        for r in incorrect:
            gt_str = ''.join(str(d) for d in r['gt_digits'])
            pred_str = ''.join(str(d) for d in r['pred_digits']) if r['detected'] else 'NO_DET'
            print(f"    Image {r['idx']:5d}: GT={gt_str:6s} Pred={pred_str:6s} "
                  f"(conf={r['confidence']:.3f}, yolo={r['yolo_conf']:.3f})")


def save_results(metrics: Dict, output_file: str):
    """Save detailed results to CSV file."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_idx', 'ground_truth', 'prediction', 'detected',
            'correct', 'digit_confidence', 'yolo_confidence', 'inference_time_ms'
        ])

        for r in metrics['results']:
            gt_str = ''.join(str(d) for d in r['gt_digits'])
            pred_str = ''.join(str(d) for d in r['pred_digits']) if r['detected'] else ''

            writer.writerow([
                r['idx'],
                gt_str,
                pred_str,
                int(r['detected']),
                int(r['correct']),
                f"{r['confidence']:.4f}",
                f"{r['yolo_conf']:.4f}",
                f"{r['inference_time']*1000:.2f}"
            ])

    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO + CNN pipeline')

    parser.add_argument('--yolo_model', type=str, required=True,
                       help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--cnn_model', type=str, required=True,
                       help='Path to trained CNN model directory')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to SVHN dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to evaluate (None for all)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='YOLO confidence threshold')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference')
    parser.add_argument('--output', type=str, default='yolo_cnn_evaluation.csv',
                       help='Output CSV file for detailed results')

    args = parser.parse_args()

    print("=" * 70)
    print("YOLO + CNN Pipeline Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  YOLO model: {args.yolo_model}")
    print(f"  CNN model: {args.cnn_model}")
    print(f"  Dataset: {args.data_dir} ({args.split})")
    print(f"  Device: {args.device}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")

    # Check ultralytics
    if not check_ultralytics():
        return

    # Check models exist
    if not Path(args.yolo_model).exists():
        print(f"\n✗ YOLO model not found: {args.yolo_model}")
        return

    cnn_model_path = Path(args.cnn_model) / 'best_model.pth'
    if not cnn_model_path.exists():
        print(f"\n✗ CNN model not found: {cnn_model_path}")
        return

    # Load models
    yolo_model, cnn_model = load_models(args.yolo_model, args.cnn_model, args.device)

    # Load dataset
    print(f"\nLoading SVHN {args.split} data from {args.data_dir}...")
    data_loader = SVHNDataLoader(args.data_dir, split=args.split)
    print(f"  Found {len(data_loader)} images")

    # Evaluate
    metrics = evaluate(
        yolo_model,
        cnn_model,
        data_loader,
        args.device,
        args.max_samples,
        args.conf_threshold
    )

    # Print results
    print_metrics(metrics)

    # Save results
    save_results(metrics, args.output)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
