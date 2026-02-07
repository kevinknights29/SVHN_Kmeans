#!/usr/bin/env python3
"""
Run inference on all trained models and save results.

This script processes a single image with all available models:
- K-means + SVM
- CNN with STN
- Character Query Transformer (CQT)
- YOLO + CNN

Usage:
    python inference_all_models.py --image test.png --output_dir results
"""

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def check_model_exists(model_path: str, model_name: str) -> bool:
    """Check if model exists and print status."""
    exists = Path(model_path).exists()
    if exists:
        print(f"  ✓ {model_name} model found: {model_path}")
    else:
        print(f"  ✗ {model_name} model not found: {model_path}")
        print(f"    Train it first with the appropriate training script")
    return exists


def run_kmeans(image_path: str, model_dir: str, output_path: str, device: str = 'cpu') -> dict:
    """Run K-means + SVM inference."""
    print("\n" + "=" * 70)
    print("K-means + SVM Inference")
    print("=" * 70)

    try:
        from src.svhn_kmeans import SVHNRecognitionPipeline

        # Load model
        print("Loading K-means + SVM model...")
        pipeline = SVHNRecognitionPipeline()
        pipeline.load(model_dir)
        print("  ✓ Model loaded")

        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"  Image shape: {image.shape}")

        # Run inference
        start_time = time.time()
        result = pipeline.process_image(image)
        inference_time = time.time() - start_time

        # Get results
        digits = result['digits']
        sequence = ''.join(str(d) for d in digits)
        bboxes = result['bboxes']

        print(f"  Predicted sequence: {sequence}")
        print(f"  Number of digits: {len(digits)}")
        print(f"  Inference time: {inference_time:.3f}s")

        # Visualize
        vis_image = image.copy()
        for i, (digit, bbox) in enumerate(zip(digits, bboxes)):
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_image, str(digit), (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add title
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(vis_image)
        ax.axis('off')
        ax.set_title(f"K-means + SVM: {sequence}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Results saved: {output_path}")

        return {
            'success': True,
            'sequence': sequence,
            'digits': digits,
            'inference_time': inference_time,
            'bboxes': bboxes
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_cnn(image_path: str, model_dir: str, output_path: str, device: str = 'cuda') -> dict:
    """Run CNN inference."""
    print("\n" + "=" * 70)
    print("CNN with STN Inference")
    print("=" * 70)

    try:
        from src.svhn_kmeans.cnn_model import create_model

        # Load model
        print("Loading CNN model...")
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
        print("  ✓ Model loaded")

        # Load and preprocess image
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"  Image shape: {image.shape}")

        # Resize to 32x32 (CNN model expects 32x32 input)
        from src.svhn_kmeans.utils import resize_to_32x32
        image_resized = resize_to_32x32(image, maintain_aspect=True)

        # Normalize to [0, 1] (CNN uses simple normalization, not ImageNet)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
        inference_time = time.time() - start_time

        # Parse outputs
        num_digits = torch.argmax(outputs['num'], dim=1).item()

        digits = []
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

        sequence = ''.join(str(d) for d in digits)
        avg_conf = np.mean(confidences) if confidences else 0.0

        print(f"  Predicted sequence: {sequence}")
        print(f"  Number of digits: {num_digits}")
        print(f"  Average confidence: {avg_conf:.3f}")
        print(f"  Inference time: {inference_time:.3f}s")

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')

        title = f"CNN with STN: {sequence}"
        if confidences:
            title += f" (conf: {avg_conf:.2f})"
        ax.set_title(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Results saved: {output_path}")

        return {
            'success': True,
            'sequence': sequence,
            'digits': digits,
            'confidences': confidences,
            'inference_time': inference_time
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_cqt(image_path: str, model_dir: str, output_path: str, device: str = 'cuda') -> dict:
    """Run CQT inference."""
    print("\n" + "=" * 70)
    print("Character Query Transformer Inference")
    print("=" * 70)

    try:
        from src.svhn_kmeans.cqt_model import create_cqt_model

        # Load model
        print("Loading CQT model...")
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
        print("  ✓ Model loaded")

        # Load and preprocess image
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"  Image shape: {image.shape}")

        orig_h, orig_w = image.shape[:2]

        # Resize to 224x224 (CQT expects 224x224 with pretrained ResNet50)
        image_resized = cv2.resize(image, (224, 224))

        # Normalize with ImageNet stats (for pretrained ResNet)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
        inference_time = time.time() - start_time

        # Parse outputs
        pred_logits = outputs['class_logits']  # [B, num_queries, num_classes]
        pred_boxes = outputs['bbox_coords']    # [B, num_queries, 4]

        # Get predictions
        probs = torch.softmax(pred_logits, dim=-1)[0]  # [num_queries, num_classes]
        boxes = pred_boxes[0]  # [num_queries, 4]

        # Filter predictions (class != 10 which is empty)
        detections = []
        for i in range(probs.shape[0]):
            class_id = torch.argmax(probs[i]).item()
            confidence = probs[i, class_id].item()

            # Skip empty class (10) and low confidence
            if class_id < 10 and confidence > 0.5:
                cx, cy, w, h = boxes[i].cpu().numpy()

                # Convert to pixel coordinates (denormalize)
                x1 = int((cx - w/2) * orig_w)
                y1 = int((cy - h/2) * orig_h)
                x2 = int((cx + w/2) * orig_w)
                y2 = int((cy + h/2) * orig_h)

                # Clip to image bounds
                x1 = max(0, min(orig_w, x1))
                y1 = max(0, min(orig_h, y1))
                x2 = max(0, min(orig_w, x2))
                y2 = max(0, min(orig_h, y2))

                detections.append({
                    'digit': class_id,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

        # Sort by x-coordinate (left to right)
        detections.sort(key=lambda d: d['bbox'][0])

        digits = [d['digit'] for d in detections]
        sequence = ''.join(str(d) for d in digits)
        avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0.0

        print(f"  Predicted sequence: {sequence}")
        print(f"  Number of detections: {len(detections)}")
        print(f"  Average confidence: {avg_conf:.3f}")
        print(f"  Inference time: {inference_time:.3f}s")

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{det['digit']}", fontsize=12,
                   color='g', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.axis('off')

        title = f"CQT: {sequence}"
        if detections:
            title += f" (conf: {avg_conf:.2f})"
        ax.set_title(title, fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Results saved: {output_path}")

        return {
            'success': True,
            'sequence': sequence,
            'digits': digits,
            'detections': detections,
            'inference_time': inference_time
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_yolo(image_path: str, yolo_model_path: str, cnn_model_dir: str,
             output_path: str, device: str = 'cuda') -> dict:
    """Run YOLO + CNN inference."""
    print("\n" + "=" * 70)
    print("YOLO + CNN Inference")
    print("=" * 70)

    try:
        # Check for ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  ✗ ultralytics not installed")
            print("    Install with: pip install ultralytics")
            return {'success': False, 'error': 'ultralytics not installed'}

        from src.svhn_kmeans.cnn_model import create_model

        # Load YOLO
        print("Loading YOLO model...")
        yolo_model = YOLO(yolo_model_path)
        print("  ✓ YOLO model loaded")

        # Load CNN
        print("Loading CNN model...")
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
        print("  ✓ CNN model loaded")

        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"  Image shape: {image.shape}")

        # Run YOLO detection
        start_time = time.time()
        yolo_results = yolo_model(image, conf=0.25, verbose=False)
        yolo_time = time.time() - start_time

        # Process detections
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })

        print(f"  YOLO detections: {len(detections)}")
        print(f"  YOLO inference time: {yolo_time:.3f}s")

        # Recognize digits for each detection
        results = []
        cnn_time = 0

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Resize to 32x32 and normalize
            from src.svhn_kmeans.utils import resize_to_32x32
            crop_resized = resize_to_32x32(crop, maintain_aspect=True)
            crop_normalized = crop_resized.astype(np.float32) / 255.0
            crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1).float()
            crop_tensor = crop_tensor.unsqueeze(0).to(device)

            # Run CNN
            start_time = time.time()
            with torch.no_grad():
                outputs = cnn_model(crop_tensor)
            cnn_time += time.time() - start_time

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

            results.append({
                'bbox': det['bbox'],
                'yolo_confidence': det['confidence'],
                'digits': digits,
                'sequence': sequence
            })

        total_time = yolo_time + cnn_time

        # Combine all sequences
        all_digits = []
        for r in results:
            all_digits.extend(r['digits'])
        combined_sequence = ''.join(str(d) for d in all_digits)

        print(f"  Predicted sequence: {combined_sequence}")
        print(f"  CNN inference time: {cnn_time:.3f}s")
        print(f"  Total inference time: {total_time:.3f}s")

        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, res['sequence'], fontsize=12,
                   color='g', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.axis('off')
        ax.set_title(f"YOLO + CNN: {combined_sequence}", fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Results saved: {output_path}")

        return {
            'success': True,
            'sequence': combined_sequence,
            'results': results,
            'inference_time': total_time
        }

    except Exception as e:
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Run inference on all trained models')

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')

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
    print("Inference on All Trained Models")
    print("=" * 70)
    print(f"\nInput image: {args.image}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Check if image exists
    if not Path(args.image).exists():
        print(f"\n✗ Error: Image not found: {args.image}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which models exist
    print("\nChecking available models:")

    models_to_run = []

    if not args.skip_kmeans:
        kmeans_path = Path(args.kmeans_model) / 'feature_extractor.pkl'
        if check_model_exists(str(kmeans_path), "K-means + SVM"):
            models_to_run.append('kmeans')

    if not args.skip_cnn:
        cnn_path = Path(args.cnn_model) / 'best_model.pth'
        if check_model_exists(str(cnn_path), "CNN"):
            models_to_run.append('cnn')

    if not args.skip_cqt:
        cqt_path = Path(args.cqt_model) / 'best_model.pth'
        if check_model_exists(str(cqt_path), "CQT"):
            models_to_run.append('cqt')

    if not args.skip_yolo:
        if check_model_exists(args.yolo_model, "YOLO"):
            # Also check CNN for YOLO pipeline
            cnn_path = Path(args.cnn_model) / 'best_model.pth'
            if check_model_exists(str(cnn_path), "CNN (for YOLO pipeline)"):
                models_to_run.append('yolo')

    if not models_to_run:
        print("\n✗ No trained models found!")
        print("\nPlease train at least one model first:")
        print("  - K-means + SVM: python train.py")
        print("  - CNN: python train_cnn.py")
        print("  - CQT: python train_cqt.py")
        print("  - YOLO: python train_yolo.py")
        return

    print(f"\nRunning inference on {len(models_to_run)} model(s)...")

    # Run inference
    results = {}

    if 'kmeans' in models_to_run:
        output_path = output_dir / 'results_kmeans.png'
        results['kmeans'] = run_kmeans(
            args.image,
            args.kmeans_model,
            str(output_path),
            args.device
        )

    if 'cnn' in models_to_run:
        output_path = output_dir / 'results_cnn.png'
        results['cnn'] = run_cnn(
            args.image,
            args.cnn_model,
            str(output_path),
            args.device
        )

    if 'cqt' in models_to_run:
        output_path = output_dir / 'results_cqt.png'
        results['cqt'] = run_cqt(
            args.image,
            args.cqt_model,
            str(output_path),
            args.device
        )

    if 'yolo' in models_to_run:
        output_path = output_dir / 'results_yolo.png'
        results['yolo'] = run_yolo(
            args.image,
            args.yolo_model,
            args.cnn_model,
            str(output_path),
            args.device
        )

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    successful = [k for k, v in results.items() if v['success']]
    failed = [k for k, v in results.items() if not v['success']]

    if successful:
        print(f"\n✓ Successful: {len(successful)}/{len(results)}")
        print("\nPredictions:")
        for model_name in successful:
            result = results[model_name]
            model_label = {
                'kmeans': 'K-means + SVM',
                'cnn': 'CNN',
                'cqt': 'CQT',
                'yolo': 'YOLO + CNN'
            }[model_name]

            print(f"  {model_label:15s}: {result['sequence']:6s} ({result['inference_time']:.3f}s)")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for model_name in failed:
            print(f"  - {model_name}: {results[model_name]['error']}")

    print(f"\nResults saved to: {output_dir}/")
    print("  - results_kmeans.png (if K-means ran)")
    print("  - results_cnn.png (if CNN ran)")
    print("  - results_cqt.png (if CQT ran)")
    print("  - results_yolo.png (if YOLO ran)")


if __name__ == '__main__':
    main()
