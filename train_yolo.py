#!/usr/bin/env python3
"""
Fine-tune YOLOv8 for SVHN house number detection.

This script fine-tunes a pretrained YOLOv8 model on the SVHN dataset
converted to YOLO format.

Usage:
    # Quick test (5-10 minutes)
    python train_yolo.py --data data_yolo/dataset.yaml --epochs 10 --imgsz 320

    # Full training (1-2 hours)
    python train_yolo.py --data data_yolo/dataset.yaml --epochs 100 --imgsz 640
"""

import argparse
import os
from pathlib import Path
import time

import torch


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
        print("\nThis will install YOLOv8 and its dependencies.")
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv8 for SVHN house number detection')

    # Data arguments
    parser.add_argument('--data', type=str, default='data_yolo/dataset.yaml',
                       help='Path to dataset.yaml file')
    parser.add_argument('--output_dir', type=str, default='models_yolo',
                       help='Output directory for trained models')

    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (-1 for auto)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size (must be multiple of 32)')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate (as fraction of lr0)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                       help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')

    # Data augmentation arguments
    parser.add_argument('--hsv_h', type=float, default=0.015,
                       help='HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                       help='HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                       help='HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Image rotation (+/- degrees)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Image shear (+/- degrees)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Probability of vertical flip')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Probability of horizontal flip')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Probability of mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Probability of mixup augmentation')

    # Device arguments
    parser.add_argument('--device', type=str,
                       default='0' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (0, 1, 2, ... or cpu)')

    # Other arguments
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--exist_ok', action='store_true',
                       help='Allow overwriting existing project')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                       help='Don\'t use pretrained weights')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Check for ultralytics
    if not check_ultralytics():
        return

    from ultralytics import YOLO

    print("=" * 70)
    print("YOLOv8 Fine-tuning for SVHN House Number Detection")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.data}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr0} -> {args.lr0 * args.lrf}")
    print(f"  Pretrained: {args.pretrained}")

    # Validate data file
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n✗ Error: Dataset file not found: {args.data}")
        print(f"\nPlease run the data converter first:")
        print(f"  python convert_svhn_to_yolo.py --data_dir data --output_dir data_yolo")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print(f"\nLoading model...")
    model = YOLO(args.model)

    if args.pretrained:
        print(f"  ✓ Using pretrained weights from {args.model}")
    else:
        print(f"  Using random initialization")

    # Training configuration
    train_kwargs = {
        'data': str(data_path),
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'device': args.device,
        'project': str(output_dir),
        'name': 'train',
        'exist_ok': args.exist_ok,
        'pretrained': args.pretrained,
        'verbose': args.verbose,
        # Data augmentation
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        # Other settings
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'val': True,
        'plots': True,
    }

    if args.resume:
        train_kwargs['resume'] = True
        print(f"\n  Resuming from last checkpoint...")

    # Train model
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    start_time = time.time()

    try:
        results = model.train(**train_kwargs)
        training_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("Training complete!")
        print("=" * 70)
        print(f"\nTraining time: {training_time / 3600:.2f} hours")

        # Print best metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\nBest metrics:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")

        # Model saved locations
        train_dir = output_dir / 'train'
        best_model = train_dir / 'weights' / 'best.pt'
        last_model = train_dir / 'weights' / 'last.pt'

        print(f"\nModel files:")
        if best_model.exists():
            print(f"  Best model: {best_model}")
        if last_model.exists():
            print(f"  Last model: {last_model}")

        print(f"\nResults directory: {train_dir}")
        print(f"  - weights/: Model checkpoints")
        print(f"  - results.png: Training curves")
        print(f"  - confusion_matrix.png: Confusion matrix")
        print(f"  - val_batch*_pred.jpg: Validation predictions")

        print(f"\nTo test the model:")
        print(f"  python test_yolo.py --model {best_model}")

        print(f"\nTo use in end-to-end pipeline:")
        print(f"  python inference_yolo.py --yolo_model {best_model} --cnn_model models_cnn")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("You can resume training with:")
        print(f"  python train_yolo.py --data {args.data} --resume")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
