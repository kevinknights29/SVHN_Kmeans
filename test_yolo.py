#!/usr/bin/env python3
"""
Test script to verify YOLO model setup for SVHN house number detection.

This script tests:
1. YOLOv8 model loading
2. Model inference on dummy input
3. Output format validation
4. Basic detection functionality

Run this before training to ensure everything is set up correctly.

Usage:
    python test_yolo.py
    python test_yolo.py --model yolov8n.pt
    python test_yolo.py --model models_yolo/train/weights/best.pt  # Test trained model
"""

import argparse
import sys

import numpy as np
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


def test_model_loading(model_name: str):
    """Test loading a YOLO model."""
    from ultralytics import YOLO

    print(f"\nTest 1: Loading YOLO model ({model_name})...")

    try:
        model = YOLO(model_name)
        print(f"  ✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None


def test_dummy_inference(model):
    """Test inference on dummy input."""
    print(f"\nTest 2: Inference on dummy input...")

    try:
        # Create dummy RGB image (640x640x3)
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference
        results = model(dummy_image, verbose=False)

        print(f"  ✓ Inference successful")
        print(f"  Number of results: {len(results)}")

        # Check result structure
        for i, result in enumerate(results):
            print(f"\n  Result {i}:")
            print(f"    Has boxes: {hasattr(result, 'boxes')}")
            if hasattr(result, 'boxes'):
                boxes = result.boxes
                print(f"    Number of detections: {len(boxes)}")
                if len(boxes) > 0:
                    print(f"    Box attributes: {dir(boxes)}")
                    print(f"    First box shape (xyxy): {boxes.xyxy[0].shape}")
                    print(f"    First box confidence: {float(boxes.conf[0]):.4f}")
                    print(f"    First box class: {int(boxes.cls[0])}")

        return True

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detection(model):
    """Test detection with a simple synthetic image."""
    print(f"\nTest 3: Detection on synthetic image...")

    try:
        # Create synthetic image with a white rectangle on black background
        # Simulates a simple house number
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Draw a white rectangle (simulating a house number region)
        image[200:400, 250:450] = [255, 255, 255]

        # Run inference
        results = model(image, conf=0.25, verbose=False)

        print(f"  ✓ Detection successful")

        # Check detections
        total_detections = 0
        for result in results:
            if hasattr(result, 'boxes'):
                total_detections += len(result.boxes)

        print(f"  Number of detections: {total_detections}")

        if total_detections > 0:
            print(f"  Note: Model detected objects in synthetic image")
            print(f"        (This is expected for pretrained models)")
        else:
            print(f"  Note: No detections (expected for untrained model)")

        return True

    except Exception as e:
        print(f"  ✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_format(model):
    """Test that output format matches expected structure."""
    print(f"\nTest 4: Output format validation...")

    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference
        results = model(dummy_image, verbose=False)

        # Validate result structure
        assert len(results) > 0, "No results returned"

        for result in results:
            # Check for required attributes
            assert hasattr(result, 'boxes'), "Result missing 'boxes' attribute"
            assert hasattr(result, 'orig_shape'), "Result missing 'orig_shape' attribute"

            boxes = result.boxes

            # If there are detections, validate box format
            if len(boxes) > 0:
                # Check xyxy format
                assert boxes.xyxy.shape[1] == 4, f"Box xyxy should have 4 values, got {boxes.xyxy.shape[1]}"

                # Check confidence
                assert len(boxes.conf) == len(boxes), "Confidence count mismatch"

                # Check class
                assert len(boxes.cls) == len(boxes), "Class count mismatch"

                print(f"  ✓ Output format is valid")
                print(f"    - xyxy shape: {boxes.xyxy.shape}")
                print(f"    - confidence shape: {boxes.conf.shape}")
                print(f"    - class shape: {boxes.cls.shape}")
            else:
                print(f"  ✓ Output format is valid (no detections)")

        return True

    except AssertionError as e:
        print(f"  ✗ Output format validation failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    """Test CUDA availability."""
    print(f"\nTest 5: Device compatibility...")

    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")

    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU acceleration available")
    else:
        print(f"  CPU only (slower training)")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test YOLO setup for SVHN')

    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model to test (e.g., yolov8n.pt, yolov8s.pt, or path to trained model)')

    args = parser.parse_args()

    print("=" * 70)
    print("YOLO Model Test for SVHN House Number Detection")
    print("=" * 70)
    print(f"\nTesting model: {args.model}")

    # Check for ultralytics
    if not check_ultralytics():
        sys.exit(1)

    # Run tests
    all_passed = True

    # Test 1: Model loading
    model = test_model_loading(args.model)
    if model is None:
        print("\n✗ Model loading failed")
        sys.exit(1)

    # Test 2: Dummy inference
    if not test_dummy_inference(model):
        all_passed = False

    # Test 3: Detection
    if not test_detection(model):
        all_passed = False

    # Test 4: Output format
    if not test_output_format(model):
        all_passed = False

    # Test 5: Device compatibility
    if not test_device_compatibility():
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nYou're ready to train YOLO:")
        print("  1. Convert SVHN to YOLO format:")
        print("     python convert_svhn_to_yolo.py --data_dir data --output_dir data_yolo")
        print("\n  2. Train YOLO:")
        print("     python train_yolo.py --data data_yolo/dataset.yaml --epochs 100")
        print("\n  3. Run inference:")
        print("     python inference_yolo.py --yolo_model models_yolo/train/weights/best.pt \\")
        print("                              --cnn_model models_cnn")
    else:
        print("✗ Some tests failed")
        print("=" * 70)
        print("\nPlease fix the errors above before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
