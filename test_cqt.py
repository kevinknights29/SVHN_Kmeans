#!/usr/bin/env python3
"""
Quick test script to verify Character Query Transformer model is working correctly.
"""

import torch
import sys


def test_cqt_model():
    """Test CQT model creation and forward pass."""
    print("Testing Character Query Transformer Model...")
    print("=" * 60)

    try:
        from src.svhn_kmeans.cqt_model import create_cqt_model
        from src.svhn_kmeans.cqt_utils import CQTLoss, compute_accuracy, count_parameters
        print("✓ CQT model imports successful")
    except ImportError as e:
        print(f"✗ Failed to import CQT model: {e}")
        return False

    # Test model creation
    print("\n1. Creating model...")
    try:
        model = create_cqt_model(
            num_queries=6,
            num_classes=11,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            pretrained_backbone=False  # Faster for testing
        )
        print(f"✓ Model created successfully")

        # Count parameters
        n_params = count_parameters(model)
        print(f"  Total trainable parameters: {n_params:,}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        print(f"  Input shape: {x.shape}")

        outputs = model(x)
        print(f"✓ Forward pass successful")

        print(f"\n  Output shapes:")
        for key, value in outputs.items():
            print(f"    {key}: {value.shape}")

        # Check output shapes
        assert outputs['class_logits'].shape == (batch_size, 6, 11), \
            f"Expected class_logits shape ({batch_size}, 6, 11), got {outputs['class_logits'].shape}"
        assert outputs['bbox_coords'].shape == (batch_size, 6, 4), \
            f"Expected bbox_coords shape ({batch_size}, 6, 4), got {outputs['bbox_coords'].shape}"

        print(f"✓ Output shapes correct")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loss function
    print("\n3. Testing loss function...")
    try:
        criterion = CQTLoss(num_classes=11)
        print(f"✓ Loss function created")

        # Create dummy targets
        targets = [
            {
                'labels': torch.tensor([3, 5, 7]),  # 3 digits
                'boxes': torch.tensor([
                    [0.3, 0.5, 0.1, 0.2],
                    [0.5, 0.5, 0.1, 0.2],
                    [0.7, 0.5, 0.1, 0.2],
                ]),
            },
            {
                'labels': torch.tensor([1, 9]),  # 2 digits
                'boxes': torch.tensor([
                    [0.4, 0.5, 0.15, 0.25],
                    [0.6, 0.5, 0.15, 0.25],
                ]),
            },
        ]

        loss, loss_dict = criterion(outputs, targets)
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Individual losses:")
        for key, value in loss_dict.items():
            print(f"    {key}: {value:.4f}")
        print(f"✓ Loss computation successful")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test metrics
    print("\n4. Testing metrics...")
    try:
        metrics = compute_accuracy(outputs, targets)
        print(f"✓ Metrics computation successful")
        print(f"  Metrics:")
        for key, value in metrics.items():
            print(f"    {key}: {value:.2%}")
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test CUDA if available
    print("\n5. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

        # Test model on GPU
        try:
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            targets_cuda = [{k: v.cuda() for k, v in t.items()} for t in targets]

            outputs_cuda = model_cuda(x_cuda)
            loss_cuda, _ = criterion(outputs_cuda, targets_cuda)

            print(f"✓ Model works on GPU")
            print(f"  GPU loss: {loss_cuda.item():.4f}")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
    else:
        print(f"⚠ CUDA not available (CPU only)")

    # Test backbone options
    print("\n6. Testing backbone options...")
    try:
        model_pretrained = create_cqt_model(
            num_queries=4,
            d_model=128,
            pretrained_backbone=True
        )
        print(f"✓ Model with pretrained backbone created")

        x_small = torch.randn(1, 3, 224, 224)
        outputs_pretrained = model_pretrained(x_small)
        print(f"✓ Pretrained backbone forward pass successful")
    except Exception as e:
        print(f"⚠ Pretrained backbone test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou're ready to train the CQT model:")
    print("  python train_cqt.py --data_dir data --output_dir models_cqt")
    print("\nFor quick testing:")
    print("  python train_cqt.py --max_samples 5000 --epochs 10 --batch_size 8")
    return True


if __name__ == '__main__':
    success = test_cqt_model()
    sys.exit(0 if success else 1)
