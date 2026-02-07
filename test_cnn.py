#!/usr/bin/env python3
"""
Quick test script to verify CNN model is working correctly.
"""

import torch
import sys

def test_cnn_model():
    """Test CNN model creation and forward pass."""
    print("Testing CNN Model...")
    print("=" * 60)

    try:
        from src.svhn_kmeans.cnn_model import create_model
        print("✓ CNN model imports successful")
    except ImportError as e:
        print(f"✗ Failed to import CNN model: {e}")
        print("\nTo install PyTorch:")
        print("  pip install torch torchvision")
        return False

    # Test model creation
    print("\n1. Creating model...")
    try:
        model = create_model('enhanced', use_stn=True)
        print(f"✓ Model created successfully")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {n_params:,}")
        print(f"  Trainable parameters: {n_trainable:,}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        print(f"  Input shape: {x.shape}")

        outputs, theta = model(x, return_theta=True)
        print(f"✓ Forward pass successful")

        print(f"\n  Output shapes:")
        for key, value in outputs.items():
            print(f"    {key}: {value.shape}")

        if theta is not None:
            print(f"    theta (STN): {theta.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test without STN
    print("\n3. Testing without STN...")
    try:
        model_no_stn = create_model('enhanced', use_stn=False)
        outputs_no_stn = model_no_stn(x)
        print(f"✓ Model without STN works")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test loss function
    print("\n4. Testing loss function...")
    try:
        from src.svhn_kmeans.cnn_utils import MultiOutputLoss

        criterion = MultiOutputLoss()
        print(f"✓ Loss function created")

        # Create dummy targets
        targets = {
            'num': torch.randint(0, 5, (batch_size,)),
            'dig1': torch.randint(0, 11, (batch_size,)),
            'dig2': torch.randint(0, 11, (batch_size,)),
            'dig3': torch.randint(0, 11, (batch_size,)),
            'dig4': torch.randint(0, 11, (batch_size,)),
            'nC': torch.randint(0, 2, (batch_size,)),
        }

        loss, losses = criterion(outputs, targets)
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Individual losses:")
        for key, value in losses.items():
            print(f"    {key}: {value:.4f}")
        print(f"✓ Loss computation successful")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test metrics
    print("\n5. Testing metrics...")
    try:
        from src.svhn_kmeans.cnn_utils import measure_prediction

        per_digit_acc, sequence_acc = measure_prediction(outputs, targets)
        print(f"✓ Metrics computation successful")
        print(f"  Per-digit accuracies: {[f'{acc:.1f}%' for acc in per_digit_acc]}")
        print(f"  Sequence accuracy: {sequence_acc:.1f}%")
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        return False

    # Test CUDA if available
    print("\n6. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

        # Test model on GPU
        try:
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            outputs_cuda = model_cuda(x_cuda)
            print(f"✓ Model works on GPU")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
    else:
        print(f"⚠ CUDA not available (CPU only)")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou're ready to train the CNN model:")
    print("  python train_cnn.py --data_dir data --output_dir models_cnn")
    return True


if __name__ == '__main__':
    success = test_cnn_model()
    sys.exit(0 if success else 1)
