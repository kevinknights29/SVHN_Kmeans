#!/usr/bin/env python3
"""
CNN training script for SVHN digit recognition.

This script trains an enhanced CNN model with:
- Spatial Transformer Network
- Inception-ResNet blocks
- SE attention blocks
- Multi-task learning

Based on CNN_IMPLEMENTATION.md specifications.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.svhn_kmeans import SVHNDataLoader
from src.svhn_kmeans.cnn_model import create_model
from src.svhn_kmeans.cnn_utils import (
    MultiOutputLoss,
    measure_prediction,
    EarlyStopping,
    ModelCheckpoint,
    count_parameters,
    save_model,
)
from src.svhn_kmeans.utils import resize_to_32x32


def prepare_data(data_loader: SVHNDataLoader, max_samples: int = None,
                verbose: bool = True):
    """
    Prepare data for CNN training.

    Returns:
        Tuple of (images, labels_dict)
    """
    if verbose:
        print(f"Loading data from {data_loader.split} set...")

    n_samples = min(max_samples, len(data_loader)) if max_samples else len(data_loader)

    images = []
    labels_num = []
    labels_dig1 = []
    labels_dig2 = []
    labels_dig3 = []
    labels_dig4 = []
    labels_nC = []

    for i in range(n_samples):
        if verbose and i % 5000 == 0:
            print(f"  Processing {i}/{n_samples}")

        # Get full house number crop
        crop, bboxes = data_loader.get_full_number_crop(i)

        # Resize to 32x32
        crop_resized = resize_to_32x32(crop, maintain_aspect=True)

        # Normalize to [0, 1]
        crop_resized = crop_resized.astype(np.float32) / 255.0

        # Extract digit labels
        # In SVHN dataset: labels 1-9 are digits 1-9, label 10 is digit 0
        digits = [bbox['label'] % 10 for bbox in bboxes]  # Convert 10->0, keep 1-9
        num_digits = len(digits)

        # Clip to maximum 4 digits (model only supports 0-4)
        if num_digits > 4:
            digits = digits[:4]
            num_digits = 4

        # Pad to 4 digits with blank (10)
        while len(digits) < 4:
            digits.append(10)  # blank

        # Has digits flag
        has_digits = 1 if num_digits > 0 else 0

        images.append(crop_resized)
        labels_num.append(num_digits)  # 0-4
        labels_dig1.append(digits[0])
        labels_dig2.append(digits[1])
        labels_dig3.append(digits[2])
        labels_dig4.append(digits[3])
        labels_nC.append(has_digits)

    # Convert to numpy arrays
    images = np.array(images)

    # Convert to PyTorch tensors
    # Images: (N, H, W, C) -> (N, C, H, W)
    X = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    # Validate labels are in correct ranges
    labels_num_arr = np.array(labels_num)
    labels_dig1_arr = np.array(labels_dig1)
    labels_dig2_arr = np.array(labels_dig2)
    labels_dig3_arr = np.array(labels_dig3)
    labels_dig4_arr = np.array(labels_dig4)
    labels_nC_arr = np.array(labels_nC)

    assert labels_num_arr.min() >= 0 and labels_num_arr.max() <= 4, \
        f"num labels out of range [0,4]: min={labels_num_arr.min()}, max={labels_num_arr.max()}"
    assert labels_dig1_arr.min() >= 0 and labels_dig1_arr.max() <= 10, \
        f"dig1 labels out of range [0,10]: min={labels_dig1_arr.min()}, max={labels_dig1_arr.max()}"
    assert labels_dig2_arr.min() >= 0 and labels_dig2_arr.max() <= 10, \
        f"dig2 labels out of range [0,10]: min={labels_dig2_arr.min()}, max={labels_dig2_arr.max()}"
    assert labels_dig3_arr.min() >= 0 and labels_dig3_arr.max() <= 10, \
        f"dig3 labels out of range [0,10]: min={labels_dig3_arr.min()}, max={labels_dig3_arr.max()}"
    assert labels_dig4_arr.min() >= 0 and labels_dig4_arr.max() <= 10, \
        f"dig4 labels out of range [0,10]: min={labels_dig4_arr.min()}, max={labels_dig4_arr.max()}"
    assert labels_nC_arr.min() >= 0 and labels_nC_arr.max() <= 1, \
        f"nC labels out of range [0,1]: min={labels_nC_arr.min()}, max={labels_nC_arr.max()}"

    if verbose:
        print(f"  Label validation passed:")
        print(f"    num: [{labels_num_arr.min()}, {labels_num_arr.max()}]")
        print(f"    dig1-4: [{min(labels_dig1_arr.min(), labels_dig2_arr.min(), labels_dig3_arr.min(), labels_dig4_arr.min())}, "
              f"{max(labels_dig1_arr.max(), labels_dig2_arr.max(), labels_dig3_arr.max(), labels_dig4_arr.max())}]")
        print(f"    nC: [{labels_nC_arr.min()}, {labels_nC_arr.max()}]")

    labels = {
        'num': torch.from_numpy(labels_num_arr).long(),
        'dig1': torch.from_numpy(labels_dig1_arr).long(),
        'dig2': torch.from_numpy(labels_dig2_arr).long(),
        'dig3': torch.from_numpy(labels_dig3_arr).long(),
        'dig4': torch.from_numpy(labels_dig4_arr).long(),
        'nC': torch.from_numpy(labels_nC_arr).long(),
    }

    if verbose:
        print(f"\nData prepared:")
        print(f"  Images: {X.shape}")
        print(f"  Samples: {len(X)}")

    return X, labels


def train_epoch(model: nn.Module, train_loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: str, epoch: int) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_losses = {key: 0.0 for key in ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']}

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move to device
        data = data.to(device)
        targets = {key: val.to(device) for key, val in targets.items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)

        # Compute loss
        loss, losses = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for key in all_losses:
            if key in losses:
                all_losses[key] += losses[key]

    # Average losses
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_losses = {key: val / n_batches for key, val in all_losses.items()}

    return avg_loss, avg_losses


def validate(model: nn.Module, val_loader: DataLoader,
            criterion: nn.Module, device: str) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in val_loader:
            # Move to device
            data = data.to(device)
            targets = {key: val.to(device) for key, val in targets.items()}

            # Forward pass
            outputs = model(data)

            # Compute loss
            loss, _ = criterion(outputs, targets)
            total_loss += loss.item()

            # Store for metrics
            all_outputs.append(outputs)
            all_targets.append(targets)

    # Average loss
    avg_loss = total_loss / len(val_loader)

    # Concatenate all outputs and targets
    combined_outputs = {
        key: torch.cat([o[key] for o in all_outputs], dim=0)
        for key in all_outputs[0].keys()
    }
    combined_targets = {
        key: torch.cat([t[key] for t in all_targets], dim=0)
        for key in all_targets[0].keys()
    }

    # Compute metrics
    per_digit_acc, sequence_acc = measure_prediction(combined_outputs, combined_targets)

    return avg_loss, per_digit_acc, sequence_acc


def main():
    parser = argparse.ArgumentParser(description='Train CNN for SVHN digit recognition')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models_cnn',
                       help='Output directory for trained models')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum training samples (for quick testing)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='enhanced',
                       choices=['enhanced', 'basic'],
                       help='Type of model to train')
    parser.add_argument('--use_stn', action='store_true', default=True,
                       help='Use Spatial Transformer Network')
    parser.add_argument('--no_stn', dest='use_stn', action='store_false',
                       help='Disable STN')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')

    # Loss arguments
    parser.add_argument('--uncertainty_weighting', action='store_true',
                       help='Use uncertainty-based task weighting')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')

    # Checkpoint arguments
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("=" * 70)
    print("SVHN CNN Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Use STN: {args.use_stn}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Uncertainty weighting: {args.uncertainty_weighting}")

    # Load data
    print(f"\nLoading training data from {args.data_dir}...")
    train_loader_data = SVHNDataLoader(args.data_dir, split='train')

    print(f"\nLoading test data from {args.data_dir}...")
    test_loader_data = SVHNDataLoader(args.data_dir, split='test')

    # Prepare data
    X_train, y_train = prepare_data(train_loader_data, args.max_samples, verbose=True)
    X_test, y_test = prepare_data(test_loader_data, verbose=True)

    # Create data loaders
    train_dataset = TensorDataset(X_train, *y_train.values())
    test_dataset = TensorDataset(X_test, *y_test.values())

    # Custom collate function to create dict targets
    def collate_fn(batch):
        data = torch.stack([item[0] for item in batch])
        targets = {
            'num': torch.stack([item[1] for item in batch]),
            'dig1': torch.stack([item[2] for item in batch]),
            'dig2': torch.stack([item[3] for item in batch]),
            'dig3': torch.stack([item[4] for item in batch]),
            'dig4': torch.stack([item[5] for item in batch]),
            'nC': torch.stack([item[6] for item in batch]),
        }
        return data, targets

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # Create model
    print(f"\nCreating model...")
    model = create_model(args.model_type, input_channels=3, use_stn=args.use_stn)
    model = model.to(args.device)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Loss and optimizer
    criterion = MultiOutputLoss(use_uncertainty_weighting=args.uncertainty_weighting)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5)

    # Callbacks
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.output_dir, 'best_model.pth'),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Training loop
    print(f"\nStarting training...")
    print("=" * 70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_seq_acc': []
    }

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )

        # Validate
        val_loss, per_digit_acc, sequence_acc = validate(
            model, test_loader, criterion, args.device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Time
        epoch_time = time.time() - start_time

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracies: num={per_digit_acc[0]:.2f}%, dig1={per_digit_acc[1]:.2f}%, "
              f"dig2={per_digit_acc[2]:.2f}%, dig3={per_digit_acc[3]:.2f}%, dig4={per_digit_acc[4]:.2f}%, "
              f"nC={per_digit_acc[5]:.2f}%")
        print(f"  Sequence Accuracy: {sequence_acc:.2f}%")

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(np.mean(per_digit_acc))
        history['val_seq_acc'].append(sequence_acc)

        # Checkpoint
        metrics = {'val_loss': val_loss, 'val_seq_acc': sequence_acc}
        checkpoint(model, optimizer, epoch, metrics)

        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            save_model(model, optimizer, epoch, val_loss, save_path)

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    save_model(model, optimizer, args.epochs, val_loss, final_path,
               additional_info={'history': history})

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nBest sequence accuracy: {max(history['val_seq_acc']):.2f}%")
    print(f"Models saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
