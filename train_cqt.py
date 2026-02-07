#!/usr/bin/env python3
"""
Character Query Transformer training script for SVHN digit detection.

This script trains a DETR-style transformer with learnable character queries
for end-to-end digit detection and recognition.
"""

import argparse
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.svhn_kmeans import SVHNDataLoader
from src.svhn_kmeans.cqt_model import create_cqt_model
from src.svhn_kmeans.cqt_utils import (
    CQTLoss,
    compute_accuracy,
    EarlyStopping,
    ModelCheckpoint,
    count_parameters,
    box_xyxy_to_cxcywh,
)
from src.svhn_kmeans.utils import resize_to_32x32


class SVHNDetectionDataset(Dataset):
    """
    SVHN dataset for detection (returns images with bboxes and labels).
    """

    def __init__(self, data_loader: SVHNDataLoader, max_samples: int = None,
                 image_size: int = 224, num_queries: int = 6):
        self.data_loader = data_loader
        self.image_size = image_size
        self.num_queries = num_queries

        self.n_samples = min(max_samples, len(data_loader)) if max_samples else len(data_loader)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get full house number crop with bounding boxes
        crop, bboxes = self.data_loader.get_full_number_crop(idx)

        # Resize image to fixed size
        orig_h, orig_w = crop.shape[:2]

        # Simple resize (will distort aspect ratio slightly)
        import cv2
        crop_resized = cv2.resize(crop, (self.image_size, self.image_size))

        # Normalize to [0, 1] and convert to tensor
        image = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet stats (for pretrained ResNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        # Process bounding boxes
        labels = []
        boxes = []

        for bbox in bboxes:
            # Label: 1-9 -> 1-9, 10 -> 0
            label = bbox['label'] % 10
            labels.append(label)

            # Box coordinates (convert to normalized [cx, cy, w, h])
            left = bbox['left']
            top = bbox['top']
            width = bbox['width']
            height = bbox['height']

            # Normalize coordinates
            x1 = left / orig_w
            y1 = top / orig_h
            x2 = (left + width) / orig_w
            y2 = (top + height) / orig_h

            # Clip to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))

            # Convert to [cx, cy, w, h]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            boxes.append([cx, cy, w, h])

        # Limit to max num_queries digits
        if len(labels) > self.num_queries:
            labels = labels[:self.num_queries]
            boxes = boxes[:self.num_queries]

        # Convert to tensors
        target = {
            'labels': torch.tensor(labels, dtype=torch.long),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
        }

        return image, target


def collate_fn(batch):
    """Custom collate function that doesn't stack targets."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def train_epoch(model, dataloader, criterion, optimizer, device, verbose=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_dict_sum = {}

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Compute loss
        loss, loss_dict = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_dict_sum[key] = loss_dict_sum.get(key, 0) + value

        if verbose and batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}")

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}

    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    loss_dict_sum = {}
    metrics_sum = {'accuracy': 0, 'recall': 0, 'precision': 0}

    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss, loss_dict = criterion(predictions, targets)

        # Accumulate losses
        total_loss += loss.item()
        for key, value in loss_dict.items():
            loss_dict_sum[key] = loss_dict_sum.get(key, 0) + value

        # Compute metrics
        metrics = compute_accuracy(predictions, targets)
        for key, value in metrics.items():
            metrics_sum[key] += value

    # Average
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}

    return avg_loss, avg_loss_dict, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Character Query Transformer for SVHN')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models_cqt',
                       help='Output directory for trained models')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum training samples (for quick testing)')

    # Model arguments
    parser.add_argument('--num_queries', type=int, default=6,
                       help='Maximum number of digits to detect')
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                       help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                       help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--pretrained_backbone', action='store_true', default=True,
                       help='Use pretrained backbone')
    parser.add_argument('--no_pretrained_backbone', dest='pretrained_backbone',
                       action='store_false',
                       help='Don\'t use pretrained backbone')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                       help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')

    # Loss arguments
    parser.add_argument('--weight_class', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--weight_bbox', type=float, default=5.0,
                       help='Weight for bbox L1 loss')
    parser.add_argument('--weight_giou', type=float, default=2.0,
                       help='Weight for bbox GIoU loss')
    parser.add_argument('--empty_weight', type=float, default=0.1,
                       help='Weight for empty class')

    # Device arguments
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')

    # Checkpoint arguments
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("=" * 70)
    print("Character Query Transformer Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Pretrained: {args.pretrained_backbone}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.nhead}")
    print(f"  Encoder layers: {args.num_encoder_layers}")
    print(f"  Decoder layers: {args.num_decoder_layers}")
    print(f"  Num queries: {args.num_queries}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Backbone LR: {args.lr_backbone}")
    print(f"  Epochs: {args.epochs}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading training data from {args.data_dir}...")
    train_loader_data = SVHNDataLoader(args.data_dir, split='train')

    print(f"\nLoading test data from {args.data_dir}...")
    test_loader_data = SVHNDataLoader(args.data_dir, split='test')

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = SVHNDetectionDataset(
        train_loader_data,
        max_samples=args.max_samples,
        image_size=args.image_size,
        num_queries=args.num_queries
    )
    test_dataset = SVHNDetectionDataset(
        test_loader_data,
        image_size=args.image_size,
        num_queries=args.num_queries
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Create model
    print("\nCreating model...")
    model = create_cqt_model(
        num_queries=args.num_queries,
        num_classes=11,  # 0-9 + empty
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
    )
    model = model.to(args.device)

    n_params = count_parameters(model)
    print(f"  Trainable parameters: {n_params:,}")

    # Loss and optimizer
    criterion = CQTLoss(
        num_classes=11,
        weight_class=args.weight_class,
        weight_bbox=args.weight_bbox,
        weight_giou=args.weight_giou,
        empty_weight=args.empty_weight,
    )
    criterion = criterion.to(args.device)  # Move criterion to device

    # Separate learning rates for backbone and rest
    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                   if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                   if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]

    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Callbacks
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    checkpoint = ModelCheckpoint(
        filepath=str(output_dir / 'best_model.pth'),
        monitor='val_loss',
        mode='min',
        verbose=True
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_recall': [],
        'val_precision': [],
    }

    for epoch in range(args.epochs):
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, args.device, verbose=False
        )

        # Validate
        val_loss, val_loss_dict, val_metrics = validate_epoch(
            model, test_loader, criterion, args.device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_precision'].append(val_metrics['precision'])

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Class: {train_loss_dict['loss_class']:.4f}, "
              f"BBox: {train_loss_dict['loss_bbox']:.4f}, "
              f"GIoU: {train_loss_dict['loss_giou']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    Class: {val_loss_dict['loss_class']:.4f}, "
              f"BBox: {val_loss_dict['loss_bbox']:.4f}, "
              f"GIoU: {val_loss_dict['loss_giou']:.4f}")
        print(f"  Val Metrics:")
        print(f"    Accuracy: {val_metrics['accuracy']:.2%}, "
              f"Recall: {val_metrics['recall']:.2%}, "
              f"Precision: {val_metrics['precision']:.2%}")

        # Checkpoint
        checkpoint(model, {'val_loss': val_loss}, epoch)

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            save_path = output_dir / f'model_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

        # Early stopping
        if early_stopping(val_loss):
            print("\nEarly stopping triggered!")
            break

    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
