#!/usr/bin/env python3
"""
Training script for SVHN K-means recognition system.

This script:
1. Loads the SVHN training dataset
2. Trains the K-means feature extractor
3. Trains the L2-SVM classifier
4. Evaluates on a validation set
5. Saves the trained model
"""

import argparse
import os
import numpy as np
from src.svhn_kmeans import SVHNDataLoader, SVHNRecognitionPipeline
from src.svhn_kmeans.utils import resize_to_32x32


def main():
    parser = argparse.ArgumentParser(description='Train SVHN recognition system')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for trained models')
    parser.add_argument('--K', type=int, default=500,
                       help='Number of K-means clusters (default: 500)')
    parser.add_argument('--beam_width', type=int, default=10,
                       help='Beam width for recognition (default: 10)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum training samples (for quick testing)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split fraction (default: 0.1)')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores, 1 for sequential)')
    args = parser.parse_args()

    print("=" * 70)
    print("SVHN K-means Recognition System - Training")
    print("=" * 70)

    # Load training data
    print(f"\nLoading training data from {args.data_dir}...")
    train_loader = SVHNDataLoader(args.data_dir, split='train')
    print(f"Total training images: {len(train_loader)}")

    # Create checkpoint directory early
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check for cached preprocessed data
    data_cache_file = os.path.join(checkpoint_dir, 'preprocessed_data.npz')

    if os.path.exists(data_cache_file):
        print(f"\n✓ Loading preprocessed data from cache: {data_cache_file}")
        cached = np.load(data_cache_file)
        train_images = cached['images']
        train_labels = cached['labels']
        print(f"✓ Loaded {len(train_images)} preprocessed digit crops (skipped data extraction)")
    else:
        # Prepare training data (extract individual digit crops)
        print("\nExtracting individual digit crops for training...")
        train_images = []
        train_labels = []

        n_samples = args.max_samples if args.max_samples else len(train_loader)

        for i in range(min(n_samples, len(train_loader))):
            if i % 5000 == 0:
                print(f"  Processing image {i}/{n_samples}")

            # Get digit crops from this image
            crops = train_loader.get_digit_crops(i)

            for crop, label in crops:
                # Resize to 32x32
                crop_resized = resize_to_32x32(crop, maintain_aspect=True)
                train_images.append(crop_resized)
                train_labels.append(label)

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        # Cache the preprocessed data
        print(f"\n✓ Caching preprocessed data to {data_cache_file}...")
        np.savez_compressed(data_cache_file, images=train_images, labels=train_labels)
        print(f"✓ Saved {len(train_images)} preprocessed digit crops")

    print(f"\nTotal digit crops: {len(train_images)}")
    print(f"Label distribution:")
    for digit in range(1, 11):
        count = np.sum(train_labels == digit)
        print(f"  Label {digit} (digit {digit if digit < 10 else 0}): {count}")

    # Split into train and validation
    n_val = int(len(train_images) * args.val_split)
    indices = np.random.permutation(len(train_images))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = train_images[train_indices]
    y_train = train_labels[train_indices]
    X_val = train_images[val_indices]
    y_val = train_labels[val_indices]

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Create and train pipeline
    print("\nInitializing pipeline...")
    print(f"GPU acceleration: {'Disabled' if args.no_gpu else 'Enabled (if available)'}")
    print(f"Parallel jobs: {args.n_jobs if args.n_jobs > 0 else 'All cores'}")

    from src.svhn_kmeans import KMeansFeatureExtractor, DigitClassifier

    feature_extractor = KMeansFeatureExtractor(
        K=args.K,
        use_gpu=not args.no_gpu,
        n_jobs=args.n_jobs
    )
    classifier = DigitClassifier(
        n_jobs=args.n_jobs,
        max_iter=2000  # Increased for better convergence
    )
    pipeline = SVHNRecognitionPipeline(
        feature_extractor,
        classifier,
        beam_width=args.beam_width
    )

    # Train
    print("\nStarting training...")
    print(f"Checkpoints directory: {checkpoint_dir}")
    pipeline.train(
        train_images=X_train,
        train_labels=y_train,
        n_samples_per_image=1000,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        cache_features=True
    )

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = pipeline.evaluate(
        test_images=X_val,
        test_labels=y_val,
        verbose=True
    )

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving model to {args.output_dir}...")
    pipeline.save(args.output_dir)

    # Save training info
    info_path = os.path.join(args.output_dir, 'training_info.txt')
    with open(info_path, 'w') as f:
        f.write("SVHN K-means Recognition System - Training Info\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"K (number of clusters): {args.K}\n")
        f.write(f"Beam width: {args.beam_width}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"\nValidation Accuracy: {val_results['accuracy']:.4f}\n")
        f.write("\nPer-class accuracies:\n")
        for digit, acc in sorted(val_results['per_class_accuracy'].items()):
            f.write(f"  Digit {digit}: {acc:.4f}\n")

    print(f"Training info saved to {info_path}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
