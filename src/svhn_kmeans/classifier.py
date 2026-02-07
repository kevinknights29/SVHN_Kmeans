"""
Digit classifier using L2-SVM (one-vs-all).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
from typing import List, Optional
import os
import time


class DigitClassifier:
    """
    One-vs-all L2 classifier for digit recognition.

    Uses 10 binary classifiers (one for each digit 0-9).
    Now uses LogisticRegression instead of LinearSVC for parallel processing support.
    """

    def __init__(self, C: float = 1.0, use_scaling: bool = True, n_jobs: int = -1,
                 max_iter: int = 1000):
        """
        Initialize classifier.

        Args:
            C: Regularization parameter (inverse of regularization strength)
            use_scaling: If True, standardize features before training
            n_jobs: Number of parallel jobs (-1 for all cores)
            max_iter: Maximum iterations for solver
        """
        self.C = C
        self.use_scaling = use_scaling
        self.n_jobs = n_jobs
        self.max_iter = max_iter

        # Use LogisticRegression with L2 penalty (supports n_jobs)
        # solver='lbfgs' is good for large datasets
        self.classifiers = [
            LogisticRegression(
                penalty='l2',
                C=C,
                max_iter=max_iter,
                solver='lbfgs',
                random_state=42,
                n_jobs=1,  # We'll parallelize at the digit level
                verbose=0
            ) for _ in range(10)
        ]
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False
        self.trained_digits = set()  # Track which digits are trained

    def train(self, features: np.ndarray, labels: np.ndarray, verbose: bool = True,
              checkpoint_dir: Optional[str] = None, resume: bool = True):
        """
        Train one-vs-all classifiers with checkpointing.

        Args:
            features: Feature matrix of shape (N, feature_dim)
            labels: Labels of shape (N,) with values 1-10 (1-9 for digits 1-9, 10 for digit 0)
            verbose: Print training progress
            checkpoint_dir: Directory to save checkpoints (None to disable)
            resume: If True, resume from checkpoint if available
        """
        if verbose:
            print(f"Training classifier on {len(features)} samples...")
            print(f"  Feature dimension: {features.shape[1]}")
            print(f"  Using {self.n_jobs if self.n_jobs > 0 else 'all'} CPU cores per classifier")
            print(f"  Max iterations: {self.max_iter}")
            if checkpoint_dir:
                print(f"  Checkpoints: {checkpoint_dir}")

        # Try to resume from checkpoint
        start_digit = 0
        if resume and checkpoint_dir and os.path.exists(checkpoint_dir):
            loaded = self._load_checkpoint(checkpoint_dir, verbose=verbose)
            if loaded:
                # Find which digits are already trained
                start_digit = len(self.trained_digits)
                if start_digit >= 10:
                    if verbose:
                        print("All classifiers already trained from checkpoint!")
                    self.is_trained = True
                    return
                if verbose:
                    print(f"Resuming from digit {start_digit}")

        # Scale features if enabled (only fit once)
        if self.use_scaling:
            if verbose:
                print("Scaling features...")
                start_time = time.time()

            if start_digit == 0:  # Only fit scaler if starting fresh
                features = self.scaler.fit_transform(features)
            else:
                features = self.scaler.transform(features)

            if verbose:
                print(f"  Scaling took {time.time() - start_time:.1f}s")

        # Train one-vs-all classifiers
        for digit in range(start_digit, 10):
            if verbose:
                print(f"\nTraining classifier for digit {digit}...")

            # Create binary labels for this digit
            # Note: Labels are 1-9 for digits 1-9, and 10 for digit 0
            target_label = digit if digit > 0 else 10
            binary_labels = (labels == target_label).astype(int)

            if verbose:
                n_positive = np.sum(binary_labels)
                n_negative = len(binary_labels) - n_positive
                print(f"  Positive samples: {n_positive}/{len(binary_labels)} ({100*n_positive/len(binary_labels):.1f}%)")
                print(f"  Negative samples: {n_negative}/{len(binary_labels)} ({100*n_negative/len(binary_labels):.1f}%)")

            # Train classifier
            if verbose:
                print(f"  Training (this may take 1-3 minutes)...")
                start_time = time.time()

            self.classifiers[digit].fit(features, binary_labels)
            self.trained_digits.add(digit)

            if verbose:
                elapsed = time.time() - start_time
                print(f"  ✓ Completed in {elapsed:.1f}s")

                # Check convergence
                if not self.classifiers[digit].n_iter_ < self.max_iter:
                    print(f"  ⚠ Warning: Reached max iterations ({self.max_iter}), may not have converged")

            # Save checkpoint after each digit
            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, digit, verbose=verbose)

        self.is_trained = True

        if verbose:
            print("\n" + "=" * 60)
            print("✓ All classifiers trained successfully!")
            print("=" * 60)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict digit labels.

        Args:
            features: Feature matrix of shape (N, feature_dim)

        Returns:
            Predicted labels (0-9)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet. Call train() first.")

        # Scale features if enabled
        if self.use_scaling:
            features = self.scaler.transform(features)

        # Get decision scores from all classifiers
        scores = self.decision_function(features)

        # Return class with highest score
        predictions = np.argmax(scores, axis=1)

        return predictions

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """
        Get decision scores for all classes.

        Args:
            features: Feature matrix of shape (N, feature_dim)

        Returns:
            Decision scores of shape (N, 10)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet. Call train() first.")

        # Scale features if needed (but not if already scaled in predict)
        if self.use_scaling and not isinstance(features, np.ndarray):
            features = self.scaler.transform(features)

        N = features.shape[0]
        scores = np.zeros((N, 10))

        for digit in range(10):
            scores[:, digit] = self.classifiers[digit].decision_function(features)

        return scores

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability-like scores (using softmax on decision scores).

        Args:
            features: Feature matrix of shape (N, feature_dim)

        Returns:
            Probability scores of shape (N, 10)
        """
        scores = self.decision_function(features)

        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """
        Evaluate classifier performance.

        Args:
            features: Feature matrix
            labels: True labels (1-10 format)

        Returns:
            Dictionary with accuracy and per-class accuracies
        """
        predictions = self.predict(features)

        # Convert labels from 1-10 format to 0-9 format
        true_labels = np.where(labels == 10, 0, labels)

        # Overall accuracy
        accuracy = np.mean(predictions == true_labels)

        # Per-class accuracy
        per_class_acc = {}
        for digit in range(10):
            mask = (true_labels == digit)
            if np.sum(mask) > 0:
                digit_acc = np.mean(predictions[mask] == true_labels[mask])
                per_class_acc[digit] = digit_acc

        return {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc
        }

    def _save_checkpoint(self, checkpoint_dir: str, digit: int, verbose: bool = True):
        """Save checkpoint after training a digit classifier."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_file = os.path.join(checkpoint_dir, f'classifier_digit_{digit}.pkl')
        scaler_file = os.path.join(checkpoint_dir, 'scaler.pkl')

        # Save this digit's classifier
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.classifiers[digit], f)

        # Save scaler (only once)
        if digit == 0 and self.scaler is not None:
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)

        if verbose:
            print(f"  Checkpoint saved: {checkpoint_file}")

    def _load_checkpoint(self, checkpoint_dir: str, verbose: bool = True) -> bool:
        """Load checkpoint from directory. Returns True if any checkpoints loaded."""
        if not os.path.exists(checkpoint_dir):
            return False

        loaded_any = False

        # Load scaler
        scaler_file = os.path.join(checkpoint_dir, 'scaler.pkl')
        if self.use_scaling and os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            if verbose:
                print(f"Loaded scaler from checkpoint")
            loaded_any = True

        # Load digit classifiers
        for digit in range(10):
            checkpoint_file = os.path.join(checkpoint_dir, f'classifier_digit_{digit}.pkl')
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    self.classifiers[digit] = pickle.load(f)
                self.trained_digits.add(digit)
                if verbose:
                    print(f"Loaded classifier for digit {digit}")
                loaded_any = True

        return loaded_any

    def save(self, filepath: str):
        """Save the trained classifier."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier.")

        data = {
            'C': self.C,
            'use_scaling': self.use_scaling,
            'n_jobs': self.n_jobs,
            'max_iter': self.max_iter,
            'classifiers': self.classifiers,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'trained_digits': self.trained_digits
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Classifier saved to {filepath}")

    def load(self, filepath: str):
        """Load a previously trained classifier."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.C = data['C']
        self.use_scaling = data['use_scaling']
        self.n_jobs = data.get('n_jobs', -1)  # Backward compatibility
        self.max_iter = data.get('max_iter', 1000)
        self.classifiers = data['classifiers']
        self.scaler = data['scaler']
        self.is_trained = data['is_trained']
        self.trained_digits = data.get('trained_digits', set(range(10)))

        print(f"Classifier loaded from {filepath}")
