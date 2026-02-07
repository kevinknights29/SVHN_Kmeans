"""
Training utilities for CNN models.

Includes loss functions, metrics, and training helpers.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from pathlib import Path


class MultiOutputLoss(nn.Module):
    """
    Combined loss for multi-output model with optional task weighting.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 use_uncertainty_weighting: bool = False):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        # Default equal weights
        self.weights = weights or {
            'num': 1.0,
            'dig1': 1.0,
            'dig2': 1.0,
            'dig3': 1.0,
            'dig4': 1.0,
            'nC': 1.0,
        }

        # Uncertainty weighting (learnable task weights)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(len(self.weights)))

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-output loss.

        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        losses = {}
        total_loss = 0.0

        for idx, key in enumerate(['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']):
            if key in outputs and key in targets:
                loss = self.criterion(outputs[key], targets[key])
                losses[key] = loss.item()

                if self.use_uncertainty_weighting:
                    # Uncertainty-weighted loss
                    precision = torch.exp(-self.log_vars[idx])
                    weighted_loss = precision * loss + self.log_vars[idx]
                    total_loss += weighted_loss
                else:
                    # Fixed weights
                    total_loss += self.weights[key] * loss

        return total_loss, losses


def measure_prediction(outputs: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> Tuple[List[float], float]:
    """
    Measure per-digit accuracy and sequence accuracy.

    Returns:
        Tuple of (per_digit_accuracies, sequence_accuracy)
    """
    output_keys = ['num', 'dig1', 'dig2', 'dig3', 'dig4', 'nC']
    per_digit_acc = []

    # Get predictions
    predictions = {key: torch.argmax(outputs[key], dim=1).cpu().numpy()
                   for key in output_keys if key in outputs}
    labels = {key: targets[key].cpu().numpy()
              for key in output_keys if key in targets}

    num_samples = len(labels['num'])

    # Calculate per-digit accuracies
    for key in output_keys:
        if key in predictions and key in labels:
            correct = np.sum(predictions[key] == labels[key])
            accuracy = (correct / num_samples) * 100
            per_digit_acc.append(accuracy)

    # Calculate sequence accuracy (all 4 digits must match)
    digit_keys = ['dig1', 'dig2', 'dig3', 'dig4']
    if all(k in predictions for k in digit_keys):
        predicted_array = np.stack([predictions[key] for key in digit_keys], axis=1)
        labels_array = np.stack([labels[key] for key in digit_keys], axis=1)

        all_correct = np.all(predicted_array == labels_array, axis=1)
        sequence_acc = (np.sum(all_correct) / num_samples) * 100
    else:
        sequence_acc = 0.0

    return per_digit_acc, sequence_acc


class EarlyStopping:
    """
    Early stopping to stop training when monitored metric stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"  Early stopping triggered!")

        return self.early_stop


class ModelCheckpoint:
    """
    Save model checkpoints based on monitored metric.
    """

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 verbose: bool = True):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = None

        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 epoch: int, metrics: Dict[str, float]):
        current_value = metrics.get(self.monitor)

        if current_value is None:
            return

        should_save = False

        if self.save_best_only:
            if self.best_value is None:
                should_save = True
            elif self.mode == 'min' and current_value < self.best_value:
                should_save = True
            elif self.mode == 'max' and current_value > self.best_value:
                should_save = True
        else:
            should_save = True

        if should_save:
            if self.save_best_only:
                self.best_value = current_value

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }

            torch.save(checkpoint, self.filepath)

            if self.verbose:
                print(f"  Checkpoint saved: {self.filepath}")


def enforce_task_consistency(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Post-process outputs to ensure logical consistency between tasks.

    - If presence = No, all digits should be blank
    - If length = N, only first N digits should be non-blank
    """
    # Get predictions
    presence = torch.argmax(outputs['nC'], dim=1)
    length = torch.argmax(outputs['num'], dim=1)

    # If no digits present, force all to blank (class 10)
    no_digit_mask = (presence == 0)
    for i in range(4):
        digit_key = f'dig{i+1}'
        if no_digit_mask.any():
            outputs[digit_key][no_digit_mask, 10] = 10.0
            outputs[digit_key][no_digit_mask, :10] = -10.0

    # Mask digits beyond sequence length
    for batch_idx, seq_len in enumerate(length):
        for digit_pos in range(seq_len.item(), 4):
            digit_key = f'dig{digit_pos+1}'
            outputs[digit_key][batch_idx, 10] = 10.0
            outputs[digit_key][batch_idx, :10] = -10.0

    return outputs


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer,
               epoch: int, loss: float, filepath: str,
               additional_info: Optional[Dict] = None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str,
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: str = 'cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)

    print(f"Model loaded from {filepath}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")

    return epoch, loss
