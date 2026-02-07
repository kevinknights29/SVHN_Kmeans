"""
Utilities for Character Query Transformer training.

Includes:
- Hungarian matcher for bipartite matching
- Loss functions (classification, bbox L1, bbox GIoU)
- Box conversion utilities
- Training callbacks
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ============================================================================
# Box Utilities
# ============================================================================

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Tensor of shape [..., 4] in [cx, cy, w, h] format

    Returns:
        boxes_xyxy: Tensor of shape [..., 4] in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format.

    Args:
        boxes: Tensor of shape [..., 4] in [x1, y1, x2, y2] format

    Returns:
        boxes_cxcywh: Tensor of shape [..., 4] in [cx, cy, w, h] format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.

    Args:
        boxes: Tensor of shape [..., 4] in [x1, y1, x2, y2] format

    Returns:
        area: Tensor of shape [...]
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IoU and union between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        boxes2: Tensor of shape [M, 4] in [x1, y1, x2, y2] format

    Returns:
        iou: Tensor of shape [N, M]
        union: Tensor of shape [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)

    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized IoU (GIoU) between two sets of boxes.

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box.

    Args:
        boxes1: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
        boxes2: Tensor of shape [M, 4] in [x1, y1, x2, y2] format

    Returns:
        giou: Tensor of shape [N, M]
    """
    iou, union = box_iou(boxes1, boxes2)

    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-6)

    return giou


# ============================================================================
# Hungarian Matcher
# ============================================================================

class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for bipartite matching between predictions and targets.

    Computes matching based on:
    - Classification cost
    - L1 bounding box cost
    - GIoU bounding box cost
    """

    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.

        Args:
            predictions: Dictionary with:
                - 'class_logits': [B, num_queries, num_classes]
                - 'bbox_coords': [B, num_queries, 4]
            targets: List of B dictionaries with:
                - 'labels': [num_targets] (class labels)
                - 'boxes': [num_targets, 4] (normalized [cx, cy, w, h])

        Returns:
            indices: List of B tuples (pred_indices, target_indices)
        """
        B, num_queries = predictions['class_logits'].shape[:2]

        # Flatten batch dimension
        pred_logits = predictions['class_logits'].flatten(0, 1)  # [B*num_queries, num_classes]
        pred_boxes = predictions['bbox_coords'].flatten(0, 1)  # [B*num_queries, 4]

        # Compute class probabilities
        pred_probs = pred_logits.softmax(-1)  # [B*num_queries, num_classes]

        # Concatenate all target labels and boxes
        target_labels = torch.cat([t['labels'] for t in targets])
        target_boxes = torch.cat([t['boxes'] for t in targets])

        # Compute classification cost
        # Cost is -prob[target_class]
        cost_class = -pred_probs[:, target_labels]  # [B*num_queries, total_targets]

        # Compute L1 bbox cost
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)  # [B*num_queries, total_targets]

        # Compute GIoU cost
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)

        # Total cost matrix
        C = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        C = C.view(B, num_queries, -1).cpu()

        # Split costs by batch and perform Hungarian matching
        indices = []
        offset = 0
        for i, target in enumerate(targets):
            num_targets = len(target['labels'])
            if num_targets == 0:
                # No targets, no matching
                indices.append((torch.tensor([], dtype=torch.long),
                               torch.tensor([], dtype=torch.long)))
            else:
                c = C[i, :, offset:offset + num_targets]
                pred_idx, target_idx = linear_sum_assignment(c)
                indices.append((torch.as_tensor(pred_idx, dtype=torch.long),
                               torch.as_tensor(target_idx, dtype=torch.long)))
            offset += num_targets

        return indices


# ============================================================================
# Loss Functions
# ============================================================================

class CQTLoss(nn.Module):
    """
    Loss function for Character Query Transformer.

    Combines:
    - Classification loss (cross-entropy)
    - L1 bounding box loss
    - GIoU bounding box loss
    """

    def __init__(
        self,
        num_classes: int = 11,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        empty_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        # Matcher
        self.matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=weight_bbox,
            cost_giou=weight_giou
        )

        # Class weights (lower weight for empty class)
        self.register_buffer('class_weights',
                            torch.ones(num_classes))
        self.class_weights[-1] = empty_weight  # Empty class (10)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            predictions: Dictionary with:
                - 'class_logits': [B, num_queries, num_classes]
                - 'bbox_coords': [B, num_queries, 4]
            targets: List of B dictionaries with:
                - 'labels': [num_targets]
                - 'boxes': [num_targets, 4]

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Get matching indices
        indices = self.matcher(predictions, targets)

        # Prepare targets for each query
        B, num_queries = predictions['class_logits'].shape[:2]
        device = predictions['class_logits'].device

        # Classification loss
        target_classes = torch.full((B, num_queries), self.num_classes - 1,
                                   dtype=torch.long, device=device)  # Default: empty
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(target_idx) > 0:
                target_classes[i, pred_idx] = targets[i]['labels'][target_idx]

        loss_class = F.cross_entropy(
            predictions['class_logits'].flatten(0, 1),
            target_classes.flatten(),
            weight=self.class_weights
        )

        # Bounding box losses (only for matched queries)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        loss_bbox = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)

        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(target_idx) > 0:
                pred_boxes = predictions['bbox_coords'][i, pred_idx]
                target_boxes = targets[i]['boxes'][target_idx]

                # L1 loss
                loss_bbox += F.l1_loss(pred_boxes, target_boxes, reduction='sum')

                # GIoU loss
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
                giou = torch.diag(generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy))
                loss_giou += (1 - giou).sum()

        loss_bbox = loss_bbox / num_boxes
        loss_giou = loss_giou / num_boxes

        # Total loss
        loss = (
            self.weight_class * loss_class +
            self.weight_bbox * loss_bbox +
            self.weight_giou * loss_giou
        )

        loss_dict = {
            'loss_class': loss_class.item(),
            'loss_bbox': loss_bbox.item(),
            'loss_giou': loss_giou.item(),
            'loss_total': loss.item(),
        }

        return loss, loss_dict


# ============================================================================
# Metrics
# ============================================================================

def compute_accuracy(
    predictions: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute detection and classification accuracy.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        iou_threshold: IoU threshold for considering a detection correct

    Returns:
        metrics: Dictionary of metrics
    """
    B = len(targets)
    total_targets = sum(len(t['labels']) for t in targets)
    total_correct = 0
    total_detections = 0

    with torch.no_grad():
        # Get predicted classes and boxes
        pred_logits = predictions['class_logits']
        pred_boxes = predictions['bbox_coords']

        for i in range(B):
            target_labels = targets[i]['labels']
            target_boxes = targets[i]['boxes']

            if len(target_labels) == 0:
                continue

            # Get predictions (exclude empty class)
            scores, classes = pred_logits[i].max(-1)
            non_empty = classes != (pred_logits.shape[-1] - 1)

            pred_classes = classes[non_empty]
            pred_boxes_i = pred_boxes[i][non_empty]

            if len(pred_classes) == 0:
                continue

            total_detections += len(pred_classes)

            # Compute IoU
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_i)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
            iou_matrix, _ = box_iou(pred_boxes_xyxy, target_boxes_xyxy)

            # Match predictions to targets
            for j in range(len(pred_classes)):
                if len(iou_matrix[j]) == 0:
                    continue
                max_iou, max_idx = iou_matrix[j].max(0)
                if max_iou >= iou_threshold and pred_classes[j] == target_labels[max_idx]:
                    total_correct += 1

    metrics = {
        'accuracy': total_correct / max(total_detections, 1),
        'recall': total_correct / max(total_targets, 1),
        'precision': total_correct / max(total_detections, 1),
    }

    return metrics


# ============================================================================
# Training Callbacks
# ============================================================================

class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    """Model checkpoint callback."""

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None

    def __call__(self, model: nn.Module, metrics: Dict[str, float], epoch: int):
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.best is None:
            self.best = current
            self._save_checkpoint(model, epoch, current)
        elif self.mode == 'min' and current < self.best:
            self.best = current
            self._save_checkpoint(model, epoch, current)
        elif self.mode == 'max' and current > self.best:
            self.best = current
            self._save_checkpoint(model, epoch, current)

    def _save_checkpoint(self, model: nn.Module, epoch: int, value: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            self.monitor: value,
        }, self.filepath)

        if self.verbose:
            print(f"  Checkpoint saved: {self.filepath} ({self.monitor}={value:.4f})")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
