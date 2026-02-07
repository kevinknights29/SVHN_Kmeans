# CQT Quick Start Guide

Get the Character Query Transformer running in 5 minutes!

## What is CQT?

Character Query Transformer is a modern **DETR-style transformer** that:
- **Detects** digit locations (bounding boxes)
- **Recognizes** digit classes (0-9)
- **All in one shot** using learnable character queries

Think of it as: "YOLO meets Transformers for digit detection"

## Installation

### 1. Install PyTorch

**For CUDA 12.x (recommended for your server):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.x:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision
```

### 2. Install scipy (for Hungarian matching)

```bash
pip install scipy
```

### 3. Verify Installation

```bash
python test_cqt.py
```

You should see:
```
Testing Character Query Transformer Model...
============================================================
✓ CQT model imports successful
✓ Model created successfully
  Total trainable parameters: 28,547,915
✓ Forward pass successful
✓ Loss computation successful
✓ Metrics computation successful
✓ All tests passed!
```

## Training

### Quick Test (10-15 minutes)

```bash
python train_cqt.py \
    --data_dir data \
    --output_dir models_cqt_test \
    --max_samples 5000 \
    --epochs 20 \
    --batch_size 8
```

### Full Training (3-5 hours on GPU)

```bash
python train_cqt.py \
    --data_dir data \
    --output_dir models_cqt \
    --epochs 100 \
    --batch_size 16 \
    --device cuda
```

### What You'll See

```
Epoch 1/100 (245.3s)
  Train Loss: 12.3456
    Class: 2.1234, BBox: 0.5678, GIoU: 0.4321
  Val Loss: 11.2345
    Class: 2.0123, BBox: 0.5432, GIoU: 0.4123
  Val Metrics:
    Accuracy: 15.23%, Recall: 18.45%, Precision: 12.67%

Epoch 50/100 (238.7s)
  Train Loss: 2.1234
    Class: 0.3456, BBox: 0.0987, GIoU: 0.0654
  Val Loss: 2.3456
    Class: 0.3876, BBox: 0.1123, GIoU: 0.0723
  Val Metrics:
    Accuracy: 85.67%, Recall: 87.23%, Precision: 84.12%
  Checkpoint saved: models_cqt/best_model.pth
```

## Key Features

### 1. DETR-Style Detection
- Predicts **both** digit class and bounding box for each query
- No need for anchor boxes or NMS
- End-to-end differentiable

### 2. Hungarian Matching
- Optimal assignment between predictions and ground truth
- Handles variable number of digits (1-6)
- Bipartite matching based on class + bbox costs

### 3. Transformer Architecture
```
Image → ResNet50 → Flatten → Encoder → Decoder → Outputs
                      ↑                    ↑
                   Pos Embed         Query Embed
```

- **Encoder**: Processes image features with global context
- **Decoder**: 6 learnable queries attend to features
- **Outputs**: Each query predicts [class, bbox]

### 4. Multi-Loss Training
- **Classification Loss**: Which digit (0-9) or empty
- **BBox L1 Loss**: Box coordinate accuracy
- **BBox GIoU Loss**: Box overlap quality

## Command Options

### Basic Options
```bash
--data_dir data             # SVHN dataset location
--output_dir models_cqt     # Where to save models
--epochs 100                # Training epochs
--batch_size 16             # Batch size
--device cuda               # Use GPU
```

### Model Options
```bash
--num_queries 6             # Max digits to detect
--d_model 256               # Model dimension
--nhead 8                   # Attention heads
--num_encoder_layers 3      # Encoder depth
--num_decoder_layers 3      # Decoder depth
--pretrained_backbone       # Use pretrained ResNet (recommended)
```

### Advanced Options
```bash
--lr 1e-4                   # Learning rate
--lr_backbone 1e-5          # Lower LR for pretrained backbone
--weight_decay 1e-4         # L2 regularization
--patience 15               # Early stopping patience
--max_samples 10000         # Limit samples (for testing)
--image_size 224            # Input image size
```

### Loss Weights
```bash
--weight_class 1.0          # Classification loss weight
--weight_bbox 5.0           # BBox L1 loss weight
--weight_giou 2.0           # BBox GIoU loss weight
--empty_weight 0.1          # Empty class weight
```

## Expected Results

| Training | Quick Test (5K samples) | Full (33K images) |
|----------|------------------------|-------------------|
| Time | 10-15 min | 3-5 hours |
| Sequence Accuracy | 60-70% | 85-92% |
| Detection Recall | 65-75% | 85-90% |
| Detection Precision | 70-80% | 88-93% |
| Model Size | ~110MB | ~110MB |

## Comparison Table

| Metric | K-means + SVM | CNN | CQT |
|--------|--------------|-----|-----|
| Sequence Accuracy | ~85% | ~90-95% | ~85-92% |
| Training Time | 35 min | 2-4 hours | 3-5 hours |
| Inference Speed | 0.5-2s | 5-20ms | 20-50ms |
| Detects BBoxes? | Yes (seg) | No | Yes (direct) |
| Global Context? | No | Limited | Yes (attn) |
| Model Size | ~200MB | ~35MB | ~110MB |

## Why Use CQT?

### Advantages ✓
- **Joint Detection + Recognition**: One model does both
- **Global Context**: Transformer sees whole image
- **Modern Architecture**: Based on DETR (SOTA object detection)
- **Interpretable**: Attention maps show what model sees
- **Bounding Boxes**: Direct bbox predictions (normalized)

### When to Use
- You need both digit classes AND bounding boxes
- You want to try state-of-the-art transformers
- You have GPU resources (3-5 hours training is okay)
- You care about global context between digits

### When NOT to Use
- You only need digit sequence (use CNN - faster)
- Limited GPU memory (CNN is more efficient)
- Need fastest inference (K-means or CNN better)
- Working with very limited data (<5K samples)

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train_cqt.py --batch_size 8

# Or reduce model size
python train_cqt.py --d_model 128 --batch_size 16
```

### "ModuleNotFoundError: No module named 'scipy'"
```bash
# Install scipy for Hungarian matching
pip install scipy
```

### "CUDA not available"
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Train on CPU (slower)
python train_cqt.py --device cpu --batch_size 4
```

### Training is slow
```bash
# Verify GPU is being used
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Increase batch size if you have memory
python train_cqt.py --batch_size 32
```

### Poor accuracy
```bash
# Make sure pretrained backbone is enabled
python train_cqt.py --pretrained_backbone

# Train longer
python train_cqt.py --epochs 150 --patience 25

# Try different loss weights
python train_cqt.py --weight_bbox 10.0 --weight_giou 5.0
```

## Next Steps

1. **Run test to verify setup:**
   ```bash
   python test_cqt.py
   ```

2. **Quick test training:**
   ```bash
   python train_cqt.py --max_samples 5000 --epochs 20 --batch_size 8
   ```

3. **Full training:**
   ```bash
   python train_cqt.py --epochs 100 --batch_size 16
   ```

4. **Compare with other approaches:**
   - K-means: See `train.py` results
   - CNN: See `train_cnn.py` results
   - CQT: Current model

## Files Created

After training:
```
models_cqt/
├── best_model.pth          # Best model (by validation loss)
├── final_model.pth         # Final model with training history
├── model_epoch_10.pth      # Checkpoint at epoch 10
├── model_epoch_20.pth      # Checkpoint at epoch 20
└── ...
```

## Using the Trained Model

```python
import torch
from src.svhn_kmeans.cqt_model import create_cqt_model

# Load model
model = create_cqt_model(num_queries=6)
checkpoint = torch.load('models_cqt/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    predictions = model(image)

    # Get predictions
    class_logits = predictions['class_logits']  # [B, 6, 11]
    bbox_coords = predictions['bbox_coords']     # [B, 6, 4] (normalized)

    # Get top predictions
    scores, classes = class_logits[0].softmax(-1).max(-1)
    boxes = bbox_coords[0]

    # Filter empty class and low confidence
    keep = (classes != 10) & (scores > 0.5)
    final_classes = classes[keep]
    final_boxes = boxes[keep]

    print(f"Detected digits: {final_classes.tolist()}")
    print(f"Bounding boxes: {final_boxes.tolist()}")
```

## Documentation

- **CQT_README.md**: Comprehensive guide with all details
- **CQT_IMPLEMENTATION.md**: Architecture design discussion
- **test_cqt.py**: Verification script

## How It Works (Simple Explanation)

1. **Image → Features**: ResNet extracts visual features
2. **Add Position**: Tell model where each feature is in the image
3. **Encoder**: Process features with attention (see whole image)
4. **Decoder**: 6 queries ask "is there a digit here?"
5. **Outputs**: Each query predicts class (0-9 or empty) + bbox
6. **Matching**: Hungarian algorithm matches predictions to ground truth
7. **Loss**: Penalize wrong classes and bad bboxes

Think of it like 6 detectives (queries) searching the image for digits!

Happy training! 🚀
