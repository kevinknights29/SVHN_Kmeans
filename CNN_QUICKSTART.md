# CNN Quick Start Guide

Get the CNN model training in 5 minutes!

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

### 2. Verify Installation

```bash
python test_cnn.py
```

You should see:
```
Testing CNN Model...
============================================================
✓ CNN model imports successful
✓ Model created successfully
✓ Forward pass successful
✓ All tests passed!
```

## Training

### Quick Test (5-10 minutes)

```bash
python train_cnn.py \
    --data_dir data \
    --output_dir models_cnn_test \
    --max_samples 5000 \
    --epochs 10 \
    --batch_size 64
```

### Full Training (2-4 hours on GPU)

```bash
python train_cnn.py \
    --data_dir data \
    --output_dir models_cnn \
    --epochs 100 \
    --batch_size 32 \
    --device cuda
```

### What You'll See

```
Epoch 1/100 (45.2s)
  Train Loss: 2.3456
  Val Loss: 2.1234
  Val Accuracies: num=85.23%, dig1=87.45%, dig2=84.32%, dig3=82.11%, dig4=80.45%, nC=95.67%
  Sequence Accuracy: 65.43%

Epoch 50/100 (42.8s)
  Train Loss: 0.3214
  Val Loss: 0.4567
  Val Accuracies: num=97.82%, dig1=98.12%, dig2=97.45%, dig3=96.89%, dig4=95.23%, nC=99.45%
  Sequence Accuracy: 92.15%
  Checkpoint saved: models_cnn/best_model.pth
```

## Key Features

### 1. Spatial Transformer Network (STN)
- Automatically learns to correct rotation, scale, perspective
- Can be disabled with `--no_stn` to see impact

### 2. Inception-ResNet Architecture
- Multi-scale feature extraction (1x1, 3x3, 5x5 receptive fields)
- Residual connections for better gradient flow
- 4 layers: 64 → 128 → 256 → 512 channels

### 3. SE (Squeeze-and-Excitation) Blocks
- Channel-wise attention mechanism
- Adaptively weights feature channels

### 4. Multi-Task Learning
- 6 outputs trained simultaneously:
  - `num`: How many digits (0-4)
  - `dig1-dig4`: Each digit (0-9 + blank)
  - `nC`: Has any digits (yes/no)

## Command Options

### Basic Options
```bash
--data_dir data          # SVHN dataset location
--output_dir models_cnn  # Where to save models
--epochs 100             # Training epochs
--batch_size 32          # Batch size
--device cuda            # Use GPU
```

### Model Options
```bash
--model_type enhanced    # Architecture type
--use_stn                # Enable STN (default)
--no_stn                 # Disable STN
```

### Advanced Options
```bash
--lr 1e-3                  # Learning rate
--weight_decay 1e-4        # L2 regularization
--patience 15              # Early stopping patience
--uncertainty_weighting    # Learn task weights automatically
--max_samples 10000        # Limit training samples (for testing)
```

## Expected Results

| Training | Quick Test (5K samples) | Full (33K images) |
|----------|------------------------|-------------------|
| Time | 5-10 min | 2-4 hours |
| Sequence Accuracy | 70-80% | 85-95% |
| Model Size | ~35MB | ~35MB |

## Comparison with K-means

| Metric | K-means + SVM | CNN |
|--------|---------------|-----|
| Sequence Accuracy | ~85% | ~90-95% |
| Training Time | 35 min (with GPU) | 2-4 hours |
| Inference Speed | 0.5-2s/image | 5-20ms/image |
| Spatial Invariance | Limited | Strong (STN) |

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python train_cnn.py --batch_size 16
```

### "No module named 'torch'"
```bash
# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA not available"
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Train on CPU (slower)
python train_cnn.py --device cpu
```

### Training is slow
```bash
# Verify GPU is being used
nvidia-smi

# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Next Steps

1. **Run test to verify setup:**
   ```bash
   python test_cnn.py
   ```

2. **Quick test training:**
   ```bash
   python train_cnn.py --max_samples 5000 --epochs 10
   ```

3. **Full training:**
   ```bash
   python train_cnn.py --epochs 100
   ```

4. **Compare with K-means:**
   ```bash
   # Check K-means results from train.py
   # Compare sequence accuracy
   ```

## Files Created

After training:
```
models_cnn/
├── best_model.pth       # Best model (by validation loss)
├── final_model.pth      # Final model with training history
├── model_epoch_10.pth   # Checkpoint at epoch 10
├── model_epoch_20.pth   # Checkpoint at epoch 20
└── ...
```

## Using the Trained Model

```python
import torch
from src.svhn_kmeans.cnn_model import create_model

# Load model
model = create_model('enhanced', use_stn=True)
checkpoint = torch.load('models_cnn/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    outputs = model(image_tensor)
    predictions = {key: torch.argmax(val, dim=1)
                  for key, val in outputs.items()}
```

## Documentation

- **CNN_README.md**: Comprehensive guide with all details
- **CNN_IMPLEMENTATION.md**: Architecture design discussion
- **test_cnn.py**: Verification script

Happy training! 🚀
