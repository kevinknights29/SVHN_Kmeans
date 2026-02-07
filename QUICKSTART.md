# Quick Start Guide

Get started with the SVHN K-means recognition system in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- SVHN dataset in the `data/` directory

## Setup (2 minutes)

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv pip install numpy scipy scikit-learn opencv-python matplotlib Pillow h5py joblib

# Or using pip
pip install numpy scipy scikit-learn opencv-python matplotlib Pillow h5py joblib

# Optional: GPU acceleration (10-30x speedup)
# Check your CUDA version first: nvidia-smi
pip install cupy-cuda12x  # For CUDA 12.x
# pip install cupy-cuda11x  # For CUDA 11.x
```

### 2. Verify Dataset

Your `data/` directory should look like:
```
data/
├── train/
│   ├── 1.png
│   ├── 2.png
│   ├── ...
│   └── digitStruct.mat
└── test/
    ├── 1.png
    ├── 2.png
    ├── ...
    └── digitStruct.mat
```

## Quick Test (1 minute)

### Option 1: Test with Pre-trained Model

If you have a pre-trained model in `models/`:

```bash
python inference.py --model_dir models --indices 1 10 100
```

### Option 2: Run Quick Training

Train a smaller model for testing (10-15 minutes):

```bash
python train.py --max_samples 5000 --K 200 --output_dir models_test
```

Then run inference:

```bash
python inference.py --model_dir models_test --indices 1
```

## Full Training (2-4 hours)

For best accuracy, train on the full dataset:

```bash
python train.py --K 500 --output_dir models
```

Parameters:
- `--K 500`: Use 500 K-means clusters (as in the paper)
- `--output_dir models`: Save model to `models/` directory

## Running Inference

### Process Test Images

```bash
# Process specific images from the test set
python inference.py --model_dir models --indices 1 10 100 200

# Process from train set
python inference.py --model_dir models --split train --indices 5 50 500
```

### Process Your Own Image

```bash
python inference.py --model_dir models --image path/to/house_number.png
```

### Save Results Without Display

```bash
python inference.py --model_dir models --indices 1 10 100 --no_display --output results/
```

## Using the Example Script

Run the interactive example:

```bash
python example.py
```

This will:
1. Load the trained model
2. Process a test image
3. Show ground truth vs prediction side-by-side
4. Display bounding boxes for each digit

## Expected Results

With K=500 and full training:
- Single digit accuracy: ~90%
- Training time: 2-4 hours
- Inference: ~0.5-2 seconds per image

With K=200 and 5000 samples (quick test):
- Single digit accuracy: ~80-85%
- Training time: 10-15 minutes
- Inference: ~0.5 seconds per image

## Using in Python

```python
from src.svhn_kmeans import SVHNRecognitionPipeline
from PIL import Image
import numpy as np

# Load model
pipeline = SVHNRecognitionPipeline()
pipeline.load('models')

# Process image
image = np.array(Image.open('house.png').convert('RGB'))
result = pipeline.process_image(image)

# Get results
house_number = ''.join(str(d) for d in result['digits'])
print(f"House number: {house_number}")
```

## Troubleshooting

### "Model not found" error
- Train a model first: `python train.py --max_samples 5000`

### "Dataset not found" error
- Ensure SVHN dataset is in `data/train/` and `data/test/`
- Check that `digitStruct.mat` files exist

### Low accuracy
- Use more training samples (`--max_samples` or remove for full dataset)
- Increase K-means clusters (`--K 500` instead of 200)
- Train longer (default should be sufficient)

### Slow inference
- Reduce beam width: `--beam_width 5`
- This trades some accuracy for speed

### Import errors
- Make sure all dependencies are installed
- Run from the project root directory

## Next Steps

1. **Experiment with parameters**:
   - Try different K values (200, 300, 500, 1000)
   - Adjust beam width (5, 10, 20)
   - Modify score weights in the code

2. **Evaluate performance**:
   - Check `models/training_info.txt` for validation accuracy
   - Run inference on multiple test images
   - Compare predictions with ground truth

3. **Extend the system**:
   - Add a detection stage for full images
   - Implement data augmentation
   - Try different classifiers (SVM kernels, neural networks)

## Resources

- **Full Documentation**: See `README.md`
- **Code Examples**: See `example.py`

## Support

If you encounter issues:
1. Check the error message carefully
2. Verify dataset and dependencies
3. Try the quick test first (5000 samples)
4. Review the troubleshooting section above

Happy recognizing! 🎯
