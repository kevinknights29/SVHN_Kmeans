# SVHN Recognition System

End-to-end system for Street View House Number (SVHN) digit recognition with four different approaches:
- **K-means + SVM** (classical approach)
- **CNN with STN** (deep learning approach)
- **Character Query Transformer** (modern transformer approach)
- **YOLO + CNN** (production-ready approach)

## Overview

This project provides four complete implementations for SVHN digit recognition:

### 1. K-means + SVM (Classical Approach)

A two-stage pipeline using classical computer vision and machine learning:

1. **Feature Extraction**: K-means clustering learns a dictionary of 500 filters from 8×8 image patches
2. **Classification**: L2-SVM (one-vs-all) classifies individual digits
3. **Segmentation**: Character segmentation using vertical projection analysis
4. **Recognition**: Beam search combines segmentation hypotheses with classification scores

Achieves ~85% sequence accuracy with ~35 min training time (with GPU acceleration).

### 2. CNN with STN (Deep Learning Approach)

Modern deep learning approach with:
- **Spatial Transformer Network (STN)** for geometric invariance
- **Inception-ResNet blocks** for multi-scale feature extraction
- **SE blocks** for channel-wise attention
- **Multi-task learning** with 6 output heads

Achieves ~90-95% sequence accuracy with 2-4 hours training time.

### 3. Character Query Transformer (CQT)

State-of-the-art transformer-based approach:
- **DETR-style architecture** with learnable character queries
- **Joint detection + recognition** (predicts classes and bboxes)
- **Hungarian matching** for optimal assignment
- **ResNet50 backbone** with transformer encoder/decoder

Achieves ~85-92% sequence accuracy with 3-5 hours training time. Provides bounding box predictions.

### 4. YOLO + CNN (Production-Ready Approach)

End-to-end two-stage pipeline for real-world deployment:
- **YOLOv8 detection** for localizing house numbers in full images
- **CNN recognition** for classifying detected digit sequences
- **Modular design** with decoupled detection and recognition
- **Pretrained backbone** from COCO dataset for better initialization

Achieves ~80-85% end-to-end accuracy with 3-6 hours training time. Fast inference (10-35ms) for production deployment.

## Features

- **K-means Feature Learning**: Unsupervised learning of convolutional filters from image patches
- **Spatial Pyramid Pooling**: 5×5 grid pooling for translation invariance
- **Beam Search**: Explores multiple segmentation hypotheses simultaneously
- **Geometric Scoring**: Models expected character sizes and aspect ratios
- **YOLOv8 Detection**: Single-stage object detection for real-time localization
- **Modular Pipeline**: Decoupled detection and recognition stages
- **Visualization**: Draws bounding boxes and predicted digits on images

## Project Structure

```
SVHN_Kmeans/
├── src/
│   └── svhn_kmeans/
│       ├── __init__.py            # Package initialization
│       ├── data_loader.py         # SVHN dataset loading
│       ├── utils.py               # Image processing utilities
│       # K-means approach
│       ├── feature_extractor.py   # K-means feature extraction
│       ├── classifier.py          # L2-SVM classifier
│       ├── segmentation.py        # Character segmentation
│       ├── recognizer.py          # Beam search recognizer
│       ├── pipeline.py            # End-to-end pipeline
│       # CNN approach
│       ├── cnn_model.py           # CNN model (STN, Inception-ResNet, SE)
│       ├── cnn_utils.py           # CNN training utilities
│       # CQT approach
│       ├── cqt_model.py           # Character Query Transformer
│       └── cqt_utils.py           # CQT training utilities (Hungarian matcher, losses)
├── data/
│   ├── train/                     # Training images and annotations
│   └── test/                      # Test images and annotations
# Training scripts
├── train.py                       # K-means + SVM training
├── train_cnn.py                   # CNN training
├── train_cqt.py                   # CQT training
├── train_yolo.py                  # YOLO training
├── convert_svhn_to_yolo.py        # SVHN to YOLO format converter
# Test scripts
├── test_cnn.py                    # CNN verification
├── test_cqt.py                    # CQT verification
├── test_yolo.py                   # YOLO verification
# Inference scripts
├── inference_yolo.py              # YOLO + CNN end-to-end inference
# Documentation
├── IMPLEMENTATION.md              # K-means implementation guide
├── CNN_IMPLEMENTATION.md          # CNN architecture design
├── CNN_README.md                  # CNN comprehensive guide
├── CNN_QUICKSTART.md              # CNN quick start
├── CQT_IMPLEMENTATION.md          # CQT architecture strategy
├── CQT_README.md                  # CQT comprehensive guide
├── CQT_QUICKSTART.md              # CQT quick start
├── YOLO_README.md                 # YOLO comprehensive guide
├── YOLO_QUICKSTART.md             # YOLO quick start
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.9-3.13
- uv (recommended) or pip

### Install Dependencies

Using uv (recommended):
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# For K-means + SVM approach
uv pip install numpy scipy scikit-learn opencv-python matplotlib Pillow h5py joblib

# For deep learning approaches (CNN, CQT, YOLO)
uv pip install torch torchvision ultralytics
```

Using pip:
```bash
# For K-means + SVM approach
pip install numpy scipy scikit-learn opencv-python matplotlib Pillow h5py joblib

# For deep learning approaches (CNN, CQT, YOLO)
pip install torch torchvision ultralytics
```

### GPU Acceleration

**For K-means approach (CuPy - optional):**

10-30x speedup on NVIDIA GPUs for K-means feature extraction:

```bash
# For CUDA 12.x (e.g., RTX 40xx, H100, newer servers)
pip install cupy-cuda12x

# For CUDA 11.x (e.g., A10, A100, RTX 30xx)
pip install cupy-cuda11x
```

**Note:** Without CuPy, the system will automatically use CPU with parallel processing (still 4-9x faster than sequential).

**For CNN, CQT, and YOLO approaches (PyTorch - required):**

```bash
# For CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

**Check your CUDA version:**
```bash
nvidia-smi  # Look for "CUDA Version: X.Y"
```

## Quick Start

Choose your approach based on your needs:

| Approach | When to Use | Quick Start |
|----------|------------|-------------|
| **K-means + SVM** | Fast training, interpretable, don't need PyTorch | See [QUICKSTART.md](QUICKSTART.md) |
| **CNN** | Best accuracy, need digit sequences only | See [CNN_QUICKSTART.md](CNN_QUICKSTART.md) |
| **CQT** | Need bounding boxes, want transformer architecture | See [CQT_QUICKSTART.md](CQT_QUICKSTART.md) |
| **YOLO + CNN** | Production deployment, full images, real-time | See [YOLO_QUICKSTART.md](YOLO_QUICKSTART.md) |

## Usage

### 1. K-means + SVM Training

Train the classical pipeline:

```bash
python train.py --data_dir data --output_dir models
```

Options:
- `--data_dir`: Path to SVHN data directory (default: `data`)
- `--output_dir`: Output directory for trained models (default: `models`)
- `--K`: Number of K-means clusters (default: 500)
- `--beam_width`: Beam width for recognition (default: 10)
- `--max_samples`: Limit training samples for quick testing
- `--val_split`: Validation split fraction (default: 0.1)
- `--no_gpu`: Disable GPU acceleration (use CPU only)
- `--n_jobs`: Number of parallel jobs (-1 for all cores, 1 for sequential)

### 2. CNN Training

Train the CNN model:

```bash
# Quick test (5-10 minutes)
python train_cnn.py --max_samples 5000 --epochs 10

# Full training (2-4 hours)
python train_cnn.py --data_dir data --output_dir models_cnn --epochs 100
```

Test model first:
```bash
python test_cnn.py
```

### 3. CQT Training

Train the transformer model:

```bash
# Quick test (10-15 minutes)
python train_cqt.py --max_samples 5000 --epochs 20 --batch_size 8

# Full training (3-5 hours)
python train_cqt.py --data_dir data --output_dir models_cqt --epochs 100
```

Test model first:
```bash
python test_cqt.py
```

### 4. YOLO + CNN Training

Train the YOLO detector and use existing CNN recognizer:

```bash
# Step 1: Convert SVHN to YOLO format (5-10 minutes)
python convert_svhn_to_yolo.py --data_dir data --output_dir data_yolo

# Step 2: Test YOLO setup
python test_yolo.py

# Step 3: Train YOLO (quick test: 5-10 minutes)
python train_yolo.py --data data_yolo/dataset.yaml --epochs 10 --imgsz 320

# Full training (1-2 hours)
python train_yolo.py --data data_yolo/dataset.yaml --epochs 100
```

**Note**: CNN model is required for recognition. Train it first if needed:
```bash
python train_cnn.py --epochs 100
```

End-to-end inference:
```bash
python inference_yolo.py \
    --yolo_model models_yolo/train/weights/best.pt \
    --cnn_model models_cnn \
    --indices 1 10 100
```

### Inference

Process images and visualize results:

#### Process specific images from dataset:
```bash
python inference.py --model_dir models --indices 1 10 100 --split test
```

#### Process a custom image:
```bash
python inference.py --model_dir models --image path/to/image.png --output results
```

Options:
- `--model_dir`: Directory containing trained model (default: `models`)
- `--image`: Path to input image file (alternative to dataset)
- `--data_dir`: Path to SVHN dataset (default: `data`)
- `--split`: Dataset split to use: `train` or `test` (default: `test`)
- `--indices`: Image indices to process from dataset (default: 1 10 100)
- `--output`: Output directory for results (default: `output`)
- `--no_display`: Don't display images, only save
- `--beam_width`: Beam width for recognition (default: 10)

### Programmatic Usage

```python
from src.svhn_kmeans import SVHNRecognitionPipeline
from PIL import Image
import numpy as np

# Load trained model
pipeline = SVHNRecognitionPipeline()
pipeline.load('models')

# Load image
image = np.array(Image.open('house_number.png').convert('RGB'))

# Recognize digits
result = pipeline.process_image(image)

print(f"Predicted digits: {result['digits']}")
print(f"House number: {''.join(str(d) for d in result['digits'])}")
print(f"Bounding boxes: {result['bboxes']}")

# Visualize
import matplotlib.pyplot as plt
plt.imshow(result['visualization'])
plt.show()
```

## System Architecture

### 1. K-means Feature Extraction

- Extracts 8×8 patches from grayscale images
- Learns K=500 cluster centroids as dictionary filters
- Convolves filters with input images
- Applies non-linearity: g(z) = max{0, |z| - α}
- Performs 5×5 spatial average pooling
- Output: 12,500-dimensional feature vector (500 × 5 × 5)

### 2. L2-SVM Classification

- One-vs-all approach: 10 binary classifiers for digits 0-9
- L2 regularization for generalization
- Optional feature standardization
- Decision function outputs used for scoring

### 3. Character Segmentation

- Computes vertical projection profile
- Finds local minima as character boundaries
- Generates candidate segments between breakpoints
- Filters by minimum/maximum character width

### 4. Beam Search Recognition

- Maintains top-K hypotheses (beam)
- Scores combine:
  - Classifier confidence (70%)
  - Geometric constraints (30%)
- Explores character segmentation incrementally
- Returns top-5 complete sequences

## Algorithm Details

### Feature Extraction Pipeline

For a 32×32 input image:

1. Convert to grayscale
2. Convolve with 500 learned 8×8 filters → 500 × 25×25 activations
3. Apply thresholded absolute value: max(0, |z| - α)
4. Average pool in 5×5 grid → 500 × 5×5 features
5. Flatten → 12,500-dimensional vector

### Training Process

1. **Dictionary Learning** (2-3 hours on full dataset):
   - Extract ~1000 patches per training image
   - Run mini-batch K-means with K=500
   - Normalize centroids to unit norm
   - Learn threshold α from mean activation

2. **Classifier Training** (10-20 minutes):
   - Extract features from all training digits
   - Train 10 binary L2-SVM classifiers
   - Standardize features for numerical stability

### Beam Search Algorithm

```
beam = [{digits: [], bboxes: [], score: 0, position: 0}]

while beam not empty:
    for each hypothesis in beam:
        for each next breakpoint:
            segment = extract_segment(image, current_bp, next_bp)
            features = extract_features(resize(segment, 32x32))
            scores = classify(features)  # 10 scores

            for top-3 digit classes:
                new_hypothesis = extend(hypothesis, digit, score)
                if complete:
                    add to final_paths
                else:
                    add to new_beam

    beam = top_K(new_beam, K=beam_width)

return top_5(final_paths)
```

## Performance

Experimental results on SVHN dataset (validation set):

### Model Comparison

| Model | Overall Accuracy | Training Time | Notes |
|-------|-----------------|---------------|-------|
| **K-means + SVM** | **88.38%** | ~35 min (GPU) | Per-digit classification |
| **CNN (STN)** | **84.75%** | ~18 min (epoch 35) | Sequence accuracy, early stopped |
| **CQT (Transformer)** | **89.47%*** | ~3.5 hours (epoch 44) | Detection accuracy |
| **YOLO + CNN** | **80.59%** (end-to-end) | ~24 min (epoch 10) | Sequence accuracy, full pipeline |

\* CQT metrics show high recall (160%), indicating multiple detections per digit - under investigation.

### Detailed Results

#### K-means + SVM
- **Overall Accuracy**: 88.38%
- **Training Time**: ~35 minutes (with GPU acceleration)
- **Per-Digit Accuracy**:
  - Digit 0: 88.87%
  - Digit 1: 91.98%
  - Digit 2: 92.21%
  - Digit 3: 85.13%
  - Digit 4: 91.64%
  - Digit 5: 85.17%
  - Digit 6: 84.83%
  - Digit 7: 87.84%
  - Digit 8: 84.79%
  - Digit 9: 83.08%

#### CNN with Spatial Transformer Network
- **Sequence Accuracy**: 84.75% (all digits correct)
- **Training**: Stopped at epoch 35/100 (early stopping)
- **Val Accuracies**:
  - Number of digits: 96.37%
  - Digit 1: 93.37%
  - Digit 2: 91.61%
  - Digit 3: 95.31%
  - Digit 4: 99.46%
  - Has digits (nC): 100.00%
- **Training Loss**: 0.3615
- **Validation Loss**: 1.1723

#### Character Query Transformer (DETR-style)
- **Detection Accuracy**: 89.47%
- **Precision**: 89.47%
- **Recall**: 160.83% (indicating multiple detections - metrics under review)
- **Training**: Stopped at epoch 44/100 (early stopping)
- **Loss Components**:
  - Classification: 0.7058
  - BBox L1: 0.1043
  - BBox GIoU: 0.1235
- **Training Time**: ~3.5 hours (epoch 44)

#### YOLO + CNN (Two-Stage Pipeline)
- **YOLO Detection Performance** (epoch 10):
  - **mAP50**: 99.5% (mean Average Precision at IoU=0.5)
  - **mAP50-95**: 99.47% (mean AP at IoU=0.5:0.95)
  - **Precision**: 99.996%
  - **Recall**: 100%
- **End-to-End Performance** (13,068 test images):
  - **Detection Rate**: 99.68% (13,026/13,068 images)
  - **Sequence Accuracy**: 80.59% (10,531/13,068 images)
  - **Digit Accuracy**: 88.61% (23,067/26,032 digits)
- **Training**: Stopped at epoch 10/100 (early stopping)
- **Loss Components** (validation):
  - Box Loss: 0.10454
  - Classification Loss: 0.19214
  - DFL Loss: 0.42273
- **Training Time**: ~24 minutes (epoch 10, YOLO only)

### Performance Summary

| Metric | K-means + SVM | CNN | CQT | YOLO + CNN |
|--------|--------------|-----|-----|------------|
| Accuracy | 88.38% | 84.75%* | 89.47%** | 80.59%* |
| Training Time | 35 min | 18 min | 3.5 hours | 24 min (YOLO only) |
| Inference Speed | 0.5-2s | 5-20ms | 20-50ms | ~28ms |
| Model Size | ~200MB | ~35MB | ~110MB | ~38MB |
| GPU Required | No (optional) | Yes | Yes | Yes |
| Outputs Bboxes | No | No | Yes | Yes |
| Detection Rate | N/A | N/A | N/A | 99.68% |

\* Sequence accuracy (all digits must be correct)
\** Detection accuracy (may include duplicates)

### Hardware Configuration

All experiments run on:
- **GPU**: NVIDIA A10 (24GB)
- **CPU**: 30 cores
- **CUDA**: 12.8
- **Dataset**: SVHN train set (~73K digit crops)

## Dataset

The system uses the SVHN (Street View House Numbers) dataset:

- **Format**: PNG images with .mat annotation files
- **Training**: ~33,000 images with ~73,000 digit crops
- **Test**: ~13,000 images with ~26,000 digit crops
- **Labels**: 1-9 for digits 1-9, 10 for digit 0
- **Annotations**: Bounding boxes (left, top, width, height) for each digit

## Limitations and Future Work

### Current Limitations

**K-means + SVM:**
1. **No Detection Stage**: Assumes input is pre-cropped house number (✓ Addressed in YOLO + CNN)
2. **Fixed Architecture**: K-means features are less powerful than learned CNNs
3. **Segmentation Challenges**: Struggles with touching or overlapping digits
4. **Computational Cost**: Convolving 500 filters is expensive

**YOLO + CNN:**
1. **Two-Stage Pipeline**: Requires training and running two separate models
2. **Error Propagation**: Detection errors affect recognition accuracy
3. **Not End-to-End**: No joint optimization of detection and recognition

### Potential Improvements

1. **Unified End-to-End Model**: Integrate YOLO and CNN into single trainable model
2. **Attention Mechanisms**: Add cross-attention between detection and recognition
3. **Multi-Task Learning**: Joint training with shared backbone
4. **Data Augmentation**: More aggressive augmentation for robustness
5. **Model Distillation**: Compress models for edge deployment

## References

Based on the approach described in:
- "Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks"
- K-means feature learning literature
- Beam search for sequence recognition

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Areas for improvement:

- GPU acceleration
- Additional datasets
- Modern detection integration
- Performance optimizations
- Extended documentation

## Contact

For questions or issues, please open an issue on the project repository.
