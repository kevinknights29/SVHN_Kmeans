# YOLO + CNN Quick Start (5 minutes)

Get started with fine-tuned YOLO for SVHN house number detection and recognition.

## Overview

This approach combines:
1. **YOLOv8**: Fast object detector for localizing house numbers
2. **CNN**: Multi-digit recognizer for classifying detected house numbers

Perfect for real-world deployment where you need to process full images.

## Prerequisites

```bash
# Install dependencies
uv pip install ultralytics torch torchvision opencv-python matplotlib Pillow PyYAML

# Or with pip
pip install ultralytics torch torchvision opencv-python matplotlib Pillow PyYAML
```

## Step 1: Test Setup (30 seconds)

Verify YOLOv8 installation:

```bash
python test_yolo.py
```

**Expected output:**
```
Test 1: Loading YOLO model (yolov8n.pt)...
  ✓ Model loaded successfully
Test 2: Inference on dummy input...
  ✓ Inference successful
...
✓ All tests passed!
```

## Step 2: Convert Data (5-10 minutes)

Convert SVHN dataset to YOLO format:

```bash
python convert_svhn_to_yolo.py --data_dir data --output_dir data_yolo
```

**What it does:**
- Computes house number bounding boxes from individual digit bboxes
- Converts to YOLO format (normalized [cx, cy, w, h])
- Creates train/val/test splits with images/ and labels/ subdirectories
- Generates dataset.yaml configuration file

**Output:**
```
data_yolo/
├── train/images/ (~29,700 images)
├── train/labels/ (~29,700 labels)
├── val/images/ (~3,300 images)
├── val/labels/ (~3,300 labels)
├── test/images/ (~13,068 images)
├── test/labels/ (~13,068 labels)
└── dataset.yaml
```

## Step 3: Train YOLO (1-2 hours)

### Quick Test (5-10 minutes)

```bash
python train_yolo.py --data data_yolo/dataset.yaml --epochs 10 --imgsz 320
```

### Full Training (1-2 hours)

```bash
python train_yolo.py --data data_yolo/dataset.yaml --epochs 100
```

**Training options:**
```bash
--model yolov8n.pt        # Nano (fastest, default)
--model yolov8s.pt        # Small (more accurate)
--batch_size 16           # Batch size (default: 16)
--imgsz 640               # Input size (default: 640)
--device 0                # GPU 0 (default)
```

**Output:**
```
models_yolo/
└── train/
    ├── weights/
    │   ├── best.pt       # Best model
    │   └── last.pt       # Last epoch
    ├── results.png       # Training curves
    └── ...
```

**Training progress:**
```
Epoch   Box Loss   Cls Loss   DFL Loss   Precision   Recall   mAP50
  1/100   1.234      0.567      0.890      0.456      0.567    0.423
 10/100   0.543      0.234      0.456      0.823      0.845    0.812
...
100/100   0.234      0.123      0.234      0.901      0.913    0.894
```

## Step 4: Train CNN (2-4 hours)

If you haven't already trained the CNN recognizer:

```bash
# Quick test (5-10 minutes)
python train_cnn.py --max_samples 5000 --epochs 10

# Full training (2-4 hours)
python train_cnn.py --epochs 100
```

**Output:**
```
models_cnn/
├── best_model.pth        # Best model
└── ...
```

## Step 5: Run Inference

### Test on Dataset Images

```bash
python inference_yolo.py \
    --yolo_model models_yolo/train/weights/best.pt \
    --cnn_model models_cnn \
    --indices 1 10 100
```

**Output:**
```
Image 1:
  Ground truth: 2468
  Inference time: 0.042s
  Detections: 1
    ✓ 1. Sequence: 2468, YOLO conf: 0.942, Avg digit conf: 0.987
  Saved: output_yolo/test_1_result.png
```

### Process Custom Image

```bash
python inference_yolo.py \
    --yolo_model models_yolo/train/weights/best.pt \
    --cnn_model models_cnn \
    --image path/to/your/image.png
```

**Inference options:**
```bash
--conf_threshold 0.25     # YOLO confidence threshold
--device cuda             # Inference device
--output output_yolo      # Output directory
--no_display              # Don't display images
```

## Performance

**Expected results:**
- **YOLO Detection**: 90-95% mAP50, 5-15ms inference
- **CNN Recognition**: 85-90% sequence accuracy, 5-20ms inference
- **End-to-End**: 80-85% accuracy, 10-35ms total time

## Common Issues

### "ultralytics not found"

```bash
pip install ultralytics
```

### "Dataset file not found"

Run the data converter:
```bash
python convert_svhn_to_yolo.py
```

### "CNN model not found"

Train the CNN first:
```bash
python train_cnn.py --epochs 100
```

### Out of Memory

Reduce batch size or image size:
```bash
python train_yolo.py --batch_size 8 --imgsz 512
```

## What's Next?

1. **Tune hyperparameters**: Adjust learning rate, augmentation, etc.
2. **Try larger model**: Use `yolov8s.pt` or `yolov8m.pt` for better accuracy
3. **Longer training**: Train for 150-200 epochs for better convergence
4. **Custom images**: Test on your own house number images
5. **Deploy**: Integrate into your application

## Architecture Summary

```
Input Image (any size)
    ↓
[YOLOv8 Detection]
  - CSPDarknet53 backbone
  - PAN neck
  - Decoupled heads
  - Output: house number bbox
    ↓
Crop & Resize (224×224)
    ↓
[CNN Recognition]
  - Spatial Transformer Network
  - Inception-ResNet blocks
  - SE attention
  - Multi-task heads
  - Output: digit sequence
    ↓
Final Result: "2468"
```

## Key Benefits

✓ **Real-world ready**: Processes full images with multiple house numbers
✓ **Fast inference**: Single-stage detection + efficient recognition
✓ **Modular**: Detection and recognition are decoupled
✓ **Pretrained**: YOLOv8 starts with COCO weights
✓ **Industry-standard**: YOLOv8 is widely used in production

## Comparison

| Approach | Training Time | Inference | Best For |
|----------|--------------|-----------|----------|
| K-means + SVM | 35 min | 0.5-2s | Learning, interpretability |
| CNN | 2-4 hours | 5-20ms | Pre-cropped house numbers |
| CQT | 3-5 hours | 20-50ms | Research, bboxes needed |
| **YOLO + CNN** | **3-6 hours** | **10-35ms** | **Production, full images** |

---

**Ready to start?** Run the test script:

```bash
python test_yolo.py
```
