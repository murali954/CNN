# ResNet-YOLO Object Detection

YOLO-style object detector using ResNet-50 backbone for real-time detection on COCO dataset.

## Features

- ResNet-50 backbone with YOLO detection head
- Multi-scale detection and comprehensive evaluation metrics
- COCO dataset support with GPU acceleration
- Custom training pipeline with NMS post-processing

## Project Structure

```
resnet-yolo/
├── models/
│   ├── resnet_yolo.py          # Model architecture
│   ├── utils.py                # NMS, loss functions
│   └── model_weights.pth       # Trained weights
├── utils/
│   ├── coco_dataset.py         # Dataset loader
│   └── transforms.py           # Data augmentation
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Inference script
├── dataset/
│   ├── train/images/           # Training images
│   ├── train/labels/           # Training labels
│   ├── valid/images/           # Validation images
│   └── valid/labels/           # Validation labels
└── requirements.txt
```

## Installation

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Install PyTorch:**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Dataset Preparation

**YOLO Label Format:**
```
class_id center_x center_y width height
```
All coordinates normalized to [0, 1].

## Usage

**Training:**
```bash
python scripts/train.py
python scripts/train.py --epochs 100 --batch_size 16 --learning_rate 0.001
```

**Evaluation:**
```bash
python scripts/evaluate.py
python scripts/evaluate.py --weights models/custom_weights.pth
```

**Inference:**
```bash
python scripts/inference.py --image path/to/image.jpg
python scripts/inference.py --input_dir images/ --output_dir results/
```

## Model Architecture

- **Backbone**: ResNet-50 (pre-trained on ImageNet)
- **Head**: YOLO detection with bbox regression + classification
- **Input**: 640×640×3
- **Output**: [batch_size, num_anchors, 85] (4 bbox + 1 obj + 80 classes)
- **Parameters**: ~25.6M

## Performance Metrics

- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **Precision/Recall**: Detection accuracy metrics
- **F1-Score**: Harmonic mean of precision and recall
- **FPS**: Inference speed

Example results saved to `metrics.json`:
```json
{
  "performance_metrics": {
    "mean_precision": 0.7234,
    "mean_recall": 0.6891,
    "f1_score": 0.7058,
    "mAP": 0.6745
  },
  "inference_metrics": {
    "fps": 42.73
  }
}
```

