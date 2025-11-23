# FSS Network for Wild Plant Disease and Pest Segmentation

A Few-Shot Segmentation (FSS) Network designed for wild plant disease and pest clustering segmentation with reduced annotation cost.

## Overview

This repository implements an FSS Network that consists of four main modules:

1. **Feature Extraction**: ResNet50 backbone with PSPNet feature extractor
2. **SFEM** (Sparse Feature Enhancement Module): Integrates ACM (Attention Coordination Module) with learnable correction coefficients to focus on foreground and reduce irrelevant background information
3. **Feature Matching**: Uses cosine similarity to match high-level query and support features
4. **HPKIM** (High-level Prior Knowledge Integration Module): Provides prior knowledge features as additional cues for feature fusion, combining coarse-grained and fine-grained features

## Architecture

```
Input (Query & Support Images)
    ↓
ResNet50 + PSPNet (Feature Extraction)
    ↓
SFEM (Sparse Feature Enhancement)
    ↓
Feature Matching (Cosine Similarity)
    ↓
HPKIM (Prior Knowledge Integration)
    ↓
Segmentation Head
    ↓
Output (Segmentation Mask)
```

## Features

- **Meta-learning approach**: Episodic training for few-shot learning
- **ResNet50 backbone**: Pretrained on ImageNet with frozen parameters
- **PSPNet feature extractor**: Multi-scale feature extraction
- **Attention mechanisms**: ACM for foreground focus
- **Prior knowledge integration**: Combines multi-level features
- **AdamW optimizer**: With learning rate 0.0001
- **SegPPD-101 dataset support**: For plant disease and pest segmentation

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- Pillow >= 8.0.0
- tensorboard >= 2.7.0
- PyYAML >= 5.4.0
- tqdm >= 4.62.0
- scipy >= 1.7.0

## Dataset Structure

The SegPPD-101 dataset should be organized as follows:

```
data/SegPPD-101/
├── images/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
└── masks/
    ├── class1/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    ├── class2/
    │   └── ...
    └── ...
```

## Usage

### Training

```bash
python train.py --config configs/config.yaml
```

The training script will:
- Train for 150+ epochs
- Use AdamW optimizer with learning rate 0.0001
- Save checkpoints every 10 epochs
- Save the best model based on validation IoU
- Log metrics to TensorBoard

### Inference

```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --query path/to/query_image.jpg \
    --support path/to/support_image.jpg \
    --support_mask path/to/support_mask.png \
    --output output_prediction.png
```

### Monitoring Training

```bash
tensorboard --logdir logs/
```

## Configuration

Key configuration parameters in `configs/config.yaml`:

```yaml
model:
  backbone: resnet50
  pretrained: true
  freeze_backbone: true
  
training:
  optimizer: adamw
  learning_rate: 0.0001
  epochs: 150
  batch_size: 4
  
dataset:
  name: SegPPD-101
  img_size: 512
```

## Model Components

### 1. Feature Extraction (backbone.py)
- ResNet50 with PSPNet
- Multi-scale feature extraction
- Frozen backbone for transfer learning

### 2. SFEM (sfem.py)
- Attention Coordination Module (ACM)
- Learnable correction coefficients
- Background suppression

### 3. Feature Matching (feature_matching.py)
- Cosine similarity computation
- Support prototype extraction via masked pooling
- Similarity refinement

### 4. HPKIM (hpkim.py)
- Multi-level feature integration
- Prior knowledge extraction
- Attention-based weighted fusion

## Results

The model is trained on SegPPD-101 dataset for 150+ epochs with:
- Learning rate: 0.0001
- Optimizer: AdamW
- Evaluation metrics: IoU, Dice coefficient

## Citation

If you use this code in your research, please cite:

```
@article{fss_plant_disease,
  title={FSS Network for Wild Plant Disease and Pest Segmentation},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- ResNet50: He et al., "Deep Residual Learning for Image Recognition"
- PSPNet: Zhao et al., "Pyramid Scene Parsing Network"
- Few-shot learning techniques from meta-learning literature
