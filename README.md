# FSS Network: Meta-learning Support Segmentation

A PyTorch implementation of FSS (Few-Shot Semantic Segmentation) Network for wild plant disease segmentation on the SegPPD-101 dataset. This implementation features meta-learning strategies to reduce annotation costs in semantic segmentation tasks.

## Architecture Overview

The FSS Network consists of four main components:

### 1. Feature Extraction Module
- **Shared ResNet-50 Backbone**: Pre-trained on ImageNet for robust feature representation
- **PSPNet Feature Extractor**: Pyramid Pooling Module for multi-scale feature extraction
- Extracts low-level, mid-level, and high-level features from both query and support images

### 2. SFEM (Support Feature Enhancement Module)
- **ACM (Attention Calibration Module)**: Includes learnable correction coefficients (α, β)
- Generates enhanced versions of query and support features
- Filters background noise from features using spatial and channel attention
- Applies masked feature enhancement for support images

### 3. Feature Matching Module
- Computes cosine similarity between query and support features
- Transfers knowledge from support set to query image
- Uses masked average pooling to generate support prototypes
- Integrates enhanced features through similarity-based weighting

### 4. HPKIM (Hierarchical Prior Knowledge Integration Module)
- Generates hierarchical query and support features at multiple scales
- Combines coarse-grained and fine-grained information
- Uses feature pooling (4×4 and 8×8) for multi-scale representation
- Backend convolutional operations for feature fusion
- Produces final segmentation predictions with high accuracy

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- numpy >= 1.21.0
- Pillow >= 8.3.0
- opencv-python >= 4.5.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0
- tensorboard >= 2.7.0

## Dataset Structure

Organize your SegPPD-101 dataset as follows:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── masks/
│       ├── img1.png
│       ├── img2.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

## Training

### Basic Training

Train the FSS Network with default parameters (150 epochs, AdamW optimizer, lr=0.0001):

```bash
python train.py --data_root ./dataset/train --epochs 150
```

### Advanced Training Options

```bash
python train.py \
    --data_root ./dataset/train \
    --epochs 150 \
    --batch_size 4 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --img_size 400 \
    --k_shot 1 \
    --iters_per_epoch 500 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs
```

### Training Parameters

- `--epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size for meta-learning episodes (default: 4)
- `--lr`: Initial learning rate (default: 0.0001)
- `--weight_decay`: Weight decay for AdamW optimizer (default: 0.0001)
- `--k_shot`: Number of support samples (default: 1)
- `--iters_per_epoch`: Iterations per epoch (default: 500)
- `--lr_step`: Learning rate decay step size (default: 50)
- `--lr_gamma`: Learning rate decay factor (default: 0.5)

### Monitor Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

## Inference

### Basic Prediction

Run prediction on a query image given a support example:

```bash
python predict.py \
    --checkpoint ./checkpoints/best.pth \
    --query_img ./test_images/query.jpg \
    --support_img ./test_images/support.jpg \
    --support_mask ./test_images/support_mask.png \
    --output_dir ./outputs \
    --save_vis
```

### With Ground Truth Comparison

```bash
python predict.py \
    --checkpoint ./checkpoints/best.pth \
    --query_img ./test_images/query.jpg \
    --support_img ./test_images/support.jpg \
    --support_mask ./test_images/support_mask.png \
    --gt_mask ./test_images/query_mask.png \
    --output_dir ./outputs \
    --save_vis \
    --show_vis
```

### Inference Parameters

- `--checkpoint`: Path to trained model checkpoint
- `--query_img`: Path to query image for segmentation
- `--support_img`: Path to support image (reference)
- `--support_mask`: Path to support mask (reference)
- `--gt_mask`: Path to ground truth mask (optional, for comparison)
- `--output_dir`: Directory to save predictions
- `--save_vis`: Save visualization images
- `--show_vis`: Display visualization in a window

## Model Architecture Details

### Feature Extraction
- **Input**: RGB images (3×H×W)
- **Backbone**: ResNet-50 (layers 1-4)
- **Low-level features**: 256 channels
- **Mid-level features**: 1024 channels  
- **High-level features**: 2048 channels → PSPNet → 512 channels

### SFEM Module
- **Channel Attention**: Adaptive average pooling + 1×1 convolutions
- **Spatial Attention**: 1×1 convolution + sigmoid activation
- **Learnable Coefficients**: α (scaling), β (bias)
- **Background Filtering**: 3×3 convolutions with ReLU

### Feature Matching
- **Similarity Metric**: Cosine similarity
- **Support Prototype**: Masked average pooling
- **Knowledge Transfer**: Element-wise multiplication with similarity scores

### HPKIM Module
- **Coarse-grained**: 4×4 adaptive pooling
- **Fine-grained**: 8×8 adaptive pooling
- **Fusion**: 3×3 convolutions with batch normalization and ReLU
- **Output**: 256-channel feature maps

### Segmentation Head
- 256 → 128 → 64 → 2 channels (background/foreground)
- 3×3 convolutions with batch normalization and ReLU

## Key Features

✅ **Meta-learning Strategy**: Few-shot learning approach reduces annotation requirements

✅ **Hierarchical Feature Integration**: Multi-scale features for improved segmentation

✅ **Attention Mechanisms**: Learnable ACM with correction coefficients

✅ **End-to-End Training**: Complete pipeline from feature extraction to prediction

✅ **Visualization Tools**: Built-in prediction visualization and overlay generation

✅ **TensorBoard Support**: Real-time training monitoring

✅ **Flexible Architecture**: Modular design for easy customization

## Performance Metrics

The model is evaluated using:
- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels

## File Structure

```
my-repo/
├── models/
│   ├── __init__.py
│   ├── fss_network.py      # Main FSS Network
│   └── modules.py           # Feature extraction, SFEM, matching, HPKIM
├── data/
│   ├── __init__.py
│   └── dataset.py           # Dataset and meta-learning dataloader
├── utils/
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics
│   └── visualization.py     # Visualization tools
├── train.py                 # Training script
├── predict.py               # Inference script
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fss_network,
  title={FSS Network: Meta-learning Support Segmentation for Few-Shot Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- ResNet-50 backbone from torchvision
- PSPNet architecture inspiration
- Meta-learning strategy for few-shot segmentation
- SegPPD-101 dataset for wild plant disease segmentation
