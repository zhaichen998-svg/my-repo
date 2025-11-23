# FSS Network Training Guide

This guide provides detailed instructions for training the FSS Network on the SegPPD-101 dataset.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the following structure:
```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/
│       ├── img001.png
│       ├── img002.png
│       └── ...
└── val/
    ├── images/
    └── masks/
```

## Quick Start

### 1. Basic Training (150 epochs)

```bash
python train.py \
    --data_root ./dataset/train \
    --epochs 150 \
    --batch_size 4 \
    --lr 0.0001
```

### 2. Training with Custom Parameters

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
    --lr_step 50 \
    --lr_gamma 0.5 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --num_workers 4 \
    --val_freq 5
```

## Training Parameters

### Data Parameters
- `--data_root`: Path to training dataset (default: `./dataset/train`)
- `--img_size`: Input image size (default: 400)

### Training Parameters
- `--epochs`: Number of training epochs (default: 150)
- `--batch_size`: Batch size for meta-learning episodes (default: 4)
- `--iters_per_epoch`: Number of iterations per epoch (default: 500)
- `--k_shot`: Number of support samples per episode (default: 1)

### Optimizer Parameters
- `--lr`: Initial learning rate for AdamW optimizer (default: 0.0001)
- `--weight_decay`: Weight decay coefficient (default: 0.0001)
- `--lr_step`: Learning rate decay step size in epochs (default: 50)
- `--lr_gamma`: Learning rate decay factor (default: 0.5)

### System Parameters
- `--num_workers`: Number of data loading workers (default: 4)
- `--val_freq`: Validation frequency in epochs (default: 5)
- `--checkpoint_dir`: Directory for saving checkpoints (default: `./checkpoints`)
- `--log_dir`: Directory for TensorBoard logs (default: `./logs`)

## Meta-Learning Strategy

The FSS Network uses a meta-learning approach for few-shot segmentation:

1. **Episode Generation**: Each training iteration creates an episode with:
   - Support set: 1 image with mask (k-shot = 1)
   - Query set: 1 image with mask
   
2. **Training Procedure**:
   - Extract features from support and query images
   - Enhance features using SFEM with learnable ACM
   - Match features using cosine similarity
   - Integrate hierarchical features using HPKIM
   - Predict segmentation mask for query image
   - Compute loss against ground truth

3. **Optimization**:
   - Optimizer: AdamW
   - Initial learning rate: 0.0001
   - Learning rate decay: Step decay (factor 0.5 every 50 epochs)
   - Loss function: Cross-entropy loss

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./logs
```

Metrics tracked:
- Training loss
- Training IoU (Intersection over Union)
- Training Dice coefficient
- Validation loss (every 5 epochs)
- Validation IoU
- Validation Dice coefficient
- Learning rate

### Checkpoints

Two checkpoint files are saved:
- `checkpoints/latest.pth`: Latest model checkpoint
- `checkpoints/best.pth`: Best model based on validation IoU

## Training Tips

1. **GPU Acceleration**: Use a GPU for faster training
   ```bash
   # Check GPU availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

2. **Batch Size**: Adjust based on your GPU memory
   - 4GB GPU: batch_size=2
   - 8GB GPU: batch_size=4
   - 12GB+ GPU: batch_size=8

3. **Learning Rate**: If training is unstable, reduce the learning rate
   ```bash
   python train.py --lr 0.00005
   ```

4. **Data Augmentation**: For better generalization, consider adding data augmentation in `data/dataset.py`

5. **Resume Training**: Load a checkpoint and continue training
   ```python
   # Add this code to train.py to resume from checkpoint
   checkpoint = torch.load('checkpoints/latest.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch'] + 1
   ```

## Expected Training Time

On different hardware configurations:

- **CPU (Intel i7)**: ~50-60 hours for 150 epochs
- **GPU (NVIDIA RTX 2080)**: ~6-8 hours for 150 epochs
- **GPU (NVIDIA V100)**: ~3-4 hours for 150 epochs

## Model Performance

Expected performance on SegPPD-101:

| Metric | Expected Value |
|--------|---------------|
| IoU    | > 0.70        |
| Dice   | > 0.80        |
| Pixel Accuracy | > 0.90 |

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 2

# Reduce image size
python train.py --img_size 256
```

### Training Loss Not Decreasing
- Check learning rate (try reducing)
- Verify dataset quality
- Ensure proper data normalization
- Check for data leakage

### Poor Validation Performance
- Increase training epochs
- Add data augmentation
- Reduce learning rate
- Check train/val data distribution

## Next Steps

After training completes:

1. **Evaluate the model**:
   ```bash
   python predict.py \
       --checkpoint ./checkpoints/best.pth \
       --query_img ./test_images/query.jpg \
       --support_img ./test_images/support.jpg \
       --support_mask ./test_images/support_mask.png \
       --save_vis
   ```

2. **Fine-tune on specific disease types**: Use the pre-trained model and fine-tune on specific plant diseases

3. **Export for deployment**: Convert the model for production use

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{fss_network_2024,
  title={FSS Network: Meta-learning Support Segmentation for Plant Disease Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```
