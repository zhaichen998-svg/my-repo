# Quick Start Guide

This guide will help you get started with the FSS Network for few-shot semantic segmentation.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/zhaichen998-svg/my-repo.git
cd my-repo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Verify Installation

Run the example script to verify everything is set up correctly:

```bash
python example.py
```

Expected output:
```
Building FSS Network...
[Model architecture printed]
Total parameters: 47,575,246
Trainable parameters: 24,067,214
✓ Forward pass successful!
Model is ready for training!
```

## Dataset Preparation

### SegPPD-101 Dataset Structure

Organize your dataset as follows:

```
data/SegPPD-101/
├── images/
│   ├── disease_class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── disease_class2/
│   │   └── ...
│   └── ...
└── masks/
    ├── disease_class1/
    │   ├── img001.png  (binary mask: 0=background, 255=foreground)
    │   ├── img002.png
    │   └── ...
    ├── disease_class2/
    │   └── ...
    └── ...
```

### Dataset Guidelines
- Images: RGB format (.jpg, .jpeg, .png)
- Masks: Binary format (.png)
  - Background: 0 (black)
  - Foreground: 255 (white)
- Same filename for image and corresponding mask

## Training

### Basic Training

```bash
python train.py --config configs/config.yaml
```

### Custom Configuration

Edit `configs/config.yaml` to customize training:

```yaml
training:
  epochs: 150              # Number of training epochs
  learning_rate: 0.0001    # Learning rate
  batch_size: 4            # Batch size
  episodes_per_epoch: 1000 # Episodes per epoch
  
dataset:
  root: ./data/SegPPD-101  # Dataset path
  img_size: 512            # Image size
```

### Monitor Training

Start TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Training Outputs

- **Logs**: `logs/` directory (TensorBoard logs)
- **Checkpoints**: `checkpoints/` directory
  - `checkpoint_epoch_N.pth`: Saved every 10 epochs
  - `best_model.pth`: Best model based on validation IoU

## Inference

### Single Image Prediction

```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --query path/to/query_image.jpg \
    --support path/to/support_image.jpg \
    --support_mask path/to/support_mask.png \
    --output prediction.png
```

### Parameters:
- `--query`: Image you want to segment
- `--support`: Example image of the target class
- `--support_mask`: Mask for the support image
- `--output`: Output path for prediction

### Example Workflow

1. **Choose a support example**: Select an image with the disease/pest you want to detect
2. **Create support mask**: Manually annotate the support image (or use existing mask)
3. **Run inference**: Use the support pair to segment query images
4. **Visualize results**: Check the output prediction

## Python API Usage

### Training in Code

```python
from models import build_fss_network
from data.dataset import build_dataloader
from utils.utils import load_config, get_optimizer

# Load config
config = load_config('configs/config.yaml')

# Build model
model = build_fss_network(config)
model = model.cuda()

# Build dataloader
train_loader = build_dataloader(config, split='train')

# Setup optimizer
optimizer = get_optimizer(model, config)

# Training loop
for epoch in range(config['training']['epochs']):
    for batch in train_loader:
        query_img = batch['query_img'].cuda()
        query_mask = batch['query_mask'].cuda()
        support_img = batch['support_img'].cuda()
        support_mask = batch['support_mask'].cuda()
        
        # Forward pass
        output = model(query_img, support_img, support_mask)
        
        # Compute loss and backprop
        loss = criterion(output, query_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference in Code

```python
from models import build_fss_network
from utils.utils import load_config, load_checkpoint
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
config = load_config('configs/config.yaml')
model = build_fss_network(config)
load_checkpoint(model, None, 'checkpoints/best_model.pth')
model.eval()
model.cuda()

# Prepare transforms
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load images
query_img = transform(Image.open('query.jpg')).unsqueeze(0).cuda()
support_img = transform(Image.open('support.jpg')).unsqueeze(0).cuda()
support_mask = T.ToTensor()(Image.open('support_mask.png')).unsqueeze(0).cuda()

# Inference
with torch.no_grad():
    output = model(query_img, support_img, support_mask)
    pred = output.argmax(dim=1)

# Save result
pred_img = (pred.squeeze().cpu().numpy() * 255).astype('uint8')
Image.fromarray(pred_img).save('output.png')
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size in config: `batch_size: 2` or `batch_size: 1`
- Reduce image size: `img_size: 384` or `img_size: 256`
- Use gradient accumulation (modify train.py)

### Dataset Not Found
- Check dataset path in config: `dataset.root`
- Verify directory structure matches expected format
- Check file permissions

### Slow Training
- Use GPU: Install CUDA-enabled PyTorch
- Increase num_workers: `num_workers: 8`
- Use mixed precision training (add to train.py)

### Poor Performance
- Increase training epochs: `epochs: 200`
- Adjust learning rate: Try `0.0005` or `0.00005`
- Verify data quality: Check masks are correct
- Add data augmentation (modify dataset.py)

## Tips for Best Results

1. **Support Selection**: Choose clear, representative support examples
2. **Mask Quality**: Ensure support masks are accurate
3. **Class Consistency**: Support and query should show same disease/pest
4. **Multiple Supports**: Can average predictions from multiple support examples
5. **Fine-tuning**: For specific dataset, consider fine-tuning on that data

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model explanation
- Check [README.md](README.md) for complete documentation
- Modify config for your specific use case
- Experiment with different hyperparameters
- Add custom data augmentation if needed

## Support

For issues or questions:
- Check existing issues on GitHub
- Review documentation files
- Check code comments for implementation details
