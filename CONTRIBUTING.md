# Code Structure and Contributing Guide

## Project Organization

```
my-repo/
├── models/                      # Neural network modules
│   ├── __init__.py             # Package initialization
│   ├── backbone.py             # ResNet50 + PSPNet feature extractor
│   ├── sfem.py                 # Sparse Feature Enhancement Module
│   ├── feature_matching.py    # Feature matching with cosine similarity
│   ├── hpkim.py                # Prior Knowledge Integration Module
│   └── fss_network.py          # Main FSS Network architecture
│
├── data/                        # Data loading and preprocessing
│   ├── __init__.py             # Package initialization
│   └── dataset.py              # SegPPD-101 dataset loader
│
├── utils/                       # Utility functions
│   ├── __init__.py             # Package initialization
│   └── utils.py                # Helper functions (metrics, I/O, etc.)
│
├── configs/                     # Configuration files
│   └── config.yaml             # Training and model configuration
│
├── train.py                     # Training script
├── inference.py                 # Inference/evaluation script
├── example.py                   # Example usage and testing
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
└── ARCHITECTURE.md             # Detailed architecture documentation
```

## Module Descriptions

### models/backbone.py
**Purpose**: Feature extraction using ResNet50 and PSPNet

**Key Classes**:
- `PSPModule`: Pyramid pooling for multi-scale features
- `ResNet50Backbone`: Main backbone network

**Design Notes**:
- ResNet50 pretrained on ImageNet
- Backbone parameters frozen by default
- PSPNet with 4 pyramid scales (1×1, 2×2, 3×3, 6×6)
- Returns multi-level features for flexible use

### models/sfem.py
**Purpose**: Enhance features by focusing on foreground

**Key Classes**:
- `AttentionCoordinationModule`: Channel + spatial attention
- `SFEM`: Main sparse enhancement module

**Design Notes**:
- ACM applies dual attention mechanism
- Learnable correction coefficient (initialized at 1.0)
- Background suppression via gating
- Separate instances for query and support

### models/feature_matching.py
**Purpose**: Match query and support features using similarity

**Key Classes**:
- `FeatureMatchingModule`: Cosine similarity-based matching

**Design Notes**:
- Masked average pooling for support prototype
- L2-normalized features for cosine similarity
- Temperature scaling for gradient smoothing
- CNN-based similarity refinement

### models/hpkim.py
**Purpose**: Integrate multi-level features with prior knowledge

**Key Classes**:
- `HPKIM`: Prior knowledge integration and fusion

**Design Notes**:
- Processes low-level (fine) and high-level (coarse) features
- Extracts prior knowledge from features + similarity
- Attention-based weighted fusion
- Upsampling for feature alignment

### models/fss_network.py
**Purpose**: Main network combining all modules

**Key Classes**:
- `FSSNetwork`: Complete few-shot segmentation network
- `build_fss_network`: Factory function

**Design Notes**:
- Sequential pipeline of all modules
- Separate SFEM for query and support
- Final segmentation head (256→128→64→2 channels)
- Binary output (foreground/background)

### data/dataset.py
**Purpose**: Load and process SegPPD-101 dataset

**Key Classes**:
- `SegPPD101Dataset`: Episodic dataset for meta-learning
- `build_dataloader`: DataLoader factory

**Design Notes**:
- Episodic sampling for few-shot learning
- Support and query pairs
- ImageNet normalization
- Handles missing data gracefully

### utils/utils.py
**Purpose**: Helper functions and utilities

**Key Functions**:
- `load_config`: YAML configuration loader
- `save_checkpoint`/`load_checkpoint`: Model I/O
- `compute_iou`/`compute_dice`: Evaluation metrics
- `get_optimizer`/`get_scheduler`: Optimizer setup
- `AverageMeter`: Running average tracker

## Key Design Patterns

### 1. Module Independence
Each module is self-contained and can be tested/modified independently.

```python
# Each module can be used standalone
from models.sfem import SFEM

sfem = SFEM(in_channels=512, out_channels=512)
enhanced, mask = sfem(features)
```

### 2. Configuration-Driven
All hyperparameters controlled via YAML config.

```python
# Easy to experiment with different settings
config = load_config('configs/config.yaml')
model = build_fss_network(config)
```

### 3. Factory Functions
Use factory functions for object creation.

```python
# Centralized creation logic
model = build_fss_network(config)
dataloader = build_dataloader(config, split='train')
optimizer = get_optimizer(model, config)
```

### 4. Meta-Learning Episodic Training
Dataset returns episodes (support + query pairs).

```python
# Each batch is an episode
batch = next(iter(dataloader))
query_img, query_mask = batch['query_img'], batch['query_mask']
support_img, support_mask = batch['support_img'], batch['support_mask']
```

## Adding New Features

### Add a New Module

1. Create module file in `models/`:
```python
# models/my_module.py
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your implementation
        
    def forward(self, x):
        # Your forward pass
        return output
```

2. Update `models/__init__.py`:
```python
from .my_module import MyModule
__all__ = [..., 'MyModule']
```

3. Integrate into `fss_network.py`:
```python
self.my_module = MyModule(config)
# Use in forward pass
```

### Add a New Metric

1. Add function to `utils/utils.py`:
```python
def compute_my_metric(pred, target):
    """Compute my custom metric"""
    # Implementation
    return metric_value
```

2. Use in training loop:
```python
metric = compute_my_metric(pred, target)
```

### Add Data Augmentation

1. Modify `data/dataset.py`:
```python
self.img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),  # Add augmentation
    transforms.ColorJitter(0.1, 0.1, 0.1),  # Add augmentation
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

### Add a New Loss Function

1. Add to `train.py`:
```python
class MyLoss(nn.Module):
    def forward(self, pred, target):
        # Implementation
        return loss

# Use in trainer
self.criterion = MyLoss()
```

## Testing Guidelines

### Unit Testing
Test individual modules:

```python
# Test SFEM
sfem = SFEM(512, 512)
x = torch.randn(2, 512, 16, 16)
out, mask = sfem(x)
assert out.shape == (2, 512, 16, 16)
```

### Integration Testing
Test complete pipeline:

```python
# Run example.py
python example.py
```

### Manual Testing
Verify on real data:

```python
# Train for few epochs
python train.py --config configs/config.yaml

# Check outputs
# - logs/ should contain TensorBoard logs
# - checkpoints/ should contain .pth files
```

## Code Style

### Python Conventions
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to classes and functions
- Type hints where helpful

### PyTorch Conventions
- Use `nn.Module` for all network components
- Implement `forward()` method
- Register submodules in `__init__`
- Use `nn.functional` for stateless operations

### Documentation
- Module-level docstring at top of file
- Class docstring explaining purpose
- Function docstring with Args/Returns
- Inline comments for complex logic

## Performance Optimization Tips

### Memory
- Use `torch.no_grad()` during inference
- Delete unused tensors: `del tensor`
- Use smaller batch sizes
- Enable gradient checkpointing for very deep models

### Speed
- Use GPU: `.cuda()` or `.to(device)`
- Increase `num_workers` in DataLoader
- Use mixed precision training (AMP)
- Profile code to find bottlenecks

### Accuracy
- Tune learning rate
- Add data augmentation
- Train for more epochs
- Ensemble multiple models

## Common Modifications

### Change Backbone
```python
# In models/backbone.py
from torchvision import models
resnet = models.resnet101(pretrained=True)  # Change to ResNet101
```

### Change Image Size
```yaml
# In configs/config.yaml
dataset:
  img_size: 384  # Change from 512
```

### Add Learning Rate Warmup
```python
# In utils/utils.py
def get_scheduler(optimizer, config):
    # Add warmup scheduler
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main = StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = SequentialLR(optimizer, [warmup, main], milestones=[5])
    return scheduler
```

### Add Model Checkpointing Every N Steps
```python
# In train.py, inside training loop
if global_step % save_every_n_steps == 0:
    save_checkpoint(model, optimizer, epoch, loss, 
                   f'checkpoint_step_{global_step}.pth')
```

## Debugging Tips

### Print Shapes
```python
print(f"Query shape: {query_img.shape}")
print(f"Features shape: {features.shape}")
```

### Visualize Activations
```python
import matplotlib.pyplot as plt
plt.imshow(features[0, 0].cpu().detach().numpy())
plt.colorbar()
plt.savefig('feature_map.png')
```

### Check Gradients
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

### Memory Profiling
```python
import torch.cuda as cuda
print(f"Memory allocated: {cuda.memory_allocated() / 1e9} GB")
print(f"Memory reserved: {cuda.memory_reserved() / 1e9} GB")
```

## Getting Help

1. **Check Documentation**: README.md, ARCHITECTURE.md, QUICKSTART.md
2. **Read Code Comments**: Inline documentation in source files
3. **Run Examples**: example.py demonstrates basic usage
4. **Check Issues**: Look for similar problems on GitHub
5. **Ask Questions**: Open an issue with details about your problem
