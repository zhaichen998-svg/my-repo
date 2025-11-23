# FSS Network Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FSS Network                              │
│                                                                   │
│  Input: Query Image (Iq) + Support Image (Is) + Support Mask    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Extraction Module                       │
│                   (ResNet50 + PSPNet)                            │
│                                                                   │
│  • Backbone: ResNet50 (pretrained on ImageNet, frozen)          │
│  • Multi-level features: layer1, layer2, layer3, layer4         │
│  • PSPNet: Pyramid Pooling Module for multi-scale context       │
│                                                                   │
│  Output: Multi-level features at different scales               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              SFEM (Sparse Feature Enhancement Module)            │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ACM (Attention Coordination Module)                      │   │
│  │  • Channel Attention: Focus on important feature channels │   │
│  │  • Spatial Attention: Focus on important spatial regions  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Learnable Correction Coefficients                        │   │
│  │  • Adaptive feature scaling                               │   │
│  │  • Initialized at 1.0, learned during training            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Background Suppression                                   │   │
│  │  • Foreground mask generation                             │   │
│  │  • Reduces irrelevant background information              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Output: Enhanced query & support features                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Feature Matching Module                          │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Support Prototype Extraction                             │   │
│  │  • Masked average pooling on support features            │   │
│  │  • Uses support mask to focus on foreground              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Cosine Similarity Computation                            │   │
│  │  • Measure similarity between query and support          │   │
│  │  • Normalized features for robust matching               │   │
│  │  • Temperature scaling for similarity refinement          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Similarity Refinement                                    │   │
│  │  • CNN-based refinement network                           │   │
│  │  • Smooth and enhance similarity map                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Output: Refined similarity map                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│      HPKIM (High-level Prior Knowledge Integration Module)       │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Multi-level Feature Processing                           │   │
│  │  • Low-level features (layer1): Fine-grained details     │   │
│  │  • High-level features (layer3): Coarse semantic info    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Prior Knowledge Extraction                               │   │
│  │  • Combines high-level features with similarity map       │   │
│  │  • Generates prior knowledge cues                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                      │
│                            ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Attention-based Feature Fusion                           │   │
│  │  • Compute attention weights for each feature level       │   │
│  │  • Weighted combination of all features                   │   │
│  │  • Adaptive fusion based on content                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  Output: Fused features with prior knowledge                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Segmentation Head                              │
│                                                                   │
│  • Progressive channel reduction: 256 → 128 → 64 → 2            │
│  • Final prediction: 2 classes (foreground/background)           │
│                                                                   │
│  Output: Segmentation mask [H x W x 2]                          │
└─────────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Feature Extraction (backbone.py)
**Purpose**: Extract multi-scale features from images

**Components**:
- ResNet50: Pretrained convolutional backbone
  - layer1: 256 channels, 1/4 resolution
  - layer2: 512 channels, 1/8 resolution
  - layer3: 1024 channels, 1/16 resolution
  - layer4: 2048 channels, 1/32 resolution
  
- PSPNet Module: Pyramid pooling with 4 scales (1×1, 2×2, 3×3, 6×6)
  - Captures multi-scale context information
  - Output: 512 channels

**Key Features**:
- Backbone parameters frozen (transfer learning)
- PSPNet parameters trainable
- Multi-level feature extraction

---

### 2. SFEM - Sparse Feature Enhancement Module (sfem.py)
**Purpose**: Focus on foreground and suppress background

**ACM (Attention Coordination Module)**:
- Channel Attention: Identifies important feature channels
  - Global average pooling → FC layers → Sigmoid
  - Reduction ratio: 16
  
- Spatial Attention: Identifies important spatial locations
  - Max + Average pooling → Conv7×7 → Sigmoid
  - Highlights foreground regions

**Learnable Correction**:
- Parameter initialized at 1.0
- Adaptively learned during training
- Balances feature magnitudes

**Background Suppression**:
- Generates foreground mask
- Multiplicative gating mechanism
- Reduces irrelevant information

---

### 3. Feature Matching Module (feature_matching.py)
**Purpose**: Match query and support features

**Masked Average Pooling**:
- Extract support prototype using mask
- Focus only on foreground pixels
- Robust representation of target class

**Cosine Similarity**:
- Normalized dot product
- Range: [-1, 1], typically [0, 1] for similar features
- Temperature scaling for smooth gradients

**Similarity Refinement**:
- CNN-based post-processing
- Smooths noisy similarity maps
- 64 intermediate channels
- Sigmoid activation for [0, 1] range

---

### 4. HPKIM - High-level Prior Knowledge Integration (hpkim.py)
**Purpose**: Combine multi-level features with prior knowledge

**Three Feature Streams**:
1. Low-level (layer1): Fine-grained details, edges, textures
2. High-level (layer3): Semantic information, object parts
3. Prior knowledge: Derived from high-level + similarity map

**Prior Knowledge Extraction**:
- Concatenate high-level features and similarity map
- 2-layer CNN for knowledge extraction
- Captures relationships between features

**Attention-based Fusion**:
- Compute 3-way attention weights (softmax)
- Weighted sum of three feature streams
- Adaptive based on input content

---

## Training Configuration

### Meta-Learning Setup
- **Approach**: Episodic training
- **Episode Structure**:
  - Support set: 1 image + mask (1-shot)
  - Query set: 1 image + mask
  - Binary segmentation: foreground vs background

### Optimization
- **Optimizer**: AdamW
  - Learning rate: 0.0001
  - Weight decay: 0.0001
  
- **Learning Rate Scheduler**: StepLR
  - Decay step: 50 epochs
  - Decay factor: 0.5

### Training Schedule
- **Total epochs**: 150+
- **Batch size**: 4
- **Episodes per epoch**: 1000
- **Loss function**: Cross-Entropy Loss

### Data Augmentation
- Resize to 512×512
- ImageNet normalization
- Random sampling for episodes

---

## Model Statistics

```
Total Parameters:      47,575,246
Trainable Parameters:  24,067,214 (50.6%)
Frozen Parameters:     23,508,032 (49.4%)

Memory Requirements (approximate):
- Model: ~190 MB (float32)
- Activations: ~400 MB per sample (512×512)
- Total GPU memory: ~2-3 GB (batch_size=4)
```

---

## Key Design Decisions

1. **Frozen Backbone**: Transfer learning from ImageNet for efficiency
2. **PSPNet**: Multi-scale context crucial for segmentation
3. **ACM in SFEM**: Dual attention for better foreground focus
4. **Cosine Similarity**: Robust metric for few-shot matching
5. **HPKIM**: Combines complementary information from multiple levels
6. **Meta-Learning**: Natural fit for few-shot scenario

---

## Inference Flow

```python
# 1. Load model and checkpoint
model = build_fss_network(config)
model.load_state_dict(checkpoint)

# 2. Prepare inputs
query_img = preprocess(query_image)      # [1, 3, 512, 512]
support_img = preprocess(support_image)   # [1, 3, 512, 512]
support_mask = preprocess(mask)           # [1, 1, 512, 512]

# 3. Forward pass
output = model(query_img, support_img, support_mask)  # [1, 2, 512, 512]

# 4. Get prediction
pred_mask = output.argmax(dim=1)         # [1, 512, 512]
```

---

## Performance Metrics

- **IoU (Intersection over Union)**: Primary metric
- **Dice Coefficient**: Alternative metric
- **Cross-Entropy Loss**: Training objective

Computed for both foreground and background classes.
