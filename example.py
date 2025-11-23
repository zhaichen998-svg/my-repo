"""
Example script demonstrating FSS Network usage
"""

import torch
from models import build_fss_network
from utils.utils import load_config


def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Build model
    print("Building FSS Network...")
    model = build_fss_network(config)
    
    # Print model architecture
    print("\n" + "="*80)
    print("FSS Network Architecture")
    print("="*80)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("="*80)
    
    # Test forward pass with dummy data
    print("\nTesting forward pass with dummy data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    img_size = config['dataset']['img_size']
    query_img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    support_img = torch.randn(batch_size, 3, img_size, img_size).to(device)
    support_mask = torch.randn(batch_size, 1, img_size, img_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(query_img, support_img, support_mask)
    
    print(f"Input shape: {query_img.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, 2, {img_size}, {img_size}]")
    
    assert output.shape == (batch_size, 2, img_size, img_size), "Output shape mismatch!"
    
    print("\n✓ Forward pass successful!")
    print("\nModel is ready for training!")
    
    # Display component information
    print("\n" + "="*80)
    print("FSS Network Components:")
    print("="*80)
    print("1. Feature Extraction: ResNet50 + PSPNet")
    print("   - Backbone: ResNet50 (pretrained, frozen)")
    print("   - Feature Extractor: PSPNet with pyramid pooling")
    print()
    print("2. SFEM (Sparse Feature Enhancement Module)")
    print("   - ACM (Attention Coordination Module)")
    print("   - Learnable correction coefficients")
    print("   - Background suppression")
    print()
    print("3. Feature Matching Module")
    print("   - Cosine similarity computation")
    print("   - Support prototype via masked pooling")
    print("   - Similarity refinement")
    print()
    print("4. HPKIM (High-level Prior Knowledge Integration)")
    print("   - Multi-level feature fusion")
    print("   - Prior knowledge extraction")
    print("   - Attention-based weighted combination")
    print()
    print("5. Segmentation Head")
    print("   - Final classification layer")
    print("   - Output: 2 classes (foreground/background)")
    print("="*80)


if __name__ == '__main__':
    main()
