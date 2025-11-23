"""
Demo script for FSS Network
Creates synthetic data and demonstrates the complete pipeline
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import FSSNetwork


def create_synthetic_data(img_size=400):
    """Create synthetic images and masks for demonstration"""
    # Create query image (random RGB image)
    query_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Create support image
    support_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Create support mask (circular region)
    y, x = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2
    radius = img_size // 4
    mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
    support_mask = mask.astype(np.float32)
    
    return query_img, support_img, support_mask


def prepare_tensors(query_img, support_img, support_mask):
    """Convert numpy arrays to PyTorch tensors"""
    # Normalize images
    query_tensor = torch.from_numpy(query_img).float().permute(2, 0, 1) / 255.0
    support_tensor = torch.from_numpy(support_img).float().permute(2, 0, 1) / 255.0
    mask_tensor = torch.from_numpy(support_mask).float()
    
    # Add batch dimension
    query_tensor = query_tensor.unsqueeze(0)
    support_tensor = support_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0)
    
    return query_tensor, support_tensor, mask_tensor


def visualize_demo_results(query_img, support_img, support_mask, pred_mask):
    """Visualize demo results"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')
    
    axes[1].imshow(support_img)
    axes[1].set_title('Support Image')
    axes[1].axis('off')
    
    axes[2].imshow(support_mask, cmap='gray')
    axes[2].set_title('Support Mask')
    axes[2].axis('off')
    
    axes[3].imshow(pred_mask, cmap='gray')
    axes[3].set_title('Predicted Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
    print("Demo visualization saved to demo_output.png")
    plt.show()


def main():
    print("=" * 50)
    print("FSS Network Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nInitializing FSS Network...")
    model = FSSNetwork(pretrained=False)  # Set to False for quick demo
    model = model.to(device)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    img_size = 400
    query_img, support_img, support_mask = create_synthetic_data(img_size)
    
    # Prepare tensors
    print("Preparing tensors...")
    query_tensor, support_tensor, mask_tensor = prepare_tensors(
        query_img, support_img, support_mask
    )
    
    # Move to device
    query_tensor = query_tensor.to(device)
    support_tensor = support_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    print(f"Query shape: {query_tensor.shape}")
    print(f"Support shape: {support_tensor.shape}")
    print(f"Mask shape: {mask_tensor.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        pred = model(query_tensor, support_tensor, mask_tensor)
        pred_mask = torch.argmax(pred, dim=1)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Predicted mask shape: {pred_mask.shape}")
    
    # Convert to numpy for visualization
    pred_mask_np = pred_mask[0].cpu().numpy()
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_demo_results(query_img, support_img, support_mask, pred_mask_np)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)
    
    # Print module information
    print("\nModel Architecture:")
    print("-" * 50)
    print("1. Feature Extraction Module: ResNet-50 + PSPNet")
    print("2. SFEM Module: Attention Calibration with learnable coefficients")
    print("3. Feature Matching Module: Cosine similarity-based matching")
    print("4. HPKIM Module: Hierarchical feature integration")
    print("5. Segmentation Head: 256 → 128 → 64 → 2 channels")
    print("-" * 50)


if __name__ == '__main__':
    main()
