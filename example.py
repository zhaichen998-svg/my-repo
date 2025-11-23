"""
Example script showing how to use a trained FSS Network
"""

import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models import FSSNetwork


def create_example_data():
    """Create example data for demonstration"""
    print("Creating example data...")
    
    os.makedirs('example_data', exist_ok=True)
    
    # Create synthetic images
    np.random.seed(42)
    
    # Query image (plant with disease)
    query_img = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
    query_img[150:250, 150:250] = [50, 150, 50]  # Green plant area
    query_img[180:220, 180:220] = [200, 150, 100]  # Disease area
    Image.fromarray(query_img).save('example_data/query.jpg')
    
    # Support image (reference plant with disease)
    support_img = np.random.randint(80, 180, (400, 400, 3), dtype=np.uint8)
    support_img[100:300, 100:300] = [40, 140, 40]  # Green plant area
    support_img[150:250, 150:250] = [220, 170, 120]  # Disease area
    Image.fromarray(support_img).save('example_data/support.jpg')
    
    # Support mask (disease annotation)
    support_mask = np.zeros((400, 400), dtype=np.uint8)
    support_mask[150:250, 150:250] = 255  # Disease region
    Image.fromarray(support_mask).save('example_data/support_mask.png')
    
    print("Example data created in 'example_data/' directory")


def load_and_preprocess(img_path, mask_path=None, img_size=400):
    """Load and preprocess image and mask"""
    from torchvision import transforms
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    
    # Image transform
    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = img_transform(img).unsqueeze(0)
    
    # Load mask if provided
    if mask_path:
        mask = Image.open(mask_path).convert('L')
        mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        mask_tensor = mask_transform(mask).squeeze(0)
        mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0)
        return img_tensor, mask_tensor
    
    return img_tensor, None


def visualize_results(query_path, support_path, support_mask_path, pred_mask):
    """Visualize prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Load original images
    query_img = Image.open(query_path)
    support_img = Image.open(support_path)
    support_mask_img = Image.open(support_mask_path)
    
    # Plot query image
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title('Query Image (Plant to Segment)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot support image
    axes[0, 1].imshow(support_img)
    axes[0, 1].set_title('Support Image (Reference)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot support mask
    axes[1, 0].imshow(support_mask_img, cmap='gray')
    axes[1, 0].set_title('Support Mask (Disease Annotation)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot predicted mask
    pred_mask_np = pred_mask.cpu().numpy()
    axes[1, 1].imshow(pred_mask_np, cmap='hot')
    axes[1, 1].set_title('Predicted Disease Mask', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('example_data/prediction_result.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'example_data/prediction_result.png'")
    

def main():
    print("=" * 70)
    print("FSS Network Usage Example")
    print("=" * 70)
    print()
    
    # Create example data
    create_example_data()
    print()
    
    # Initialize model
    print("Initializing FSS Network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FSSNetwork(pretrained=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Note: In practice, you would load a trained checkpoint:
    # checkpoint = torch.load('checkpoints/best.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load and preprocess data
    print("Loading example data...")
    query_img, _ = load_and_preprocess('example_data/query.jpg')
    support_img, support_mask = load_and_preprocess(
        'example_data/support.jpg',
        'example_data/support_mask.png'
    )
    
    # Move to device
    query_img = query_img.to(device)
    support_img = support_img.to(device)
    support_mask = support_mask.to(device)
    
    print(f"Query image shape: {query_img.shape}")
    print(f"Support image shape: {support_img.shape}")
    print(f"Support mask shape: {support_mask.shape}")
    print()
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred = model(query_img, support_img, support_mask)
        pred_mask = torch.argmax(pred, dim=1)[0]
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Predicted mask shape: {pred_mask.shape}")
    
    # Calculate statistics
    foreground_pixels = (pred_mask == 1).sum().item()
    total_pixels = pred_mask.numel()
    foreground_ratio = foreground_pixels / total_pixels * 100
    
    print(f"\nPrediction Statistics:")
    print(f"  - Foreground pixels: {foreground_pixels:,}")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Disease coverage: {foreground_ratio:.2f}%")
    print()
    
    # Visualize results
    print("Creating visualization...")
    visualize_results(
        'example_data/query.jpg',
        'example_data/support.jpg',
        'example_data/support_mask.png',
        pred_mask
    )
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Train the model on real data: python train.py --data_root ./dataset/train")
    print("2. Use trained model for inference: python predict.py --checkpoint ./checkpoints/best.pth ...")
    print("3. See TRAINING_GUIDE.md for detailed instructions")


if __name__ == '__main__':
    main()
