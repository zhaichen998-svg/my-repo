import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def denormalize_image(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        img_tensor: Normalized image tensor [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        img_array: Denormalized image array [H, W, C]
    """
    img = img_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    img = torch.clamp(img, 0, 1)
    img_array = img.permute(1, 2, 0).cpu().numpy()
    return img_array


def visualize_prediction(query_img, support_img, support_mask, pred_mask, gt_mask=None, 
                        save_path=None, show=True):
    """
    Visualize prediction results
    
    Args:
        query_img: Query image tensor [3, H, W]
        support_img: Support image tensor [3, H, W]
        support_mask: Support mask tensor [H, W]
        pred_mask: Predicted mask tensor [H, W]
        gt_mask: Ground truth mask tensor [H, W] (optional)
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    # Denormalize images
    query_img_np = denormalize_image(query_img)
    support_img_np = denormalize_image(support_img)
    
    # Convert masks to numpy
    support_mask_np = support_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()
    
    # Create figure
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        gt_mask_np = gt_mask.cpu().numpy()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot query image
    axes[0, 0].imshow(query_img_np)
    axes[0, 0].set_title('Query Image')
    axes[0, 0].axis('off')
    
    # Plot support image
    axes[0, 1].imshow(support_img_np)
    axes[0, 1].set_title('Support Image')
    axes[0, 1].axis('off')
    
    # Plot support mask
    if gt_mask is not None:
        axes[0, 2].imshow(support_mask_np, cmap='gray')
        axes[0, 2].set_title('Support Mask')
        axes[0, 2].axis('off')
    else:
        axes[1, 0].imshow(support_mask_np, cmap='gray')
        axes[1, 0].set_title('Support Mask')
        axes[1, 0].axis('off')
    
    # Plot prediction
    if gt_mask is not None:
        axes[1, 0].imshow(pred_mask_np, cmap='gray')
        axes[1, 0].set_title('Predicted Mask')
        axes[1, 0].axis('off')
        
        # Plot ground truth
        axes[1, 1].imshow(gt_mask_np, cmap='gray')
        axes[1, 1].set_title('Ground Truth Mask')
        axes[1, 1].axis('off')
        
        # Plot overlay
        overlay = query_img_np.copy()
        overlay[pred_mask_np > 0.5] = [1, 0, 0]  # Red for prediction
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Prediction Overlay')
        axes[1, 2].axis('off')
    else:
        # Plot overlay
        overlay = query_img_np.copy()
        overlay[pred_mask_np > 0.5] = [1, 0, 0]  # Red for prediction
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Prediction Overlay')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_prediction(pred_mask, save_path):
    """
    Save prediction mask as an image
    
    Args:
        pred_mask: Predicted mask tensor [H, W]
        save_path: Path to save the mask
    """
    pred_mask_np = pred_mask.cpu().numpy()
    pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
    
    img = Image.fromarray(pred_mask_np)
    img.save(save_path)
    print(f"Prediction mask saved to {save_path}")


def create_overlay(image, mask, alpha=0.5, color=[255, 0, 0]):
    """
    Create an overlay of mask on image
    
    Args:
        image: Image array [H, W, C]
        mask: Mask array [H, W]
        alpha: Transparency factor
        color: Color for the mask overlay [R, G, B]
    
    Returns:
        overlay: Image with mask overlay
    """
    overlay = image.copy()
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask > 0.5] = color
    
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
    return overlay
