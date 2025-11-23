import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms

from models import FSSNetwork
from utils import visualize_prediction, save_prediction


def load_image(image_path, img_size=400):
    """Load and preprocess an image"""
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def load_mask(mask_path, img_size=400):
    """Load and preprocess a mask"""
    mask = Image.open(mask_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), 
                        interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    mask_tensor = transform(mask).squeeze(0)
    mask_tensor = (mask_tensor > 0.5).float()
    mask_tensor = mask_tensor.unsqueeze(0)
    
    return mask_tensor


def predict(model, query_img_path, support_img_path, support_mask_path, 
           device, img_size=400):
    """
    Run prediction on a query image given a support example
    
    Args:
        model: Trained FSS model
        query_img_path: Path to query image
        support_img_path: Path to support image
        support_mask_path: Path to support mask
        device: Device to run inference on
        img_size: Image size for resizing
    
    Returns:
        pred_mask: Predicted mask tensor
        query_img: Query image tensor
        support_img: Support image tensor
        support_mask: Support mask tensor
    """
    # Load images and masks
    query_img = load_image(query_img_path, img_size).to(device)
    support_img = load_image(support_img_path, img_size).to(device)
    support_mask = load_mask(support_mask_path, img_size).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        pred = model(query_img, support_img, support_mask)
        pred_mask = torch.argmax(pred, dim=1)
    
    return pred_mask[0], query_img[0], support_img[0], support_mask[0]


def main():
    parser = argparse.ArgumentParser(description='FSS Network Inference')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=400,
                       help='Input image size')
    
    # Input parameters
    parser.add_argument('--query_img', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--support_img', type=str, required=True,
                       help='Path to support image')
    parser.add_argument('--support_mask', type=str, required=True,
                       help='Path to support mask')
    parser.add_argument('--gt_mask', type=str, default=None,
                       help='Path to ground truth mask (optional)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save predictions')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization')
    parser.add_argument('--show_vis', action='store_true',
                       help='Show visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = FSSNetwork(pretrained=False).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint IoU: {checkpoint['iou']:.4f}")
    
    # Run prediction
    print("\nRunning prediction...")
    pred_mask, query_img, support_img, support_mask = predict(
        model, args.query_img, args.support_img, args.support_mask,
        device, args.img_size
    )
    
    # Load ground truth if provided
    gt_mask = None
    if args.gt_mask and os.path.exists(args.gt_mask):
        gt_mask = load_mask(args.gt_mask, args.img_size)[0]
    
    # Save prediction
    query_name = os.path.splitext(os.path.basename(args.query_img))[0]
    pred_save_path = os.path.join(args.output_dir, f'{query_name}_pred.png')
    save_prediction(pred_mask, pred_save_path)
    
    # Visualize
    if args.save_vis or args.show_vis:
        vis_save_path = os.path.join(args.output_dir, f'{query_name}_vis.png')
        visualize_prediction(
            query_img, support_img, support_mask, pred_mask, gt_mask,
            save_path=vis_save_path if args.save_vis else None,
            show=args.show_vis
        )
    
    print("\nPrediction completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
