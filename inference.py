"""
Inference script for FSS Network
Perform few-shot segmentation on new images
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models import build_fss_network
from utils.utils import load_config, load_checkpoint


class FSSInference:
    """Inference engine for FSS Network"""
    
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        print("Building FSS Network...")
        self.model = build_fss_network(config)
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        load_checkpoint(self.model, None, checkpoint_path)
        self.model.eval()
        
        # Setup transforms
        img_size = config['dataset']['img_size']
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.img_transform(img)
        return img_tensor.unsqueeze(0)  # Add batch dimension
    
    def load_mask(self, mask_path):
        """Load and preprocess mask"""
        mask = Image.open(mask_path).convert('L')
        mask_tensor = self.mask_transform(mask)
        return mask_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, query_img_path, support_img_path, support_mask_path):
        """
        Perform few-shot segmentation
        
        Args:
            query_img_path: Path to query image
            support_img_path: Path to support image
            support_mask_path: Path to support mask
            
        Returns:
            prediction: Segmentation mask as numpy array
        """
        # Load images
        query_img = self.load_image(query_img_path).to(self.device)
        support_img = self.load_image(support_img_path).to(self.device)
        support_mask = self.load_mask(support_mask_path).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(query_img, support_img, support_mask)
            pred = output.argmax(dim=1).squeeze(0)  # [H, W]
        
        # Convert to numpy
        pred_np = pred.cpu().numpy()
        
        return pred_np
    
    def save_prediction(self, prediction, save_path):
        """Save prediction as image"""
        # Convert to uint8 (0 or 255)
        pred_img = (prediction * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(save_path)
        print(f"Prediction saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='FSS Network Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--query', type=str, required=True,
                       help='Path to query image')
    parser.add_argument('--support', type=str, required=True,
                       help='Path to support image')
    parser.add_argument('--support_mask', type=str, required=True,
                       help='Path to support mask')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Path to save prediction')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create inference engine
    inference = FSSInference(config, args.checkpoint)
    
    # Perform prediction
    print("Performing inference...")
    prediction = inference.predict(args.query, args.support, args.support_mask)
    
    # Save prediction
    inference.save_prediction(prediction, args.output)
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
