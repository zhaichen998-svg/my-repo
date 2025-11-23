"""
Dataset loader for SegPPD-101
Few-shot segmentation dataset for wild plant disease and pest
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SegPPD101Dataset(Dataset):
    """
    SegPPD-101 Dataset for few-shot semantic segmentation
    Supports meta-learning episodic training
    """
    
    def __init__(self, root, split='train', img_size=512, n_way=1, k_shot=1):
        """
        Args:
            root: Root directory of SegPPD-101 dataset
            split: 'train', 'val', or 'test'
            img_size: Image size for resizing
            n_way: Number of classes per episode (1 for binary segmentation)
            k_shot: Number of support examples per class
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.n_way = n_way
        self.k_shot = k_shot
        
        # Define transforms
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
        
        # Load dataset structure
        self.data = self._load_data()
        
    def _load_data(self):
        """
        Load dataset structure
        Expected structure:
        root/
            images/
                class_name/
                    img1.jpg
                    img2.jpg
            masks/
                class_name/
                    img1.png
                    img2.png
        """
        data = {}
        images_dir = os.path.join(self.root, 'images')
        masks_dir = os.path.join(self.root, 'masks')
        
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found at {images_dir}")
            return data
        
        # Get all classes
        classes = [d for d in os.listdir(images_dir) 
                  if os.path.isdir(os.path.join(images_dir, d))]
        
        for cls in classes:
            cls_img_dir = os.path.join(images_dir, cls)
            cls_mask_dir = os.path.join(masks_dir, cls)
            
            if not os.path.exists(cls_mask_dir):
                continue
            
            images = sorted([f for f in os.listdir(cls_img_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            samples = []
            for img_name in images:
                img_path = os.path.join(cls_img_dir, img_name)
                mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(cls_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })
            
            if samples:
                data[cls] = samples
        
        return data
    
    def __len__(self):
        """
        Return number of classes (for episodic sampling)
        """
        return len(self.data) if self.data else 1000  # Default to 1000 episodes
    
    def __getitem__(self, idx):
        """
        Generate one episode for meta-learning
        Returns:
            episode: Dictionary containing query and support samples
        """
        if not self.data:
            # Return dummy data if dataset not found
            return self._get_dummy_episode()
        
        # Random sample a class
        available_classes = list(self.data.keys())
        if not available_classes:
            return self._get_dummy_episode()
        
        selected_class = random.choice(available_classes)
        class_samples = self.data[selected_class]
        
        # Need at least k_shot + 1 samples (support + query)
        if len(class_samples) < self.k_shot + 1:
            # If not enough samples, sample with replacement
            samples = random.choices(class_samples, k=self.k_shot + 1)
        else:
            samples = random.sample(class_samples, self.k_shot + 1)
        
        # Split into support and query
        support_samples = samples[:self.k_shot]
        query_sample = samples[self.k_shot]
        
        # Load support images and masks
        support_images = []
        support_masks = []
        for sample in support_samples:
            img = Image.open(sample['image']).convert('RGB')
            mask = Image.open(sample['mask']).convert('L')
            
            support_images.append(self.img_transform(img))
            support_masks.append(self.mask_transform(mask))
        
        # Load query image and mask
        query_img = Image.open(query_sample['image']).convert('RGB')
        query_mask = Image.open(query_sample['mask']).convert('L')
        
        query_img = self.img_transform(query_img)
        query_mask = self.mask_transform(query_mask)
        
        # Stack support images and masks
        support_images = torch.stack(support_images, dim=0)  # [k_shot, 3, H, W]
        support_masks = torch.stack(support_masks, dim=0)    # [k_shot, 1, H, W]
        
        # For 1-shot, we can squeeze the first dimension
        if self.k_shot == 1:
            support_images = support_images.squeeze(0)
            support_masks = support_masks.squeeze(0)
        
        return {
            'query_img': query_img,
            'query_mask': query_mask,
            'support_img': support_images,
            'support_mask': support_masks,
            'class_name': selected_class
        }
    
    def _get_dummy_episode(self):
        """Generate dummy episode when dataset is not available"""
        dummy_img = torch.randn(3, self.img_size, self.img_size)
        dummy_mask = torch.zeros(1, self.img_size, self.img_size)
        
        return {
            'query_img': dummy_img,
            'query_mask': dummy_mask,
            'support_img': dummy_img,
            'support_mask': dummy_mask,
            'class_name': 'dummy'
        }


def build_dataloader(config, split='train'):
    """
    Build dataloader for SegPPD-101
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        
    Returns:
        dataloader: DataLoader instance
    """
    dataset = SegPPD101Dataset(
        root=config['dataset']['root'],
        split=split,
        img_size=config['dataset']['img_size'],
        n_way=config['training']['n_way'],
        k_shot=config['training']['k_shot']
    )
    
    batch_size = config['training']['batch_size'] if split == 'train' else 1
    shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
