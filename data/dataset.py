import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SegPPDDataset(Dataset):
    """
    SegPPD-101 Dataset for Few-Shot Semantic Segmentation
    
    Expected directory structure:
    dataset_root/
        images/
            img1.jpg
            img2.jpg
            ...
        masks/
            img1.png
            img2.png
            ...
    """
    def __init__(self, root_dir, split='train', img_size=400):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            img_size: Image size for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Paths
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # Get list of images
        if os.path.exists(self.img_dir):
            self.image_files = sorted([
                f for f in os.listdir(self.img_dir) 
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
        else:
            self.image_files = []
        
        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
                            interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create dummy mask if not exists
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
        
        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask).squeeze(0)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return {
            'image': image,
            'mask': mask,
            'name': img_name
        }


class MetaLearningDataLoader:
    """
    Meta-learning data loader that generates episodes for few-shot segmentation
    Each episode consists of a support set and a query set
    """
    def __init__(self, dataset, batch_size=4, n_way=1, k_shot=1, num_workers=4):
        """
        Args:
            dataset: Instance of SegPPDDataset
            batch_size: Batch size for episodes
            n_way: Number of classes per episode (1 for binary segmentation)
            k_shot: Number of support samples per class
            num_workers: Number of workers for data loading
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_workers = num_workers
        
    def create_episode(self):
        """Create a single episode with support and query samples"""
        indices = np.random.choice(len(self.dataset), size=self.k_shot + 1, replace=False)
        
        # Support samples
        support_samples = [self.dataset[idx] for idx in indices[:self.k_shot]]
        
        # Query sample
        query_sample = self.dataset[indices[-1]]
        
        # Stack support samples if k_shot > 1
        if self.k_shot == 1:
            support_img = support_samples[0]['image'].unsqueeze(0)
            support_mask = support_samples[0]['mask'].unsqueeze(0)
        else:
            support_img = torch.stack([s['image'] for s in support_samples])
            support_mask = torch.stack([s['mask'] for s in support_samples])
            # Average support features later in training
        
        return {
            'query_img': query_sample['image'],
            'query_mask': query_sample['mask'],
            'support_img': support_img.squeeze(0) if self.k_shot == 1 else support_img[0],
            'support_mask': support_mask.squeeze(0) if self.k_shot == 1 else support_mask[0]
        }
    
    def __iter__(self):
        """Iterator for creating batches of episodes"""
        while True:
            batch = []
            for _ in range(self.batch_size):
                episode = self.create_episode()
                batch.append(episode)
            
            # Collate batch
            batch_data = {
                'query_img': torch.stack([e['query_img'] for e in batch]),
                'query_mask': torch.stack([e['query_mask'] for e in batch]),
                'support_img': torch.stack([e['support_img'] for e in batch]),
                'support_mask': torch.stack([e['support_mask'] for e in batch])
            }
            
            yield batch_data
    
    def get_dataloader(self, num_episodes):
        """
        Get a finite dataloader with specified number of episodes
        
        Args:
            num_episodes: Total number of episodes to generate
        """
        class EpisodeDataset(Dataset):
            def __init__(self, meta_loader, num_episodes):
                self.meta_loader = meta_loader
                self.num_episodes = num_episodes
            
            def __len__(self):
                return self.num_episodes
            
            def __getitem__(self, idx):
                return self.meta_loader.create_episode()
        
        episode_dataset = EpisodeDataset(self, num_episodes)
        
        def collate_fn(batch):
            return {
                'query_img': torch.stack([e['query_img'] for e in batch]),
                'query_mask': torch.stack([e['query_mask'] for e in batch]),
                'support_img': torch.stack([e['support_img'] for e in batch]),
                'support_mask': torch.stack([e['support_mask'] for e in batch])
            }
        
        return DataLoader(
            episode_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
