import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import FSSNetwork
from data import SegPPDDataset, MetaLearningDataLoader
from utils import compute_iou, compute_dice


class FSSTrainer:
    """Trainer for FSS Network with Meta-learning strategy"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        print("Initializing FSS Network...")
        self.model = FSSNetwork(pretrained=True).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.lr_step, 
            gamma=config.lr_gamma
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Data loaders
        print("Loading dataset...")
        train_dataset = SegPPDDataset(
            root_dir=config.data_root,
            split='train',
            img_size=config.img_size
        )
        
        self.train_loader = MetaLearningDataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            n_way=1,
            k_shot=config.k_shot,
            num_workers=config.num_workers
        )
        
        # Validation dataset (if available)
        val_root = os.path.join(os.path.dirname(config.data_root), 'val')
        if os.path.exists(val_root):
            val_dataset = SegPPDDataset(
                root_dir=val_root,
                split='val',
                img_size=config.img_size
            )
            self.val_loader = MetaLearningDataLoader(
                dataset=val_dataset,
                batch_size=config.batch_size,
                n_way=1,
                k_shot=config.k_shot,
                num_workers=config.num_workers
            )
        else:
            self.val_loader = None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Best metrics
        self.best_iou = 0.0
        
        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        
        # Create iterator
        train_iter = iter(self.train_loader)
        num_iters = self.config.iters_per_epoch
        
        pbar = tqdm(range(num_iters), desc=f"Epoch {epoch}/{self.config.epochs}")
        
        for i in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Move to device
            query_img = batch['query_img'].to(self.device)
            query_mask = batch['query_mask'].to(self.device)
            support_img = batch['support_img'].to(self.device)
            support_mask = batch['support_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(query_img, support_img, support_mask)
            
            # Compute loss
            query_mask_long = query_mask.long()
            loss = self.criterion(pred, query_mask_long)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                iou = compute_iou(pred, query_mask_long)
                dice = compute_dice(pred, query_mask_long)
            
            epoch_loss += loss.item()
            epoch_iou += iou
            epoch_dice += dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })
            
            # Log to tensorboard
            global_step = epoch * num_iters + i
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/IoU', iou, global_step)
            self.writer.add_scalar('Train/Dice', dice, global_step)
        
        # Epoch averages
        epoch_loss /= num_iters
        epoch_iou /= num_iters
        epoch_dice /= num_iters
        
        return epoch_loss, epoch_iou, epoch_dice
    
    def validate(self, epoch):
        """Validate the model"""
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        
        val_iter = iter(self.val_loader)
        num_val_iters = min(100, self.config.iters_per_epoch // 10)  # Validate on subset
        
        with torch.no_grad():
            for i in tqdm(range(num_val_iters), desc="Validation"):
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    batch = next(val_iter)
                
                # Move to device
                query_img = batch['query_img'].to(self.device)
                query_mask = batch['query_mask'].to(self.device)
                support_img = batch['support_img'].to(self.device)
                support_mask = batch['support_mask'].to(self.device)
                
                # Forward pass
                pred = self.model(query_img, support_img, support_mask)
                
                # Compute loss
                query_mask_long = query_mask.long()
                loss = self.criterion(pred, query_mask_long)
                
                # Compute metrics
                iou = compute_iou(pred, query_mask_long)
                dice = compute_dice(pred, query_mask_long)
                
                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
        
        val_loss /= num_val_iters
        val_iou /= num_val_iters
        val_dice /= num_val_iters
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        self.writer.add_scalar('Val/IoU', val_iou, epoch)
        self.writer.add_scalar('Val/Dice', val_dice, epoch)
        
        return val_loss, val_iou, val_dice
    
    def save_checkpoint(self, epoch, iou, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iou': iou,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with IoU: {iou:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config.epochs}")
        print(f"Iterations per epoch: {self.config.iters_per_epoch}")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_iou, train_dice = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            
            # Validate
            if epoch % self.config.val_freq == 0:
                val_loss, val_iou, val_dice = self.validate(epoch)
                print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
                
                # Save checkpoint
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                
                self.save_checkpoint(epoch, val_iou, is_best)
            else:
                # Save checkpoint based on train IoU
                is_best = train_iou > self.best_iou
                if is_best:
                    self.best_iou = train_iou
                
                self.save_checkpoint(epoch, train_iou, is_best)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, epoch)
            print(f"Learning rate: {current_lr:.6f}\n")
        
        print(f"Training completed! Best IoU: {self.best_iou:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train FSS Network')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='./dataset/train',
                       help='Path to training dataset')
    parser.add_argument('--img_size', type=int, default=400,
                       help='Input image size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--iters_per_epoch', type=int, default=500,
                       help='Number of iterations per epoch')
    parser.add_argument('--k_shot', type=int, default=1,
                       help='Number of support samples (k-shot)')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=50,
                       help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                       help='Learning rate decay factor')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--val_freq', type=int, default=5,
                       help='Validation frequency (epochs)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train
    trainer = FSSTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
