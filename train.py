"""
Training script for FSS Network
Implements meta-learning approach with episodic training
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_fss_network
from data.dataset import build_dataloader
from utils.utils import (
    load_config, save_checkpoint, compute_iou, compute_dice,
    AverageMeter, get_optimizer, get_scheduler
)


class FSSTrainer:
    """Trainer for FSS Network with meta-learning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        print("Building FSS Network...")
        self.model = build_fss_network(config)
        self.model = self.model.to(self.device)
        
        # Build dataloaders
        print("Building dataloaders...")
        self.train_loader = build_dataloader(config, split='train')
        self.val_loader = build_dataloader(config, split='val')
        
        # Setup optimizer and scheduler
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup logging
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        os.makedirs(config['logging']['save_dir'], exist_ok=True)
        self.writer = SummaryWriter(config['logging']['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_iou = 0.0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()
        
        num_episodes = self.config['training']['episodes_per_epoch']
        pbar = tqdm(range(num_episodes), desc=f"Epoch {self.current_epoch}")
        
        for i in pbar:
            # Get episode (batch of episodes)
            try:
                batch = next(iter(self.train_loader))
            except StopIteration:
                self.train_loader = build_dataloader(self.config, split='train')
                batch = next(iter(self.train_loader))
            
            # Move to device
            query_img = batch['query_img'].to(self.device)
            query_mask = batch['query_mask'].to(self.device).squeeze(1).long()
            support_img = batch['support_img'].to(self.device)
            support_mask = batch['support_mask'].to(self.device)
            
            # Forward pass
            output = self.model(query_img, support_img, support_mask)
            
            # Compute loss
            loss = self.criterion(output, query_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute metrics
            pred = output.argmax(dim=1)
            iou = compute_iou(pred, query_mask)
            dice = compute_dice(pred, query_mask)
            
            # Update meters
            loss_meter.update(loss.item())
            iou_meter.update(iou)
            dice_meter.update(dice)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'iou': f'{iou_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}'
            })
            
            # Log to tensorboard
            if i % self.config['logging']['log_interval'] == 0:
                global_step = self.current_epoch * num_episodes + i
                self.writer.add_scalar('train/loss', loss_meter.avg, global_step)
                self.writer.add_scalar('train/iou', iou_meter.avg, global_step)
                self.writer.add_scalar('train/dice', dice_meter.avg, global_step)
        
        return loss_meter.avg, iou_meter.avg, dice_meter.avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        dice_meter = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                query_img = batch['query_img'].to(self.device)
                query_mask = batch['query_mask'].to(self.device).squeeze(1).long()
                support_img = batch['support_img'].to(self.device)
                support_mask = batch['support_mask'].to(self.device)
                
                # Forward pass
                output = self.model(query_img, support_img, support_mask)
                
                # Compute loss
                loss = self.criterion(output, query_mask)
                
                # Compute metrics
                pred = output.argmax(dim=1)
                iou = compute_iou(pred, query_mask)
                dice = compute_dice(pred, query_mask)
                
                # Update meters
                loss_meter.update(loss.item())
                iou_meter.update(iou)
                dice_meter.update(dice)
        
        return loss_meter.avg, iou_meter.avg, dice_meter.avg
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['epochs']} epochs")
        print(f"Device: {self.device}")
        
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_iou, train_dice = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_dice = self.validate()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_iou', train_iou, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_iou', val_iou, epoch)
            self.writer.add_scalar('epoch/learning_rate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_interval'] == 0:
                save_path = os.path.join(
                    self.config['logging']['save_dir'],
                    f'checkpoint_epoch_{epoch}.pth'
                )
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)
            
            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                best_path = os.path.join(
                    self.config['logging']['save_dir'],
                    'best_model.pth'
                )
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, best_path)
                print(f"Best model saved with IoU: {self.best_iou:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train FSS Network')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer and start training
    trainer = FSSTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
