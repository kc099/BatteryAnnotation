#!/usr/bin/env python3
"""
Battery Quality Training Pipeline

Usage:
    python train.py --data_dir ../extracted_frames_9182 --epochs 100 --batch_size 8
"""

import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path

from model import HierarchicalQualityModel, ComponentAwareLoss
from dataset import ComponentQualityDataset, get_training_augmentations, get_validation_augmentations


class BatteryQualityTrainer(pl.LightningModule):
    """PyTorch Lightning module for battery quality assessment"""
    
    def __init__(self, learning_rate=1e-4, warmup_epochs=5):
        super().__init__()
        self.model = HierarchicalQualityModel(input_size=(544, 960))
        self.loss_fn = ComponentAwareLoss()
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.save_hyperparameters()
    
    def forward(self, x, features=None):
        return self.model(x, features)
    
    def training_step(self, batch, batch_idx):
        predictions = self.model(batch['image'], batch['features'])
        losses = self.loss_fn(predictions, batch)
        
        # Log main losses
        self.log('train_loss', losses['total_loss'], prog_bar=True)
        self.log('train_seg_loss', losses['seg_loss'])
        self.log('train_overall_loss', losses['overall_loss'])
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        predictions = self.model(batch['image'], batch['features'])
        losses = self.loss_fn(predictions, batch)
        
        # Log validation losses
        self.log('val_loss', losses['total_loss'], prog_bar=True)
        self.log('val_seg_loss', losses['seg_loss'])
        self.log('val_overall_loss', losses['overall_loss'])
        
        # Calculate overall accuracy
        overall_preds = torch.sigmoid(predictions['overall_quality']).squeeze()  # Apply sigmoid to logits
        overall_targets = batch['overall_quality']
        valid_mask = overall_targets >= 0
        
        if valid_mask.sum() > 0:
            valid_preds = overall_preds[valid_mask]
            valid_targets = overall_targets[valid_mask]
            acc = ((valid_preds > 0.5) == valid_targets).float().mean()
            self.log('val_acc', acc, prog_bar=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        # Different learning rates for backbone vs heads
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.model.seg_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.feature_processor.parameters(), 'lr': self.learning_rate},
            {'params': [
                *self.model.hole_quality_head.parameters(),
                *self.model.text_quality_head.parameters(), 
                *self.model.knob_quality_head.parameters(),
                *self.model.surface_quality_head.parameters(),
                *self.model.overall_quality_head.parameters()
            ], 'lr': self.learning_rate * 1.5},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # Warmup + cosine annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                return 0.5 ** ((epoch - self.warmup_epochs) // 30)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def train_model(data_dir, epochs=100, batch_size=8, learning_rate=1e-4, num_workers=4):
    """Train the battery quality model"""
    
    print("🔋 BATTERY QUALITY TRAINING PIPELINE")
    print("=" * 50)
    
    # Create datasets
    print("📂 Creating datasets...")
    train_dataset = ComponentQualityDataset(
        data_dir=data_dir,
        split='train',
        train_ratio=0.8,
        transform=get_training_augmentations()
    )
    
    val_dataset = ComponentQualityDataset(
        data_dir=data_dir,
        split='val', 
        train_ratio=0.8,
        transform=get_validation_augmentations()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create model
    model = BatteryQualityTrainer(learning_rate=learning_rate)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            filename='best-{epoch:02d}-{val_acc:.3f}',
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_acc',
            mode='max',
            patience=20,
            min_delta=0.001,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',  # Mixed precision for speed
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger('logs/', name='battery_quality')
    )
    
    # Train model
    print(f"\n🚀 Starting training for {epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\n✅ Training completed!")
    print(f"📦 Best model saved at: {best_model_path}")
    print(f"📊 Best validation accuracy: {trainer.checkpoint_callback.best_model_score:.3f}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train Battery Quality Assessment Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing images and annotations (e.g., extracted_frames_9182)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")
    
    # Train model
    best_model_path = train_model(
        data_dir=data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers
    )
    
    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"Use the saved model for inference: {best_model_path}")


if __name__ == "__main__":
    main() 