import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import shutil
import zipfile
import json

from .model import HierarchicalQualityModel, ComponentAwareLoss
from .dataset import ComponentQualityDataset, get_training_augmentations, get_validation_augmentations

class HierarchicalQualityModule(pl.LightningModule):
    """PyTorch Lightning module for training the hierarchical model"""
    
    def __init__(self, model, loss_fn, learning_rate=1e-4, warmup_epochs=5):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        
        # Metrics for each component
        self.metrics = {}
        for component in ['hole', 'text', 'knob', 'surface', 'overall']:
            self.metrics[f'{component}_acc'] = pl.metrics.Accuracy()
            self.metrics[f'{component}_f1'] = pl.metrics.F1Score()
    
    def forward(self, x, features=None):
        return self.model(x, features)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        predictions = self.model(batch['image'], batch['features'])
        
        # Calculate losses
        losses = self.loss_fn(predictions, batch)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'train_{k}', v, prog_bar=(k in ['total_loss', 'overall_loss']))
        
        # Log accuracies for components with valid labels
        for component in ['hole', 'text', 'knob', 'surface', 'overall']:
            pred_key = f'{component}_quality'
            target_key = f'{component}_quality'
            
            preds = predictions[pred_key].squeeze()
            targets = batch[target_key]
            valid_mask = targets >= 0
            
            if valid_mask.sum() > 0:
                valid_preds = preds[valid_mask]
                valid_targets = targets[valid_mask]
                
                # Update metrics
                self.metrics[f'{component}_acc'](valid_preds, valid_targets.int())
                self.log(f'train_{component}_acc', self.metrics[f'{component}_acc'], 
                        prog_bar=(component == 'overall'))
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        predictions = self.model(batch['image'], batch['features'])
        
        # Calculate losses
        losses = self.loss_fn(predictions, batch)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val_{k}', v, prog_bar=(k in ['total_loss', 'overall_loss']))
        
        # Calculate and log metrics
        for component in ['hole', 'text', 'knob', 'surface', 'overall']:
            pred_key = f'{component}_quality'
            target_key = f'{component}_quality'
            
            preds = predictions[pred_key].squeeze()
            targets = batch[target_key]
            valid_mask = targets >= 0
            
            if valid_mask.sum() > 0:
                valid_preds = preds[valid_mask]
                valid_targets = targets[valid_mask]
                
                # Calculate accuracy and F1
                acc = ((valid_preds > 0.5) == valid_targets).float().mean()
                self.log(f'val_{component}_acc', acc, prog_bar=(component == 'overall'))
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        # Different learning rates for different parts
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.model.seg_decoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.model.feature_processor.parameters(), 'lr': self.learning_rate},
            {'params': self.model.hole_quality_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.text_quality_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.knob_quality_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.surface_quality_head.parameters(), 'lr': self.learning_rate},
            {'params': self.model.overall_quality_head.parameters(), 'lr': self.learning_rate * 1.5},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        
        # Learning rate scheduling with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                return 0.5 ** ((epoch - self.warmup_epochs) // 20)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

def train_hierarchical_model(data_config):
    """Main training function"""
    
    # Create datasets
    train_dataset = ComponentQualityDataset(
        annotations_file=data_config['train_annotations'],
        img_dir=data_config['train_img_dir'],
        transform=get_training_augmentations(),
        phase='train'
    )
    
    val_dataset = ComponentQualityDataset(
        annotations_file=data_config['val_annotations'],
        img_dir=data_config['val_img_dir'],
        transform=get_validation_augmentations(),
        phase='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.get('batch_size', 8),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.get('batch_size', 8),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Create model and loss
    model = HierarchicalQualityModel()
    loss_fn = ComponentAwareLoss()
    
    # Create Lightning module
    pl_module = HierarchicalQualityModule(
        model=model,
        loss_fn=loss_fn,
        learning_rate=data_config.get('learning_rate', 1e-4),
        warmup_epochs=data_config.get('warmup_epochs', 5)
    )
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor='val_overall_acc',
            mode='max',
            save_top_k=3,
            filename='{epoch}-{val_overall_acc:.3f}-{val_total_loss:.3f}',
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_overall_acc',
            mode='max',
            patience=20,
            min_delta=0.001
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.RichProgressBar()
    ]
    
    # Add custom callback for component-wise monitoring
    class ComponentMonitor(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            # Log component quality distribution
            metrics = trainer.callback_metrics
            print("\n" + "="*50)
            print("Component Quality Accuracies:")
            for component in ['hole', 'text', 'knob', 'surface', 'overall']:
                acc_key = f'val_{component}_acc'
                if acc_key in metrics:
                    print(f"  {component.title()}: {metrics[acc_key]:.3f}")
            print("="*50 + "\n")
    
    callbacks.append(ComponentMonitor())
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=data_config.get('epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,  # Mixed precision training
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=data_config.get('accumulate_grad_batches', 4),
        val_check_interval=0.5,
        log_every_n_steps=10,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(pl_module, train_loader, val_loader)
    
    # Test on best checkpoint
    trainer.test(pl_module, val_loader, ckpt_path='best')
    
    return pl_module

def create_submission_package(model_path, output_dir):
    """Create a deployment package with model and inference code"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model checkpoint
    shutil.copy(model_path, os.path.join(output_dir, 'model.ckpt'))
    
    # Create inference script
    inference_script = '''
import torch
import cv2
import numpy as np
from model import HierarchicalQualityModel, QualityInference

def predict_quality(image_path, model_path='model.ckpt'):
    """Predict battery cover quality"""
    
    # Initialize inference engine
    engine = QualityInference(model_path)
    
    # Get predictions
    results, masks = engine.predict(image_path, visualize=False)
    
    # Create detailed report
    report = {
        'overall_verdict': results['overall_quality'],
        'confidence': results['overall_quality_score'],
        'component_assessments': {
            'holes': {
                'quality': results['hole_quality'],
                'score': results['hole_quality_score']
            },
            'text': {
                'quality': results['text_quality'],
                'score': results['text_quality_score']
            },
            'knobs': {
                'quality': results['knob_quality'],
                'score': results['knob_quality_score']
            },
            'surface': {
                'quality': results['surface_quality'],
                'score': results['surface_quality_score']
            }
        },
        'requires_inspection': results['overall_quality'] == 'BAD'
    }
    
    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        report = predict_quality(image_path)
        print(json.dumps(report, indent=2))
    else:
        print("Usage: python inference.py <image_path>")
'''
    
    with open(os.path.join(output_dir, 'inference.py'), 'w') as f:
        f.write(inference_script)
    
    # Create requirements.txt
    requirements = '''torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.19.0
pillow>=8.0.0
shapely>=1.7.0
albumentations>=1.0.0
pytorch-lightning>=1.5.0
'''
    
    with open(os.path.join(output_dir, 'requirements.txt'), 'w') as f:
        f.write(requirements)
    
    # Create README
    readme = '''# Battery Cover Quality Inspection Model

This model performs hierarchical quality assessment of battery covers.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from inference import predict_quality

# Single image prediction
report = predict_quality('path/to/image.jpg')
print(report)
```

## Model Architecture
- Hierarchical quality assessment with separate heads for:
  - Hole quality (good/deformed/blocked detection)
  - Text quality (presence and readability)
  - Knob quality (size comparison)
  - Surface quality
  - Overall quality (combining all components)

## Output Format
```json
{
  "overall_verdict": "GOOD" or "BAD",
  "confidence": 0.95,
  "component_assessments": {
    "holes": {"quality": "GOOD", "score": 0.92},
    "text": {"quality": "GOOD", "score": 0.88},
    "knobs": {"quality": "GOOD", "score": 0.91},
    "surface": {"quality": "GOOD", "score": 0.89}
  },
  "requires_inspection": false
}
```
'''
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme)
    
    # Create deployment zip
    with zipfile.ZipFile(f'{output_dir}.zip', 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), 
                          os.path.relpath(os.path.join(root, file), output_dir))
    
    print(f"Deployment package created: {output_dir}.zip") 