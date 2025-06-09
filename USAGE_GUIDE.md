# Battery Quality Inspection System - Usage Guide

## 🎯 Quick Start

### 1. **Installation**
```bash
cd BatteryAnnotation
python install_dependencies.py
```

### 2. **Simple Debug (1-2 samples)**
```bash
python simple_debug.py
```

### 3. **Complete Pipeline Debug**
```bash
python debug_pipeline.py
```

### 4. **Training**
```bash
python example_usage.py
# Choose option 1: Train a new model
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM, 4GB+ VRAM
- **Storage**: 2GB+ free space

### Dependencies (automatically installed)
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- OpenCV 4.8+
- Albumentations 1.3+
- Shapely 2.0+
- NumPy, Matplotlib, etc.

## 🔧 Configuration

### GPU Training Enabled
✅ The system automatically detects and uses GPU if available:
- **Mixed precision training** for faster training
- **Automatic device selection**
- **Memory optimization**
- **Benchmark mode** for consistent performance

### Training Parameters
```python
config = {
    'batch_size': 4,           # Adjusted for GPU memory
    'epochs': 50,              # Reduced for testing
    'learning_rate': 1e-4,     # Optimized learning rate
    'warmup_epochs': 3,        # Learning rate warmup
    'num_workers': 2,          # Data loading workers
    'accumulate_grad_batches': 8  # Effective batch size: 4*8=32
}
```

## 📊 Debug Scripts Explained

### `simple_debug.py` - **RECOMMENDED FOR TESTING**
**Purpose**: Shows complete pipeline for 1-2 samples with detailed outputs

**What it shows**:
- ✅ GPU detection and setup
- ✅ Raw data loading (images + annotations)
- ✅ Preprocessing pipeline (transforms, masks, features)
- ✅ Model forward pass with detailed outputs
- ✅ Loss calculation breakdown
- ✅ Memory usage tracking

**Output Example**:
```
🖥️  GPU CHECK
✅ GPU Available: NVIDIA GeForce RTX 3080
   Memory: 10.0 GB

📂 LOADING SAMPLE
✅ Found annotation: frame_000348_enhanced_annotation.json
✅ Image loaded: (1080, 1920, 3)
✅ Quality labels:
   - Hole: GOOD
   - Text: GOOD
   - Knob: GOOD
   - Overall: GOOD

🔧 PREPROCESSING
✅ Preprocessing completed
📊 Processed data shapes:
   image: torch.Size([3, 224, 224]) (torch.float32)
   masks: torch.Size([6, 224, 224]) (torch.float32)
   features: torch.Size([12]) (torch.float32)

🤖 MODEL INFERENCE
✅ Model loaded on cuda
📊 Model parameters: 23,507,734
📤 Model outputs:
   hole_quality: torch.Size([1, 1])
      → GOOD (0.734)
   text_quality: torch.Size([1, 1])
      → GOOD (0.892)
   overall_quality: torch.Size([1, 1])
      → GOOD (0.821)
```

### `debug_pipeline.py` - **COMPREHENSIVE ANALYSIS**
**Purpose**: Detailed analysis of the entire pipeline including visualizations

**Additional features**:
- 📊 Dataset statistics
- 🎭 Mask visualization 
- 🔍 Augmentation comparison
- 📈 Quality distribution analysis

## 🏃‍♂️ Step-by-Step Data Flow

### 1. **Raw Data** → **Preprocessed Data**
```python
# Input: Raw image (1080, 1920, 3) + JSON annotation
# ↓ Transforms applied:
#   - Resize to (224, 224)
#   - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#   - Convert to tensor
# Output: Tensor (3, 224, 224)
```

### 2. **Annotations** → **Masks + Features**
```python
# Input: Polygon coordinates from JSON
# ↓ Processing:
#   - Create 6-channel segmentation masks
#   - Extract 12 geometric features
#   - Convert quality labels to numeric (GOOD=1, BAD=0)
# Output: 
#   - masks: (6, 224, 224) - [good_holes, bad_holes, blocked_holes, text, plus_knob, minus_knob]
#   - features: (12,) - [hole_count, hole_qualities, circularity, text_presence, etc.]
```

### 3. **Model Forward Pass**
```python
# Input: image(1, 3, 224, 224) + features(1, 12)
# ↓ Model architecture:
#   - EfficientNet-B4 backbone
#   - Segmentation decoder (6 channels)
#   - 5 quality prediction heads
#   - Attention mechanism
# Output:
#   - segmentation: (1, 6, 224, 224)
#   - hole_quality: (1, 1) 
#   - text_quality: (1, 1)
#   - knob_quality: (1, 1)
#   - surface_quality: (1, 1)
#   - overall_quality: (1, 1)
```

### 4. **Loss Calculation**
```python
# Multi-task loss:
#   - Segmentation loss (BCE)
#   - Component quality losses (BCE with weights)
#   - Overall quality loss (BCE with weights)
# Total loss = seg_weight * seg_loss + component_weight * comp_loss + overall_weight * overall_loss
```

## 🎯 Model Outputs Explained

### Quality Predictions (0-1 probabilities)
- **> 0.5**: GOOD quality
- **< 0.5**: BAD quality

### Component Breakdown:
- **hole_quality**: Overall hole condition (considers all holes)
- **text_quality**: Text region visibility and readability
- **knob_quality**: Plus/minus knob size comparison
- **surface_quality**: Overall surface condition
- **overall_quality**: Combined assessment (uses all components)

### Segmentation Masks:
- **Channel 0**: Good holes
- **Channel 1**: Deformed holes  
- **Channel 2**: Blocked holes
- **Channel 3**: Text region
- **Channel 4**: Plus knob
- **Channel 5**: Minus knob

## 🚀 Performance Optimization

### GPU Optimizations Applied:
- **Mixed Precision Training**: 16-bit floating point for speed
- **Gradient Accumulation**: Simulate larger batch sizes
- **Benchmark Mode**: Optimize for consistent input sizes
- **Pin Memory**: Faster CPU→GPU transfer
- **Non-deterministic Operations**: Allow CUDA optimizations

### Memory Management:
- **Automatic batch size**: Adjusted based on GPU memory
- **Gradient checkpointing**: Reduce memory usage
- **Empty cache**: Clear GPU memory between runs

## 🐛 Troubleshooting

### Common Issues:

#### 1. **Import Errors**
```bash
# Run the installer
python install_dependencies.py
```

#### 2. **GPU Not Detected**
```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

#### 3. **Out of Memory**
```python
# Reduce batch size in config
config['batch_size'] = 2
config['accumulate_grad_batches'] = 16
```

#### 4. **Path Issues**
```python
# The debug scripts automatically check multiple path patterns:
# - '/o:/Amaron/extracted_frames_9182/*'
# - '../extracted_frames_9182/*'
# - 'extracted_frames_9182/*'
```

## 📁 File Structure Summary

```
BatteryAnnotation/
├── simple_debug.py          # 🎯 START HERE - Quick debug
├── debug_pipeline.py        # 📊 Comprehensive analysis  
├── install_dependencies.py  # 🔧 Dependency installer
├── requirements.txt         # 📦 All dependencies
├── dataset.py              # 📊 Data loading & preprocessing
├── model.py                # 🤖 Model architecture
├── train.py                # 🏋️ Training pipeline (GPU optimized)
├── inference.py            # 🔍 Inference utilities
├── example_usage.py        # 📋 Complete examples
├── train_pipeline.py       # 🔄 Main pipeline (updated)
├── __init__.py            # 📦 Package initialization
└── USAGE_GUIDE.md         # 📖 This guide
```

## 🎉 Success Indicators

When everything is working correctly, you should see:
- ✅ GPU detection and utilization
- ✅ Successful data loading from annotations
- ✅ Model outputs in expected ranges (0-1 for quality predictions)
- ✅ Reasonable loss values during training
- ✅ Memory usage within GPU limits

## 📞 Quick Commands Reference

```bash
# Install everything
python install_dependencies.py

# Quick test (1-2 samples)
python simple_debug.py

# Full pipeline analysis
python debug_pipeline.py

# Interactive training
python example_usage.py

# Direct training
python train_pipeline.py train
``` 