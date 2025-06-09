# Battery Quality Assessment Pipeline

A robust deep learning pipeline for automated battery cover quality inspection using DeepLabV3+ with ResNet50.

## 🏗️ **Architecture**

- **Backbone**: DeepLabV3+ with ResNet50 (pretrained)
- **Resolution**: 960×544 (aspect ratio preserved)
- **Segmentation**: 6-channel masks (good/deformed/blocked holes + text + knobs)
- **Classification**: Quality assessment for 5 components
- **Normalization**: Dataset-specific values (computed from battery images)

## 🚀 **Quick Start**

### 1. **Training**
```bash
# Train model on your data
python train.py --data_dir ../extracted_frames_9182 --epochs 100 --batch_size 8

# Monitor training
tensorboard --logdir logs/
```

### 2. **Inference**
```bash
# Single image
python inference.py --model_path best_model.ckpt --image_path test_image.jpg

# Batch processing
python inference.py --model_path best_model.ckpt --image_dir test_images/ --output_path results.json
```

## 📂 **Data Structure**

The pipeline automatically discovers and processes data from a directory containing:
```
extracted_frames_9182/
├── frame_000000.jpg
├── frame_000000_enhanced_annotation.json
├── frame_000001.jpg  
├── frame_000001_enhanced_annotation.json
└── ...
```

**No manual file creation needed** - the dataset automatically:
- Matches images with annotations by filename
- Splits into train/validation (80/20)
- Filters by confidence score (>0.7)
- Creates consistent splits with random seed

## 🎯 **Model Outputs**

### Quality Assessment
- **Hole Quality**: Good/Bad hole condition
- **Text Quality**: Text presence and readability  
- **Knob Quality**: Plus/minus knob size comparison
- **Surface Quality**: Overall surface condition
- **Overall Quality**: Combined assessment

### Segmentation Maps
- **Channel 0**: Good holes
- **Channel 1**: Deformed holes
- **Channel 2**: Blocked holes  
- **Channel 3**: Text regions
- **Channel 4**: Plus knob
- **Channel 5**: Minus knob

## ⚙️ **Training Configuration**

**Default Parameters:**
- Epochs: 100
- Batch size: 8
- Learning rate: 1e-4
- Mixed precision: 16-bit
- Early stopping: 20 epochs patience
- Backbone LR: 0.1× (transfer learning)
- Quality heads LR: 1.5× (task-specific)

**Data Augmentation:**
- Aspect ratio preserving resize
- Rotation, shift, scale (mild)
- Color jittering
- Noise injection
- Motion blur

## 📊 **Performance**

**Model Size:** 42.8M parameters  
**Memory Usage:** ~0.05 GB GPU  
**Training Time:** ~1 min/epoch (RTX 3060, batch=8)  
**Inference Speed:** Real-time on GPU  

## 🔧 **Pipeline Flow**

1. **Dataset Discovery**: Auto-find images and annotations
2. **Preprocessing**: Resize, normalize, augment
3. **Training**: Multi-task learning with component-aware loss
4. **Validation**: Real-time accuracy monitoring
5. **Model Saving**: Best model by validation accuracy
6. **Inference**: Load model → predict → output results

## 📋 **Requirements**

See `requirements.txt` for complete dependencies:
- PyTorch 2.0+
- PyTorch Lightning
- OpenCV
- Albumentations
- DeepLabV3+ (torchvision)

## 🎛️ **Advanced Usage**

```bash
# Custom training parameters
python train.py \
    --data_dir /path/to/data \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --num_workers 8

# Batch inference with results export
python inference.py \
    --model_path best_model.ckpt \
    --image_dir production_images/ \
    --output_path inspection_results.json
```

## 🏆 **Key Features**

✅ **Fully Automated**: No manual annotation file creation  
✅ **Robust Architecture**: DeepLabV3+ for high-quality segmentation  
✅ **Mixed Precision**: 16-bit training for speed and memory efficiency  
✅ **Smart Augmentation**: Battery-specific data augmentation  
✅ **Multi-Scale Loss**: Segmentation + classification objectives  
✅ **Production Ready**: Simple inference API  
✅ **Aspect Ratio Preserved**: No distortion during resize  
✅ **GPU Optimized**: Automatic GPU detection and utilization

## Repository Structure and Large Files Management

This repository contains code for battery annotation, but intentionally excludes large media files and extracted frames to keep the repository size manageable. Here's how we handle different types of files:

### Ignored Directories and Files
- `extracted_frames/` - Contains extracted video frames (git-ignored)
- `extracted_frames_9182/` - Contains extracted frames from specific video (git-ignored)
- `labelled_frames/` - Contains annotated frames (git-ignored)
- `*.MOV` - All MOV video files (git-ignored)

### Best Practices for Media Files

1. **Video Files (.MOV)**
   - Store original video files in a separate location (not in git)
   - Use a consistent naming convention for videos
   - Document video metadata (resolution, duration, etc.) in a separate file

2. **Extracted Frames**
   - Frames are extracted from videos for annotation
   - These are stored locally but not in git
   - Use the provided scripts to extract frames consistently
   - Keep frame extraction parameters documented

3. **Labelled Frames**
   - Contains annotation data and processed frames
   - Stored locally but not in git
   - Use consistent annotation format
   - Back up annotation data separately

### Setting Up the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/kc099/BatteryAnnotation.git
   cd BatteryAnnotation
   ```

2. Create required directories (if they don't exist):
   ```bash
   mkdir -p extracted_frames extracted_frames_9182 labelled_frames
   ```

3. Place your video files in the appropriate location (outside git)

### Working with the Repository

1. **Adding New Videos**
   - Place new .MOV files in your working directory
   - They will be automatically ignored by git
   - Use the provided scripts to extract frames

2. **Extracting Frames**
   - Use the provided Python scripts to extract frames
   - Frames will be saved in the appropriate directory
   - These directories are git-ignored

3. **Annotation Process**
   - Annotate frames using the provided tools
   - Save annotations in the labelled_frames directory
   - Keep a backup of your annotations

### Backup Strategy

1. **Local Backups**
   - Regularly backup your media files and annotations
   - Consider using external storage for large files
   - Keep a separate backup of annotation data

2. **Version Control**
   - Only code and configuration files are version controlled
   - Use git for tracking code changes
   - Document any changes to the frame extraction or annotation process

### Troubleshooting

If you accidentally commit large files:
1. Use `git filter-branch` to remove them from history
2. Update .gitignore if needed
3. Force push changes to remote
4. Document the incident and solution

### Contributing

When contributing to this project:
1. Never commit media files or extracted frames
2. Follow the established directory structure
3. Document any changes to the frame extraction process
4. Update this README if you add new features or change the workflow

## Project Structure

```
BatteryAnnotation/
├── .gitignore           # Git ignore rules
├── README.md           # This file
├── scripts/            # Python scripts for processing
├── extracted_frames/   # Extracted frames (git-ignored)
├── extracted_frames_9182/ # Specific video frames (git-ignored)
└── labelled_frames/    # Annotated frames (git-ignored)
```


## Contact

For questions about the project or repository management, please contact the repository maintainer. 