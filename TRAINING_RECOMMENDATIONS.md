# Training Recommendations & Data Guidelines

## 🚀 **Batch Inference is Fixed!**

### ✅ **Batch Commands That Now Work**

```bash
# Batch inference without visualization
python inference.py --model_path best_model.ckpt --image_dir production_images/ --output_path results.json

# Batch inference WITH visualization (NEW!)
python inference.py --model_path best_model.ckpt --image_dir production_images/ --output_path results.json --visualize --viz_output_dir predictions/

# Custom normalization + visualization
python inference.py --model_path best_model.ckpt --image_dir production_images/ --output_path results.json --norm_stats custom_normalization.json --visualize
```

---

## 📊 **Training Recommendations**

### **Your Current Situation**
- ✅ ~200 valid samples (after filtering ~600 total)
- ❌ 1 epoch results (obviously poor!)
- 🎯 Need training strategy

### **Recommended Epochs**

#### **For 200 Samples:**
```bash
# MINIMUM training
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 150

# RECOMMENDED training  
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 300

# OPTIMAL training (if you have time)
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 500
```

#### **Why So Many Epochs?**

| Dataset Size | Recommended Epochs | Reasoning |
|--------------|-------------------|-----------|
| 200 samples  | 300-500 epochs   | Small dataset needs more iterations |
| 500 samples  | 200-300 epochs   | Medium dataset |
| 1000+ samples| 100-200 epochs   | Large dataset converges faster |

**Your 200 samples split into:**
- Training: ~160 samples
- Validation: ~40 samples

This is quite small for deep segmentation, so the model needs many epochs to learn properly.

---

## 📋 **Sample Labeling Guidelines**

### **Current Class Distribution Analysis**

You should check your class balance. Run this to analyze:

```bash
python -c "
from dataset import ComponentQualityDataset
import json

dataset = ComponentQualityDataset(['../extracted_frames_9182', '../extracted_frames_9183'], split='train')

# Analyze quality distribution
hole_good, hole_bad = 0, 0
text_good, text_bad = 0, 0
knob_good, knob_bad = 0, 0
overall_good, overall_bad = 0, 0

for i in range(len(dataset)):
    sample = dataset[i]
    
    # Count quality labels
    if sample['hole_quality'] == 1: hole_good += 1
    elif sample['hole_quality'] == 0: hole_bad += 1
    
    if sample['text_quality'] == 1: text_good += 1
    elif sample['text_quality'] == 0: text_bad += 1
    
    if sample['knob_quality'] == 1: knob_good += 1
    elif sample['knob_quality'] == 0: knob_bad += 1
    
    if sample['overall_quality'] == 1: overall_good += 1
    elif sample['overall_quality'] == 0: overall_bad += 1

print(f'📊 CLASS DISTRIBUTION:')
print(f'Hole Quality: {hole_good} GOOD, {hole_bad} BAD')
print(f'Text Quality: {text_good} GOOD, {text_bad} BAD') 
print(f'Knob Quality: {knob_good} GOOD, {knob_bad} BAD')
print(f'Overall Quality: {overall_good} GOOD, {overall_bad} BAD')
"
```

### **Recommended Sample Distribution (Per Component)**

#### **Minimum Viable Dataset:**
| Component | GOOD Samples | BAD Samples | Total |
|-----------|--------------|-------------|-------|
| Holes     | 60           | 40          | 100   |
| Text      | 60           | 40          | 100   |  
| Knobs     | 60           | 40          | 100   |
| Surface   | 60           | 40          | 100   |
| **Overall** | **120**    | **80**      | **200** |

#### **Recommended Dataset:**
| Component | GOOD Samples | BAD Samples | Total |
|-----------|--------------|-------------|-------|
| Holes     | 150          | 100         | 250   |
| Text      | 150          | 100         | 250   |
| Knobs     | 150          | 100         | 250   |
| Surface   | 150          | 100         | 250   |
| **Overall** | **300**    | **200**     | **500** |

#### **Optimal Dataset:**
| Component | GOOD Samples | BAD Samples | Total |
|-----------|--------------|-------------|-------|
| Holes     | 300          | 200         | 500   |
| Text      | 300          | 200         | 500   |
| Knobs     | 300          | 200         | 500   |
| Surface   | 300          | 200         | 500   |
| **Overall** | **600**    | **400**     | **1000** |

### **Class Balance Strategy**

#### **Current Issues (Likely):**
- Too many GOOD samples, not enough BAD samples
- Model learns to always predict GOOD → high accuracy but useless
- Need more diversity in BAD examples

#### **Labeling Priority:**
1. **🔴 HIGH PRIORITY**: Label more BAD examples
   - Deformed holes
   - Blocked holes  
   - Missing/unclear text
   - Wrong knob sizes
   - Surface defects

2. **🟡 MEDIUM PRIORITY**: Balance GOOD examples
   - Perfect holes
   - Clear text
   - Correct knob proportions

3. **🟢 LOW PRIORITY**: Edge cases
   - Partially blocked holes
   - Faded text
   - Minor surface issues

---

## ⚙️ **Training Strategy**

### **Phase 1: Quick Assessment (Start Here)**
```bash
# Train for 50 epochs to see if model is learning
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 50 --batch_size 4

# Check if validation accuracy is improving
tensorboard --logdir logs/
```

**Expected Results:**
- Epoch 1: ~50% accuracy (random)
- Epoch 10: ~60-65% accuracy  
- Epoch 25: ~70-75% accuracy
- Epoch 50: ~75-80% accuracy

### **Phase 2: Proper Training**
```bash
# If Phase 1 shows improvement, run full training
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 300 --batch_size 8
```

### **Phase 3: Fine-tuning (If Needed)**
```bash
# Resume from best checkpoint with lower learning rate
python train.py --data_dirs ../extracted_frames_9182 ../extracted_frames_9183 --epochs 100 --learning_rate 5e-5 --resume checkpoints/best_model.ckpt
```

---

## 📈 **Expected Training Timeline**

### **With 200 Samples:**

| Epochs | Training Time | Expected Val Accuracy | Notes |
|--------|---------------|---------------------|--------|
| 50     | ~1 hour       | 65-75%             | Quick check |
| 150    | ~3 hours      | 75-85%             | Decent model |
| 300    | ~6 hours      | 80-90%             | Good model |
| 500    | ~10 hours     | 85-92%             | Best with current data |

### **Signs of Good Training:**
- ✅ Validation accuracy steadily increasing
- ✅ Training and validation loss decreasing
- ✅ No large gap between train/val accuracy (<10%)

### **Signs of Problems:**
- ❌ Validation accuracy stuck at ~50-60% (class imbalance)
- ❌ Large train/val gap (>15%) → overfitting
- ❌ Loss not decreasing → learning rate too high/low

---

## 🎯 **Next Steps Recommendations**

### **Immediate Actions:**
1. **Run class distribution analysis** (code above)
2. **Start Phase 1 training** (50 epochs)
3. **Check results** with tensorboard

### **If Results Are Poor (<70% after 50 epochs):**
1. **Label more BAD examples** (prioritize diversity)
2. **Check data quality** (run validation script)
3. **Consider data augmentation** increase

### **If Results Are Good (>75% after 50 epochs):**
1. **Run full 300 epoch training**
2. **Test batch inference** with visualization
3. **Evaluate on real production data**

### **Sample Commands to Test:**

```bash
# 1. Check current data balance
python -c "from dataset import ComponentQualityDataset; d = ComponentQualityDataset(['../extracted_frames_9182'], 'train'); print(f'Training samples: {len(d)}')"

# 2. Quick training test  
python train.py --data_dirs ../extracted_frames_9182 --epochs 50 --batch_size 4

# 3. Monitor training
tensorboard --logdir logs/

# 4. Test inference with visualization
python inference.py --model_path lightning_logs/version_0/checkpoints/best*.ckpt --image_dir ../extracted_frames_9182 --visualize --output_path test_results.json
```

The key is to start with the quick 50-epoch test to see if your current data can learn, then scale up based on results! 