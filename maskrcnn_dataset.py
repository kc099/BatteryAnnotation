import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
from pathlib import Path

class MaskRCNNDataset(Dataset):
    """
    Dataset for Mask R-CNN. Each instance (hole, text, knob) is a separate object.
    Outputs: image, target dict (boxes, labels, masks)
    """
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.samples = self._discover_samples()
        self.quality_map = {'GOOD': 0, 'good': 0, 'BAD': 1, 'bad': 1}

    def _discover_samples(self):
        samples = []
        unknown_labels = ['UNKNOWN', 'unknown', 'Unknown']
        
        for ann_file in self.data_dir.glob('*_enhanced_annotation.json'):
            # Try to find corresponding image file with either .jpg or .png extension
            base_name = ann_file.name.replace('_enhanced_annotation.json', '')
            image_file_jpg = ann_file.with_name(base_name + '.jpg')
            image_file_png = ann_file.with_name(base_name + '.png')
            
            # Check which image file exists
            if image_file_jpg.exists():
                img_file = image_file_jpg
            elif image_file_png.exists():
                img_file = image_file_png
            else:
                print(f"⚠️  No image file found for annotation: {ann_file.name}")
                continue
            
            # Validate annotation and reject UNKNOWN labels
            try:
                with open(ann_file, 'r') as f:
                    ann = json.load(f)
                
                # Check for required quality fields and reject UNKNOWN labels
                quality_fields = ['hole_quality', 'text_quality', 'knob_quality', 'overall_quality']
                has_unknown = False
                
                for field in quality_fields:
                    if field not in ann:
                        has_unknown = True
                        break
                    if ann[field] in unknown_labels:
                        has_unknown = True
                        break
                
                if has_unknown:
                    print(f"⚠️  Rejecting annotation with UNKNOWN labels: {ann_file.name}")
                    continue
                
                samples.append((img_file, ann_file))
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Error loading annotation {ann_file.name}: {e}")
                continue
        
        print(f"✅ Found {len(samples)} valid samples (excluding UNKNOWN labels)")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # Masks: 4 classes (plus knob, minus knob, text, hole)
        masks = []
        boxes = []
        labels = []
        
        # Plus knob
        if ann.get('plus_knob_polygon') and len(ann['plus_knob_polygon']) >= 3:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(ann['plus_knob_polygon'], dtype=np.int32)], 1)
            masks.append(mask)
            poly_np = np.array(ann['plus_knob_polygon'])
            x1, y1 = np.min(poly_np, axis=0)
            x2, y2 = np.max(poly_np, axis=0)
            boxes.append([x1, y1, x2, y2])
            labels.append(1)
        # Minus knob
        if ann.get('minus_knob_polygon') and len(ann['minus_knob_polygon']) >= 3:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(ann['minus_knob_polygon'], dtype=np.int32)], 1)
            masks.append(mask)
            poly_np = np.array(ann['minus_knob_polygon'])
            x1, y1 = np.min(poly_np, axis=0)
            x2, y2 = np.max(poly_np, axis=0)
            boxes.append([x1, y1, x2, y2])
            labels.append(2)
        # Text area
        if ann.get('text_polygon') and len(ann['text_polygon']) >= 3:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(ann['text_polygon'], dtype=np.int32)], 1)
            masks.append(mask)
            poly_np = np.array(ann['text_polygon'])
            x1, y1 = np.min(poly_np, axis=0)
            x2, y2 = np.max(poly_np, axis=0)
            boxes.append([x1, y1, x2, y2])
            labels.append(3)
        # Hole (only one, if present)
        if ann.get('hole_polygons') and len(ann['hole_polygons']) > 0:
            poly = ann['hole_polygons'][0]
            if len(poly) >= 3:
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1)
                masks.append(mask)
                poly_np = np.array(poly)
                x1, y1 = np.min(poly_np, axis=0)
                x2, y2 = np.max(poly_np, axis=0)
                boxes.append([x1, y1, x2, y2])
                labels.append(4)

        # Handle empty case
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, orig_h, orig_w), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        # Perspective points (8-dim, normalized to original image)
        persp = ann.get('perspective_points', [])
        if persp and len(persp) == 4:
            persp = np.array(persp, dtype=np.float32)
            persp[:, 0] /= orig_w
            persp[:, 1] /= orig_h
            persp = persp.flatten()
        else:
            persp = np.zeros(8, dtype=np.float32)
        perspective = torch.tensor(persp, dtype=torch.float32)

        # Other labels
        overall_quality = self.quality_map.get(ann.get('overall_quality', 'GOOD'), 0)  # Default to GOOD if missing
        overall_quality = torch.tensor(overall_quality, dtype=torch.long)
        text_color = torch.tensor(float(ann.get('text_color_present', False)), dtype=torch.float32)
        plus_area = ann.get('plus_knob_area', 0) or 0  # Handle None case
        minus_area = ann.get('minus_knob_area', 0) or 0  # Handle None case
        knob_size = torch.tensor(float(plus_area > minus_area), dtype=torch.float32)

        # Apply transforms BEFORE creating target dict
        if self.transforms:
            # Create keypoints for perspective points (albumentations format)
            keypoints = []
            if torch.any(perspective > 0):
                for i in range(0, 8, 2):
                    keypoints.append([perspective[i] * orig_w, perspective[i+1] * orig_h])
            
            sample = self.transforms(
                image=image, 
                masks=masks.numpy(),
                bboxes=boxes.tolist() if len(boxes) > 0 else [],
                keypoints=keypoints,
                category_ids=labels.tolist() if len(labels) > 0 else []
            )
            image = sample['image']
            masks = torch.as_tensor(sample['masks'], dtype=torch.uint8)
            
            # Update boxes if they exist
            if len(sample.get('bboxes', [])) > 0:
                boxes = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            
            # Update perspective points if they exist
            if len(sample.get('keypoints', [])) > 0:
                new_h, new_w = image.shape[-2:] if isinstance(image, torch.Tensor) else image.shape[:2]
                kpts = sample['keypoints']
                perspective = torch.tensor([
                    coord for i in range(len(kpts)) for coord in [kpts[i][0] / new_w, kpts[i][1] / new_h]
                ] + [0] * (8 - len(kpts) * 2), dtype=torch.float32)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.tensor([]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
            'perspective': perspective,
            'overall_quality': overall_quality,
            'text_color': text_color,
            'knob_size': knob_size
        }

        return image, target 