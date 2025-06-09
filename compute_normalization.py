import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def compute_normalization_stats(data_dirs, sample_size=None):
    """Compute mean and std for normalization from multiple directories
    
    Args:
        data_dirs: List of directories containing images
        sample_size: If specified, randomly sample this many images
    """
    print(f"🔍 Computing normalization statistics from {len(data_dirs)} directories")
    
    # Collect all image paths
    image_paths = []
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠️  Directory not found: {data_dir}")
            continue
            
        # Find all jpg images
        imgs = list(data_path.glob("*.jpg"))
        image_paths.extend(imgs)
        print(f"   Found {len(imgs)} images in {data_dir}")
    
    if not image_paths:
        raise ValueError("No images found in any directory!")
    
    print(f"   Total: {len(image_paths)} images")
    
    # Sample if requested
    if sample_size and len(image_paths) > sample_size:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, sample_size)
        print(f"   Sampling {sample_size} images for computation")
    
    # Accumulate statistics
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    
    print("\n📊 Processing images...")
    for img_path in tqdm(image_paths, desc="Computing stats"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            # Convert to RGB and normalize to [0,1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float64) / 255.0
            
            # Accumulate pixel statistics
            pixels = image.reshape(-1, 3)
            pixel_sum += pixels.sum(axis=0)
            pixel_sum_sq += (pixels ** 2).sum(axis=0)
            pixel_count += pixels.shape[0]
            
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    if pixel_count == 0:
        raise ValueError("No valid images processed!")
    
    # Compute mean and std
    mean = pixel_sum / pixel_count
    variance = (pixel_sum_sq / pixel_count) - (mean ** 2)
    std = np.sqrt(variance)
    
    # Results
    print(f"\n✅ Computed from {len(image_paths)} images ({pixel_count:,} pixels)")
    print(f"📈 Mean: {mean}")
    print(f"📈 Std:  {std}")
    print()
    print("💻 Code for dataset.py:")
    print(f"A.Normalize(mean={mean.tolist()}, std={std.tolist()})")
    print()
    print("📄 Rounded values:")
    print(f"mean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"std=[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    
    return mean.tolist(), std.tolist()

def main():
    parser = argparse.ArgumentParser(description='Compute dataset normalization statistics')
    parser.add_argument('--data_dirs', nargs='+', required=True,
                        help='Directories containing images (e.g., ../extracted_frames_9182 ../extracted_frames_9183)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample this many images for computation (default: use all)')
    
    args = parser.parse_args()
    
    # Compute stats
    mean, std = compute_normalization_stats(args.data_dirs, args.sample_size)
    
    # Save results
    import json
    results = {
        'mean': mean,
        'std': std,
        'directories': args.data_dirs,
        'sample_size': args.sample_size
    }
    
    with open('normalization_stats.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to normalization_stats.json")

if __name__ == "__main__":
    main() 