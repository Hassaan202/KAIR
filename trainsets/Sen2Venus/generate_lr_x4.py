import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the degradation function from the preprocessing pipeline
import sys
# Add workspace root to sys.path to find preprocessing_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from preprocessing_pipeline.degradation_utils import degrade_satellite
from utils import utils_image as util

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "sf": 2,  # 128x128 -> 64x64 (which is effectively 4x for 256x256 HR)
    
    # Sensor-level (Stage 1) - Very light
    "blur_prob_1": 0.1,
    "blur_type_1": "mtf",
    "resize_prob_1": 0.05,
    "poisson_prob_1": 0.02,
    "read_noise_prob_1": 0.02,
    "haze_prob_1": 0.0,
    "jpeg_prob_1": 0.0,
    
    # Transmission (Stage 2) - Negligible
    "blur_prob_2": 0.05,
    "blur_type_2": "mtf",
    "resize_prob_2": 0.0,
    "poisson_prob_2": 0.01,
    "read_noise_prob_2": 0.01,
    "haze_prob_2": 0.0,
    "jpeg_prob_2": 0.0,
    
    # Final steps
    "final_jpeg_prob": 0.0,
    "resize_back_prob": 0.0,
    "isp_prob": 0.0,
    "isp_model": None,
    
    # Noise & MTF ranges (Keep standard but tight for "negligible" effect)
    "noise_level1": 0.1,
    "noise_level2": 1.0,
    "mtf_sigma_optics_range": (1.0, 1.5),
    "mtf_detector_width_range": (1.0, 1.1),
    "mtf_atm_sigma_range": (0.5, 0.8),
    
    # Paths
    "hr_dir": "HR",        # 256x256
    "lr_dir": "LR",        # 128x128 (source for our LRx4)
    "dest_lr_x4": "LR_x4", # 64x64 (target)
    "dest_hr_x4": "HR_x4", # 256x256 (for pair consistency if needed)
    
    "seed": 42
}

def get_degraded_image(img_path, config):
    """Loads image and applies degradation based on config."""
    img = util.imread_uint(img_path, n_channels=3)
    img = util.uint2single(img)
    
    # Seed for reproducibility during comparison
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    
    # degrade_satellite returns (lq, hq)
    # Note: hq is mod-cropped version of input
    img_lq, img_hq = degrade_satellite(
        img,
        sf=config['sf'],
        blur_prob_1=config['blur_prob_1'],
        blur_type_1=config['blur_type_1'],
        resize_prob_1=config['resize_prob_1'],
        poisson_prob_1=config['poisson_prob_1'],
        read_noise_prob_1=config['read_noise_prob_1'],
        haze_prob_1=config['haze_prob_1'],
        jpeg_prob_1=config['jpeg_prob_1'],
        blur_prob_2=config['blur_prob_2'],
        blur_type_2=config['blur_type_2'],
        resize_prob_2=config['resize_prob_2'],
        poisson_prob_2=config['poisson_prob_2'],
        read_noise_prob_2=config['read_noise_prob_2'],
        haze_prob_2=config['haze_prob_2'],
        jpeg_prob_2=config['jpeg_prob_2'],
        final_jpeg_prob=config['final_jpeg_prob'],
        resize_back_prob=config['resize_back_prob'],
        isp_model=config['isp_model'],
        isp_prob=config['isp_prob'],
        noise_level1=config['noise_level1'],
        noise_level2=config['noise_level2'],
        mtf_sigma_optics_range=config['mtf_sigma_optics_range'],
        mtf_detector_width_range=config['mtf_detector_width_range'],
        mtf_atm_sigma_range=config['mtf_atm_sigma_range']
    )
    
    return img_lq, img_hq

def visualize_comparison(num_samples=3):
    """Shows HR, LR(10m), and the new LRx4(20m)."""
    lr_files = [f for f in os.listdir(CONFIG['lr_dir']) if f.endswith('.png')]
    samples = random.sample(lr_files, min(num_samples, len(lr_files)))
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
    if len(samples) == 1: axes = [axes]
    
    for i, filename in enumerate(samples):
        lr_path = os.path.join(CONFIG['lr_dir'], filename)
        hr_path = os.path.join(CONFIG['hr_dir'], filename)
        
        # 1. Load HR (256x256)
        img_hr = util.imread_uint(hr_path, n_channels=3)
        
        # 2. Load LR (128x128)
        img_lr = util.imread_uint(lr_path, n_channels=3)
        
        # 3. Generate LRx4 (64x64)
        img_lq, _ = get_degraded_image(lr_path, CONFIG)
        img_lq_8bit = util.single2uint(img_lq)
        
        axes[i][0].imshow(img_hr)
        axes[i][0].set_title(f"HR (5m) 256x256\n{filename}")
        
        axes[i][1].imshow(img_lr)
        axes[i][1].set_title("LR (10m) 128x128")
        
        axes[i][2].imshow(img_lq_8bit)
        axes[i][2].set_title(f"LRx4 (20m) 64x64\nDegraded (sf={CONFIG['sf']})")
        
    plt.tight_layout()
    plt.show()

def generate_dataset():
    """Batch processes all images."""
    os.makedirs(CONFIG['dest_lr_x4'], exist_ok=True)
    lr_files = [f for f in os.listdir(CONFIG['lr_dir']) if f.endswith('.png')]
    
    print(f"Generating {len(lr_files)} LRx4 images in {CONFIG['dest_lr_x4']}...")
    
    for filename in lr_files:
        lr_path = os.path.join(CONFIG['lr_dir'], filename)
        img_lq, _ = get_degraded_image(lr_path, CONFIG)
        
        dest_path = os.path.join(CONFIG['dest_lr_x4'], filename)
        util.imsave(util.single2uint(img_lq), dest_path)
        
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='visualize', choices=['visualize', 'generate'], 
                        help='Mode: visualize to check quality, generate to save all images.')
    args = parser.parse_args()
    
    # Ensure relative paths work if run from the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.mode == 'visualize':
        visualize_comparison()
    else:
        generate_dataset()

