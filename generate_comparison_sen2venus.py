import os
import cv2
import numpy as np
import logging
import torch
from collections import OrderedDict
from models.network_swinir import SwinIR
from utils import utils_image as util

# ==========================================================
# CONFIGURATION (JSON-style)
# ==========================================================
CONFIG = {
    "folder_lr": "testsets/Sen2Venus/LR",
    "folder_sr": "superresolution/swinir_sr_realworld_x2_gan/images",
    "folder_hr": "testsets/Sen2Venus/HR",
    "output_dir": "comparisons_sen2venus",
    "num_samples": 20,  # Number of images to process (None for all)
    "font_size": 15,
    "padding": 10
}

def get_image_identifier(filename):
    """Extract the full identifier from filename, removing trailing iteration numbers."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    # If the last part is an iteration number (>= 4 digits), remove it
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 4:
        parts = parts[:-1]
    identifier = '_'.join(parts)
    return identifier

def add_text_label(image, text, font_size=20, position='top'):
    """Add white text label to image (consistent with NWPU script)."""
    img_labeled = image.copy()
    font_scale = font_size / 30.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == 'top':
        text_x = 10
        text_y = text_height + 15
    else:
        text_x = 10
        text_y = img_labeled.shape[0] - 15

    bg_margin = 2
    cv2.rectangle(img_labeled,
                  (text_x - bg_margin, text_y - text_height - bg_margin),
                  (text_x + text_width + bg_margin, text_y + baseline + bg_margin),
                  (0, 0, 0), -1)

    cv2.putText(img_labeled, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return img_labeled

def organize_images_by_identifier(folder_path, is_subdirectory_based=False):
    """Organize images by identifier and sort by modification time (latest first)."""
    from collections import defaultdict
    import glob
    image_dict = defaultdict(list)
    if not os.path.exists(folder_path):
        return image_dict

    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    
    if is_subdirectory_based:
        subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            subdir_images = []
            for pattern in patterns:
                subdir_images.extend(glob.glob(os.path.join(subdir_path, pattern)))
            if subdir_images:
                image_dict[subdir].extend(subdir_images)
    else:
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(folder_path, pattern)))
        for img_path in image_paths:
            identifier = get_image_identifier(img_path)
            image_dict[identifier].append(img_path)

    # Sort each list by modification time, latest first
    for identifier in image_dict:
        image_dict[identifier].sort(key=os.path.getmtime, reverse=True)
    return image_dict

def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print(f"Gathering images from folders...")
    
    # Organize images by identifier and select the latest
    hr_dict = organize_images_by_identifier(CONFIG['folder_hr'], is_subdirectory_based=False)
    lr_dict = organize_images_by_identifier(CONFIG['folder_lr'], is_subdirectory_based=False)
    # Check if SR folder uses subdirectories (like NWPU) or flat (we'll check both if needed, defaulting to flat if subdirs empty)
    sr_dict = organize_images_by_identifier(CONFIG['folder_sr'], is_subdirectory_based=True)
    if not sr_dict:
        sr_dict = organize_images_by_identifier(CONFIG['folder_sr'], is_subdirectory_based=False)

    # Find common identifiers
    common_ids = sorted(list(set(hr_dict.keys()) & set(sr_dict.keys())))
    
    if not common_ids:
        print("Error: No matching identifiers found between HR and SR folders.")
        print(f"HR sample IDs: {list(hr_dict.keys())[:5]}")
        print(f"SR sample IDs: {list(sr_dict.keys())[:5]}")
        return

    count = min(len(common_ids), CONFIG['num_samples']) if CONFIG['num_samples'] else len(common_ids)
    common_ids = common_ids[:count]

    print(f"Generating {len(common_ids)} comparisons...")

    for i, identifier in enumerate(common_ids):
        # Pick the first (latest) image for each identifier
        hr_path = hr_dict[identifier][0]
        sr_path = sr_dict[identifier][0]
        lr_path = lr_dict[identifier][0] if identifier in lr_dict else None

        img_hr = cv2.imread(hr_path)
        img_sr = cv2.imread(sr_path)
        
        if img_hr is None or img_sr is None:
            continue

        h, w = img_hr.shape[:2]

        # Handle LR
        if lr_path and os.path.exists(lr_path):
            img_lr = cv2.imread(lr_path)
            img_lr = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            img_lr = np.zeros_like(img_hr)
            cv2.putText(img_lr, "LR missing", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Resize SR to match HR if necessary
        if img_sr.shape[:2] != (h, w):
            img_sr = cv2.resize(img_sr, (w, h), interpolation=cv2.INTER_CUBIC)

        # Add labels
        img_lr_labeled = add_text_label(img_lr, "LR (Input)", CONFIG['font_size'])
        img_sr_labeled = add_text_label(img_sr, "SR (Output)", CONFIG['font_size'])
        img_hr_labeled = add_text_label(img_hr, "HR (Ground Truth)", CONFIG['font_size'])

        # Create padding
        pad = np.ones((h, CONFIG['padding'], 3), dtype=np.uint8) * 255

        # Combine
        combined = np.concatenate([img_lr_labeled, pad, img_sr_labeled, pad, img_hr_labeled], axis=1)
        
        save_path = os.path.join(CONFIG['output_dir'], f"{identifier}_comparison.png")
        cv2.imwrite(save_path, combined)
        print(f"[{i+1}/{len(common_ids)}] Saved {save_path}")

    print(f"\nDone. Comparisons saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
