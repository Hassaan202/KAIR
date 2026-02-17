"""
Image Comparison Generator for Class-based Datasets

This script generates side-by-side comparison images of HR, LR, and SR images
for each class in a dataset. It automatically selects the latest N images from
each class and creates comparison visualizations with labels.

Usage:
------
python generate_comparisons_nwpu.py --folder_hr path/to/hr --folder_lr path/to/lr \
    --folder_sr path/to/sr --output_dir comparisons --num_samples 3

Arguments:
----------
--folder_hr: Path to folder containing high-resolution (ground truth) images
--folder_lr: Path to folder containing low-resolution images
--folder_sr: Path to folder containing super-resolved images
--output_dir: Directory to save comparison images (default: 'comparisons')
--num_samples: Number of comparison images to generate per class (default: 3)
--font_size: Font size for image labels (default: 20)
--padding: Padding between images in pixels (default: 10)
"""

import argparse
import cv2
import numpy as np
import os
from collections import defaultdict
import glob


def get_class_name(filename):
    """
    Extract class name from filename (format: class_file_name.jpg)

    Args:
        filename: Name of the image file

    Returns:
        Class name extracted from filename
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Remove common suffixes like _SwinIR if present
    basename = basename.replace('_SwinIR', '')
    class_name = basename.split('_')[0] if '_' in basename else 'unknown'
    return class_name


def get_image_identifier(filename):
    """
    Extract the full identifier from filename (e.g., 'airplane_airplane_601' from 'airplane_airplane_601.jpg')
    This is used to match images across HR, LR, and SR folders.

    Args:
        filename: Name of the image file or path

    Returns:
        Image identifier without extension or iteration number
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Remove iteration numbers (e.g., _10000, _100000) from SR images
    # These are typically 4-6 digit numbers at the end
    parts = basename.split('_')

    # If the last part is a long number (iteration like 5000, 10000, etc.), remove it
    # We check if it's >= 4 digits to distinguish from image IDs (601, 602, etc.)
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 4:
        parts = parts[:-1]

    # Rejoin to get the identifier
    identifier = '_'.join(parts)
    return identifier


def organize_images_by_identifier(folder_path, is_subdirectory_based=False):
    """
    Organize images by their full identifier (e.g., 'airplane_airplane_601').

    Args:
        folder_path: Path to folder containing images
        is_subdirectory_based: If True, images are organized in subdirectories (SR structure)
                               If False, images are in a flat directory (HR/LR structure)

    Returns:
        Dictionary mapping image identifiers to lists of image paths
    """
    image_dict = defaultdict(list)

    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist")
        return image_dict

    # Support multiple image formats
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

    if is_subdirectory_based:
        # SR structure: images are in subdirectories named class_filename/
        # Each subdirectory contains class_filename_iteration.png files
        subdirs = [d for d in os.listdir(folder_path)
                   if os.path.isdir(os.path.join(folder_path, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)

            # The subdirectory name IS the identifier (e.g., 'airplane_airplane_601')
            identifier = subdir

            # Find all images in this subdirectory
            subdir_images = []
            for pattern in patterns:
                subdir_images.extend(glob.glob(os.path.join(subdir_path, pattern)))

            # Add all images from this subdirectory
            image_dict[identifier].extend(subdir_images)
    else:
        # HR/LR structure: flat directory with class_filename.jpg naming
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(folder_path, pattern)))

        for img_path in image_paths:
            identifier = get_image_identifier(img_path)
            image_dict[identifier].append(img_path)

    # Sort images by modification time (latest first) for each identifier
    for identifier in image_dict:
        image_dict[identifier].sort(key=os.path.getmtime, reverse=True)

    return image_dict


def add_text_label(image, text, font_size=20, position='top'):
    """
    Add white text label to image.

    Args:
        image: Input image (numpy array)
        text: Text to add
        font_size: Font size (default: 20)
        position: Position of text - 'top' or 'bottom' (default: 'top')

    Returns:
        Image with text label added
    """
    # Create a copy to avoid modifying original
    img_labeled = image.copy()

    # Calculate font scale based on font_size
    font_scale = font_size / 30.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate position
    if position == 'top':
        text_x = 10
        text_y = text_height + 15
    else:  # bottom
        text_x = 10
        text_y = img_labeled.shape[0] - 15

    # Add black background for better readability
    bg_margin = 2
    cv2.rectangle(img_labeled,
                  (text_x - bg_margin, text_y - text_height - bg_margin),
                  (text_x + text_width + bg_margin, text_y + baseline + bg_margin),
                  (0, 0, 0), -1)

    # Add white text
    cv2.putText(img_labeled, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return img_labeled


def create_comparison_image(hr_path, lr_path, sr_path, font_size=20, padding=10):
    """
    Create a side-by-side comparison image of HR, LR, and SR.

    Args:
        hr_path: Path to high-resolution image
        lr_path: Path to low-resolution image
        sr_path: Path to super-resolved image
        font_size: Font size for labels (default: 20)
        padding: Padding between images in pixels (default: 10)

    Returns:
        Combined comparison image
    """
    # Read images
    hr_img = cv2.imread(hr_path)
    sr_img = cv2.imread(sr_path)

    if hr_img is None:
        print(f"Error: Could not read HR image: {hr_path}")
        return None
    if sr_img is None:
        print(f"Error: Could not read SR image: {sr_path}")
        return None

    # Read LR image (may need to be upscaled)
    lr_img = cv2.imread(lr_path) if lr_path and os.path.exists(lr_path) else None

    # Ensure all images have the same height (use HR height as reference)
    target_height = hr_img.shape[0]
    target_width = hr_img.shape[1]

    # Resize SR to match HR dimensions
    if sr_img.shape != hr_img.shape:
        sr_img = cv2.resize(sr_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Resize LR to match HR dimensions (nearest neighbor to show pixelation)
    if lr_img is not None:
        lr_img = cv2.resize(lr_img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        # If LR image not found, create a placeholder
        lr_img = np.zeros_like(hr_img)
        cv2.putText(lr_img, "LR Not Found", (50, target_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add labels to each image
    hr_labeled = add_text_label(hr_img, "HR (Ground Truth)", font_size)
    lr_labeled = add_text_label(lr_img, "LR (Input)", font_size)
    sr_labeled = add_text_label(sr_img, "SR (Output)", font_size)

    # Create padding
    pad = np.ones((target_height, padding, 3), dtype=np.uint8) * 255

    # Concatenate images horizontally with padding
    comparison = np.hstack([lr_labeled, pad, sr_labeled, pad, hr_labeled])

    return comparison


def generate_comparisons(folder_hr, folder_lr, folder_sr, output_dir,
                        num_samples=3, font_size=20, padding=10):
    """
    Generate comparison images for each class.

    Args:
        folder_hr: Path to HR images folder
        folder_lr: Path to LR images folder
        folder_sr: Path to SR images folder
        output_dir: Output directory for comparison images
        num_samples: Number of samples per class (default: 3)
        font_size: Font size for labels (default: 20)
        padding: Padding between images (default: 10)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Organize images by identifier
    print("Organizing images by identifier...")
    print("  - HR folder (flat structure)...")
    hr_images = organize_images_by_identifier(folder_hr, is_subdirectory_based=False)
    print("  - LR folder (flat structure)...")
    lr_images = organize_images_by_identifier(folder_lr, is_subdirectory_based=False) if folder_lr else {}
    print("  - SR folder (subdirectory structure)...")
    sr_images = organize_images_by_identifier(folder_sr, is_subdirectory_based=True)

    print(f"\nFound {len(hr_images)} unique images in HR folder")
    print(f"Found {len(lr_images)} unique images in LR folder")
    print(f"Found {len(sr_images)} unique images in SR folder")

    # Debug: show sample identifiers
    if hr_images:
        print(f"\nSample HR identifiers: {list(sorted(hr_images.keys()))[:5]}")
    if sr_images:
        print(f"Sample SR identifiers: {list(sorted(sr_images.keys()))[:5]}")

    # Get common identifiers across HR and SR folders
    common_identifiers = set(hr_images.keys()) & set(sr_images.keys())

    if not common_identifiers:
        print("\nError: No common identifiers found between HR and SR folders!")
        print(f"HR identifiers (first 10): {sorted(hr_images.keys())[:10]}")
        print(f"SR identifiers (first 10): {sorted(sr_images.keys())[:10]}")
        return

    print(f"\nFound {len(common_identifiers)} matching images between HR and SR")

    # Group identifiers by class
    class_identifiers = defaultdict(list)
    for identifier in common_identifiers:
        class_name = get_class_name(identifier)
        class_identifiers[class_name].append(identifier)

    # Sort identifiers within each class by modification time (latest first)
    for class_name in class_identifiers:
        # Sort by the modification time of the SR image (most recent training iteration)
        class_identifiers[class_name].sort(
            key=lambda id: os.path.getmtime(sr_images[id][0]),
            reverse=True
        )

    print(f"Processing {len(class_identifiers)} classes...")
    print(f"Generating up to {num_samples} comparisons per class\n")

    total_generated = 0

    for class_name in sorted(class_identifiers.keys()):
        print(f"Processing class: {class_name}")

        identifiers = class_identifiers[class_name]

        # Take the latest N identifiers
        num_to_process = min(num_samples, len(identifiers))

        if num_to_process == 0:
            print(f"  Warning: No matching images found for class {class_name}")
            continue

        for idx in range(num_to_process):
            identifier = identifiers[idx]

            # Get the latest image for each type
            hr_path = hr_images[identifier][0] if identifier in hr_images else None
            sr_path = sr_images[identifier][0] if identifier in sr_images else None
            lr_path = lr_images[identifier][0] if identifier in lr_images else None

            if hr_path is None or sr_path is None:
                print(f"  Warning: Missing HR or SR image for identifier {identifier}")
                continue

            # Generate comparison
            comparison = create_comparison_image(hr_path, lr_path, sr_path,
                                                font_size, padding)

            if comparison is not None:
                # Save comparison image with identifier in filename
                base_identifier = os.path.basename(identifier)
                output_filename = f"{class_name}_{base_identifier}_comparison.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, comparison)
                print(f"  Generated: {output_filename}")
                total_generated += 1
            else:
                print(f"  Failed to generate comparison for {identifier}")

        print(f"  Completed {num_to_process} comparison(s) for class {class_name}\n")

    print("="*80)
    print(f"Comparison generation complete!")
    print(f"Total comparisons generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate comparison images for class-based datasets')
    parser.add_argument('--folder_hr', type=str, required=True,
                       help='Path to folder containing high-resolution (ground truth) images')
    parser.add_argument('--folder_lr', type=str, default=None,
                       help='Path to folder containing low-resolution images')
    parser.add_argument('--folder_sr', type=str, required=True,
                       help='Path to folder containing super-resolved images')
    parser.add_argument('--output_dir', type=str, default='comparisons',
                       help='Directory to save comparison images (default: comparisons)')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of comparison images to generate per class (default: 3)')
    parser.add_argument('--font_size', type=int, default=20,
                       help='Font size for image labels (default: 20)')
    parser.add_argument('--padding', type=int, default=10,
                       help='Padding between images in pixels (default: 10)')

    args = parser.parse_args()

    # Validate input directories
    if not os.path.exists(args.folder_hr):
        print(f"Error: HR folder does not exist: {args.folder_hr}")
        return

    if not os.path.exists(args.folder_sr):
        print(f"Error: SR folder does not exist: {args.folder_sr}")
        return

    if args.folder_lr and not os.path.exists(args.folder_lr):
        print(f"Warning: LR folder does not exist: {args.folder_lr}")
        print("Will proceed without LR images")

    # Generate comparisons
    generate_comparisons(
        folder_hr=args.folder_hr,
        folder_lr=args.folder_lr,
        folder_sr=args.folder_sr,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        font_size=args.font_size,
        padding=args.padding
    )


if __name__ == '__main__':
    main()

