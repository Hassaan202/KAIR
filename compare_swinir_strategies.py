import os
import cv2
import torch
import numpy as np
import logging
from datetime import datetime
from collections import OrderedDict
from models.network_swinir import SwinIR
from utils import utils_image as util

# ==========================================================
# CONFIGURATION
# ==========================================================
CONFIG = {
    "model_path_2x": "superresolution/swinir_sr_realworld_x2_gan/models/40000_G.pth",  # Update with your weight path
    "model_path_4x": "superresolution/swinir_sr_realworld_x4_gan/models/40000_G.pth",  # Update with your weight path
    "testset_L": "testsets/Sen2Venus/LR_x4",                   # Path to LR images
    "testset_H": "testsets/Sen2Venus/HR",                   # Path to HR images (GT)
    "output_dir": "comparsion_results/swinir_comparison",
    "num_samples": 10000,                                  # Number of images to sample (None for all)
    
    # SwinIR parameters (matching the provided JSONs)
    "netG_2x": {
        "upscale": 2,
        "in_chans": 3,
        "img_size": 128,
        "window_size": 8,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv"
    },
    "netG_4x": {
        "upscale": 4,
        "in_chans": 3,
        "img_size": 64,
        "window_size": 8,
        "img_range": 1.0,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": "nearest+conv",
        "resi_connection": "1conv"
    }
}

def load_model(model_path, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinIR(upscale=opt['upscale'], in_chans=opt['in_chans'],
                   img_size=opt['img_size'], window_size=opt['window_size'],
                   img_range=opt['img_range'], depths=opt['depths'],
                   embed_dim=opt['embed_dim'], num_heads=opt['num_heads'],
                   mlp_ratio=opt['mlp_ratio'], upsampler=opt['upsampler'],
                   resi_connection=opt['resi_connection'])
    
    pretrained_model = torch.load(model_path, map_location=device)
    
    # Try different common keys for state_dict, fallback to the object itself
    if 'params_ema' in pretrained_model:
        state_dict = pretrained_model['params_ema']
    elif 'params' in pretrained_model:
        state_dict = pretrained_model['params']
    elif 'netG' in pretrained_model:
        state_dict = pretrained_model['netG']
    elif 'state_dict' in pretrained_model:
        state_dict = pretrained_model['state_dict']
    else:
        state_dict = pretrained_model

    # Remove 'module.' prefix if it exists (for models trained with DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model = model.to(device)
    return model

@torch.no_grad()
def test_swinir(model, img_l):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_l = util.uint2tensor4(img_l).to(device)
    
    # Window size for padding
    window_size = model.window_size
    _, _, h_old, w_old = img_l.size()
    h_pad = (window_size - h_old % window_size) % window_size
    w_pad = (window_size - w_old % window_size) % window_size
    img_l = torch.nn.functional.pad(img_l, (0, w_pad, 0, h_pad), mode='reflect')
    
    output = model(img_l)
    output = output[..., :h_old * model.upscale, :w_old * model.upscale]
    return util.tensor2uint(output)

def add_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)
    # Position text at top-left
    cv2.putText(img, text, (20, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def main():
    # Setup Output
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    log_file = os.path.join(CONFIG["output_dir"], "metrics.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(message)s', filemode='w')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Models
    print("Loading models...")
    model_2x = load_model(CONFIG['model_path_2x'], CONFIG['netG_2x'])
    model_4x = load_model(CONFIG['model_path_4x'], CONFIG['netG_4x'])
    
    # Get Images
    lr_paths = util.get_image_paths(CONFIG['testset_L'])
    hr_paths = util.get_image_paths(CONFIG['testset_H'])
    
    count = min(len(lr_paths), CONFIG['num_samples']) if CONFIG['num_samples'] else len(lr_paths)
    lr_paths = lr_paths[:count]
    
    header = f"{'Image':<20} | {'4x Once PSNR':<15} | {'4x Once SSIM':<15} | {'2x Twice PSNR':<15} | {'2x Twice SSIM':<15}"
    logging.info(header)
    logging.info("-" * len(header))
    print(header)

    results_data = []

    for i, lr_path in enumerate(lr_paths):
        img_name = os.path.splitext(os.path.basename(lr_path))[0]
        hr_path = os.path.join(CONFIG['testset_H'], os.path.basename(lr_path))
        
        if not os.path.exists(hr_path):
            print(f"Skipping {img_name}, HR not found at {hr_path}")
            continue

        # Read images
        img_lr = util.imread_uint(lr_path, n_channels=3)
        img_hr = util.imread_uint(hr_path, n_channels=3)
        
        # 1. 4x Model Applied Once
        print(f"[{i+1}/{count}] Processing {img_name}...")
        res_4x_once = test_swinir(model_4x, img_lr)
        
        # 2. 2x Model Applied Twice
        res_2x_step1 = test_swinir(model_2x, img_lr)
        res_2x_twice = test_swinir(model_2x, res_2x_step1)
        
        # Crop/Resize HR to match 4x if needed (usually they match if correctly prepared)
        h, w = res_4x_once.shape[:2]
        img_hr = cv2.resize(img_hr, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 3. LR Upscaled for display (Bicubic)
        res_lr_upscaled = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Metrics
        psnr_4x = util.calculate_psnr(res_4x_once, img_hr)
        ssim_4x = util.calculate_ssim(res_4x_once, img_hr)
        psnr_2x2 = util.calculate_psnr(res_2x_twice, img_hr)
        ssim_2x2 = util.calculate_ssim(res_2x_twice, img_hr)
        
        logging.info(f"{img_name:<20} | {psnr_4x:<15.4f} | {ssim_4x:<15.4f} | {psnr_2x2:<15.4f} | {ssim_2x2:<15.4f}")
        results_data.append((psnr_4x, ssim_4x, psnr_2x2, ssim_2x2))
        
        # Create Comparison Image
        # Labels: LR (Bicubic), 2x Twice, 4x Once, HR
        img1 = add_label(res_lr_upscaled.copy(), "LR (Bicubic)")
        img2 = add_label(res_2x_twice.copy(), "2x Twice")
        img3 = add_label(res_4x_once.copy(), "4x Once")
        img4 = add_label(img_hr.copy(), "HR")
        
        # Combine side-by-side
        combined = np.concatenate([img1, img2, img3, img4], axis=1)
        
        # Save
        save_path = os.path.join(CONFIG['output_dir'], f"{img_name}_comp.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)) # Change back to BGR for cv2.imwrite

    # Add average results at the end of log file
    if results_data:
        avg_4x_psnr = np.mean([x[0] for x in results_data])
        avg_4x_ssim = np.mean([x[1] for x in results_data])
        avg_2x2_psnr = np.mean([x[2] for x in results_data])
        avg_2x2_ssim = np.mean([x[3] for x in results_data])

        avg_line = f"{'AVERAGE':<20} | {avg_4x_psnr:<15.4f} | {avg_4x_ssim:<15.4f} | {avg_2x2_psnr:<15.4f} | {avg_2x2_ssim:<15.4f}"
        logging.info("-" * len(header))
        logging.info(avg_line)
        print("-" * len(header))
        print(avg_line)

    print(f"\nDone. Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()

