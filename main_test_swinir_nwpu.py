"""
SwinIR Testing Script for Class-based Datasets (e.g., NWPU)

This script evaluates SwinIR on datasets organized by classes, where filenames are
prefixed with class names (e.g., class_file_name.jpg). It provides metrics both
per-class and overall.

Running Instructions:
---------------------
python main_test_swinir_nwpu.py --task classical_sr --scale 2 \
    --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
    --folder_lq path/to/lr_images \
    --folder_gt path/to/hr_images \
    --training_patch_size 48

Examples:
---------
1. Classical SR with scale x2:
   python main_test_swinir_nwpu.py --task classical_sr --scale 2 \
       --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
       --folder_lq testsets/NWPU/LR_bicubic/X2 \
       --folder_gt testsets/NWPU/HR

2. Color Denoising with noise level 15:
   python main_test_swinir_nwpu.py --task color_dn --noise 15 \
       --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth \
       --folder_gt testsets/NWPU/HR

3. With tiling for large images:
   python main_test_swinir_nwpu.py --task classical_sr --scale 4 \
       --model_path model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth \
       --folder_lq testsets/NWPU/LR_bicubic/X4 \
       --folder_gt testsets/NWPU/HR \
       --tile 400 --tile_overlap 32

Arguments:
----------
--task: Task type (classical_sr, lightweight_sr, real_sr, gray_dn, color_dn, jpeg_car)
--scale: Scale factor (1, 2, 3, 4, 8)
--noise: Noise level for denoising tasks (15, 25, 50)
--jpeg: JPEG quality for compression artifact reduction (10, 20, 30, 40)
--training_patch_size: Patch size used during training
--large_model: Use large model (for real_sr)
--model_path: Path to the pretrained model
--folder_lq: Folder containing low-quality images (for SR tasks)
--folder_gt: Folder containing ground-truth images
--tile: Tile size for testing (None for whole image)
--tile_overlap: Overlap between tiles
"""

import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict, defaultdict
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize overall test results
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    test_results['it_ssim'] = []
    test_results['sam'] = []
    test_results['uiqi'] = []
    test_results['rmse'] = []
    test_results['fsim'] = []
    test_results['srer'] = []

    # Initialize per-class test results
    class_results = defaultdict(lambda: {
        'psnr': [], 'ssim': [], 'psnr_y': [], 'ssim_y': [], 'psnr_b': [],
        'it_ssim': [], 'sam': [], 'uiqi': [], 'rmse': [], 'fsim': [], 'srer': []
    })

    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0
    it_ssim, sam, uiqi, rmse, fsim, srer = 0, 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32

        # Extract class name from filename (format: class_file_name.jpg)
        class_name = imgname.split('_')[0] if '_' in imgname else 'unknown'

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            it_ssim = util.calculate_it_ssim(output, img_gt, border=border)
            sam = util.calculate_sam(output, img_gt, border=border)
            uiqi = util.calculate_uiqi(output, img_gt, border=border)
            rmse = util.calculate_rmse(output, img_gt, border=border)
            fsim = util.calculate_fsim(output, img_gt, border=border)
            srer = util.calculate_srer(output, img_gt, border=border)

            # Store overall results
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['it_ssim'].append(it_ssim)
            test_results['sam'].append(sam)
            test_results['uiqi'].append(uiqi)
            test_results['rmse'].append(rmse)
            test_results['fsim'].append(fsim)
            test_results['srer'].append(srer)

            # Store per-class results
            class_results[class_name]['psnr'].append(psnr)
            class_results[class_name]['ssim'].append(ssim)
            class_results[class_name]['it_ssim'].append(it_ssim)
            class_results[class_name]['sam'].append(sam)
            class_results[class_name]['uiqi'].append(uiqi)
            class_results[class_name]['rmse'].append(rmse)
            class_results[class_name]['fsim'].append(fsim)
            class_results[class_name]['srer'].append(srer)

            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                class_results[class_name]['psnr_y'].append(psnr_y)
                class_results[class_name]['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car']:
                psnr_b = util.calculate_psnrb(output, img_gt, border=border)
                test_results['psnr_b'].append(psnr_b)
                class_results[class_name]['psnr_b'].append(psnr_b)
            print('Testing {:d} {:20s} [Class: {:10s}] - PSNR: {:.2f} dB; SSIM: {:.4f}; IT-SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB.'.
                  format(idx, imgname, class_name, psnr, ssim, it_ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {:20s} [Class: {:10s}]'.format(idx, imgname, class_name))

    # Print per-class results
    if img_gt is not None:
        print('\n' + '='*80)
        print('PER-CLASS RESULTS')
        print('='*80)

        for class_name in sorted(class_results.keys()):
            class_data = class_results[class_name]
            num_images = len(class_data['psnr'])

            if num_images == 0:
                continue

            ave_psnr = sum(class_data['psnr']) / num_images
            ave_ssim = sum(class_data['ssim']) / num_images
            ave_it_ssim = sum(class_data['it_ssim']) / num_images
            ave_sam = sum(class_data['sam']) / num_images
            ave_uiqi = sum(class_data['uiqi']) / num_images
            ave_rmse = sum(class_data['rmse']) / num_images
            ave_fsim = sum(class_data['fsim']) / num_images
            ave_srer = sum(class_data['srer']) / num_images

            print(f'\nClass: {class_name} ({num_images} images)')
            print(f'-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
            print(f'-- Average IT-SSIM: {ave_it_ssim:.4f}')
            print(f'-- Average SAM: {ave_sam:.4f} | UIQI: {ave_uiqi:.4f} | RMSE: {ave_rmse:.4f}')
            print(f'-- Average FSIM: {ave_fsim:.4f} | SRER: {ave_srer:.2f} dB')

            if len(class_data['psnr_y']) > 0:
                ave_psnr_y = sum(class_data['psnr_y']) / len(class_data['psnr_y'])
                ave_ssim_y = sum(class_data['ssim_y']) / len(class_data['ssim_y'])
                print(f'-- Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')
            if len(class_data['psnr_b']) > 0:
                ave_psnr_b = sum(class_data['psnr_b']) / len(class_data['psnr_b'])
                print(f'-- Average PSNR_B: {ave_psnr_b:.2f} dB')

    # Print overall results
    if img_gt is not None:
        print('\n' + '='*80)
        print('OVERALL RESULTS')
        print('='*80)

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_it_ssim = sum(test_results['it_ssim']) / len(test_results['it_ssim'])
        ave_sam = sum(test_results['sam']) / len(test_results['sam'])
        ave_uiqi = sum(test_results['uiqi']) / len(test_results['uiqi'])
        ave_rmse = sum(test_results['rmse']) / len(test_results['rmse'])
        ave_fsim = sum(test_results['fsim']) / len(test_results['fsim'])
        ave_srer = sum(test_results['srer']) / len(test_results['srer'])

        print(f'\n{save_dir}')
        print(f'Total images: {len(test_results["psnr"])}')
        print(f'-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
        print(f'-- Average IT-SSIM: {ave_it_ssim:.4f}')
        print(f'-- Average SAM: {ave_sam:.4f} | UIQI: {ave_uiqi:.4f} | RMSE: {ave_rmse:.4f}')
        print(f'-- Average FSIM: {ave_fsim:.4f} | SRER: {ave_srer:.2f} dB')

        if len(test_results['psnr_y']) > 0:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print(f'-- Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')
        if len(test_results['psnr_b']) > 0:
            ave_psnr_b = sum(test_results['psnr_b']) / len(test_results['psnr_b'])
            print(f'-- Average PSNR_B: {ave_psnr_b:.2f} dB')


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}_nwpu'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}_nwpu'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}_nwpu'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}_nwpu'
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # Look for LR image with same name in LR folder
        img_lq_path = os.path.join(args.folder_lq, f'{imgname}{imgext}')
        if not os.path.exists(img_lq_path):
            # Try with scale suffix if direct match not found
            img_lq_path = os.path.join(args.folder_lq, f'{imgname}x{args.scale}{imgext}')
        img_lq = cv2.imread(img_lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 003 real-world image sr (load lq image only)
    elif args.task in ['real_sr']:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 004 grayscale image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['gray_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)

    # 005 color image denoising (load gt image and generate lq image on-the-fly)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        np.random.seed(seed=0)
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)

    # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
    elif args.task in ['jpeg_car']:
        img_gt = cv2.imread(path, 0)
        result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq = cv2.imdecode(encimg, 0)
        img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()

