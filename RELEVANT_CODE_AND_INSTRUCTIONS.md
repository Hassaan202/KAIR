# SwinIR: Image Restoration using Swin Transformer 

This repository contains the training and testing code for SwinIR, adapted to include comprehensive evaluation metrics.

## 🧪 Testing

To evaluate the model on test datasets (e.g., Set5), use the following command.

### Usage
```bash
python main_test_swinir.py \
    --task lightweight_sr \
    --scale 4 \
    --model_path model_zoo/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth \
    --folder_lq testsets/Set5/LR_bicubic/X4 \
    --folder_gt testsets/Set5/HR
```

#### ⚠️ Important Notes for Testing
- Filename Suffixes:By default, the folder_lq (Low Quality) images should have filenames appended with the scaling 
factor (e.g., image_x2.png, image_x4.png) to align with the original SwinIR loader logic.
- Note: If the code has been modified to handle non-suffixed filenames, ensure your dataset matches the expected 
format in main_test_swinir.py.
- Paths: Ensure folder_lq and folder_gt point to the correct directories containing your test images

## 🚆 Training
To train the SwinIR model, use the command below. The training configuration is determined by the JSON options files.

Usage
```bash
python main_train_swinir.py
```
### ⚙️ Configuration & Data Preparation
- Config Files: The training configuration files (hyperparameters, paths, etc.) are located in options/swinir/.... 
Modify the appropriate JSON file before running the script.
- Dataset Requirement: * Filename Matching: The Low Resolution (LR) and High Resolution (HR) image pairs must have 
  identical filenames.
- The dataloader relies on exact name matching to pair the inputs and targets correctly. If names differ, the 
  training process will fail or load incorrect pairs.

dataset_sr for direct sr training: uses bicubic if lr not provided, geometric augmentations (crop, flips, rotations) 
and radiometric augmentations applied on the go during training.

dataset_blindsr for training using bsrgan model: performs degradation using bsrgan on-the-fly 