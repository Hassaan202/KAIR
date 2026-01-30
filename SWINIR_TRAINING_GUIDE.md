# SwinIR Training Guide

## Quick Start

To train SwinIR, use the dedicated training script:

```bash
# Single GPU training
python main_train_swinir.py --opt options/swinir/train_swinir_sr_classical.json

# Multi-GPU distributed training
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 main_train_swinir.py --opt options/swinir/train_swinir_sr_classical.json --dist True
```

## Available Config Files

The repository includes several pre-configured training configs:

1. **train_swinir_sr_classical.json** - Classical image super-resolution (x2/x3/x4/x8)
2. **train_swinir_sr_lightweight.json** - Lightweight SR model
3. **train_swinir_sr_realworld_x4_psnr.json** - Real-world SR (PSNR-based)
4. **train_swinir_sr_realworld_x4_gan.json** - Real-world SR (GAN-based)
5. **train_swinir_denoising_color.json** - Color image denoising
6. **train_swinir_denoising_gray.json** - Grayscale image denoising
7. **train_swinir_car_jpeg.json** - JPEG compression artifact removal

---

## Configuration Parameters Explained

### 🎯 CRITICAL PARAMETERS (You MUST configure these)

#### **1. Basic Task Settings**
```json
"task": "swinir_sr_classical_patch48_x2"  // Name for organizing outputs
"scale": 2                                 // Upscaling factor: 2, 3, 4, or 8
"n_channels": 3                            // 1 for grayscale, 3 for RGB
```

#### **2. GPU Configuration**
```json
"gpu_ids": [0,1,2,3,4,5,6,7]              // Which GPUs to use
"dist": true                               // Enable distributed training (multi-GPU)
```
- **Single GPU**: Set `"gpu_ids": [0]` and `"dist": false`
- **Multi-GPU**: List all GPUs and set `"dist": true`

#### **3. Dataset Paths** ⚠️ MOST IMPORTANT
```json
"datasets": {
  "train": {
    "dataroot_H": "trainsets/trainH"      // High-resolution training images
    "dataroot_L": "trainsets/trainL"      // Low-resolution training images (optional)
    "H_size": 96                           // HR patch size (LR will be H_size/scale)
    "dataloader_batch_size": 32            // Batch size (adjust based on GPU memory)
    "dataloader_num_workers": 16           // Data loading threads
  },
  "test": {
    "dataroot_H": "testsets/Set5/HR"      // HR test images
    "dataroot_L": "testsets/Set5/LR_bicubic/X2"  // LR test images
  }
}
```

**Important Notes:**
- For classical SR, you can provide only `dataroot_H` - LR images will be generated automatically
- `H_size` must be divisible by window_size (typically 8)
- Batch size depends on your GPU memory (reduce if OOM errors occur)

#### **4. Pre-trained Model (Optional)**
```json
"path": {
  "root": "superresolution"
  "pretrained_netG": null                 // Path to pre-trained model for fine-tuning
}
```
- Set to `null` for training from scratch
- Provide path like `"model_zoo/swinir/swinir_x2.pth"` for fine-tuning

---

### 🏗️ ARCHITECTURE PARAMETERS (Customize model size)

```json
"netG": {
  "net_type": "swinir"
  "upscale": 2                            // Must match "scale" above
  "in_chans": 3                           // Must match "n_channels"
  "img_size": 48                          // LR patch size (H_size/scale)
  "window_size": 8                        // Swin Transformer window size
  "depths": [6, 6, 6, 6, 6, 6]           // Number of blocks per stage
  "embed_dim": 180                        // Feature dimension (larger = more capacity)
  "num_heads": [6, 6, 6, 6, 6, 6]        // Attention heads per stage
  "mlp_ratio": 2                          // MLP expansion ratio
  "upsampler": "pixelshuffle"             // Upsampling method
  "resi_connection": "1conv"              // Residual connection type
}
```

**Model Size Guidelines:**
- **Small/Fast**: `embed_dim: 60`, `depths: [6, 6, 6, 6]`
- **Medium**: `embed_dim: 180`, `depths: [6, 6, 6, 6, 6, 6]` (default)
- **Large/Accurate**: `embed_dim: 240`, `depths: [6, 6, 6, 6, 6, 6, 6, 6]`

**Requirements:**
- `embed_dim` must be divisible by all values in `num_heads`
- `img_size` should be divisible by `window_size`

---

### 📚 TRAINING PARAMETERS

#### **Loss Function**
```json
"train": {
  "G_lossfn_type": "l1"                   // "l1" (preferred) | "l2" | "ssim" | "charbonnier"
  "G_lossfn_weight": 1.0
}
```

#### **Optimizer Settings**
```json
"G_optimizer_type": "adam"
"G_optimizer_lr": 2e-4                    // Learning rate (0.0002)
"G_optimizer_wd": 0                       // Weight decay (L2 regularization)
```

#### **Learning Rate Scheduler**
```json
"G_scheduler_type": "MultiStepLR"
"G_scheduler_milestones": [250000, 400000, 450000, 475000, 500000]  // When to reduce LR
"G_scheduler_gamma": 0.5                  // LR multiplier at each milestone
```
- At iteration 250k, LR becomes `2e-4 * 0.5 = 1e-4`
- At iteration 400k, LR becomes `1e-4 * 0.5 = 5e-5`, etc.

#### **Exponential Moving Average (EMA)**
```json
"E_decay": 0.999                          // EMA decay rate (0 to disable)
```
- Creates smoothed model weights for better test performance
- Set to 0 to disable

#### **Checkpointing**
```json
"checkpoint_test": 5000                   // Test every N iterations
"checkpoint_save": 5000                   // Save model every N iterations
"checkpoint_print": 200                   // Print loss every N iterations
```

---

## 📊 Training Recommendations

### For Quick Testing (Small Dataset)
```json
"dataloader_batch_size": 8
"G_scheduler_milestones": [50000, 100000, 150000, 175000, 200000]
"checkpoint_test": 1000
"checkpoint_save": 1000
```

### For Full Training (DIV2K or similar)
```json
"dataloader_batch_size": 32              // Total across all GPUs
"G_scheduler_milestones": [250000, 400000, 450000, 475000, 500000]
"checkpoint_test": 5000
"checkpoint_save": 5000
```

### Memory Optimization
If you get OOM (Out of Memory) errors:
1. Reduce `dataloader_batch_size` (e.g., 32 → 16 → 8)
2. Reduce `H_size` (e.g., 96 → 64 → 48)
3. Reduce `embed_dim` (e.g., 180 → 120 → 60)
4. Reduce depth (fewer stages in `depths`)

---

## 🎯 Example Configurations

### Example 1: Train x2 SR on DIV2K (8 GPUs)
```json
{
  "task": "my_swinir_x2_experiment",
  "scale": 2,
  "n_channels": 3,
  "gpu_ids": [0,1,2,3,4,5,6,7],
  "dist": true,
  "datasets": {
    "train": {
      "dataroot_H": "trainsets/DIV2K/DIV2K_train_HR",
      "H_size": 96,
      "dataloader_batch_size": 32,
      "dataloader_num_workers": 16
    },
    "test": {
      "dataroot_H": "testsets/Set5/HR",
      "dataroot_L": "testsets/Set5/LR_bicubic/X2"
    }
  },
  "netG": {
    "upscale": 2,
    "img_size": 48,
    "embed_dim": 180
  }
}
```

### Example 2: Train x4 SR on Single GPU
```json
{
  "task": "swinir_x4_single_gpu",
  "scale": 4,
  "gpu_ids": [0],
  "dist": false,
  "datasets": {
    "train": {
      "dataroot_H": "trainsets/my_training_images",
      "H_size": 128,
      "dataloader_batch_size": 4,
      "dataloader_num_workers": 4
    }
  },
  "netG": {
    "upscale": 4,
    "img_size": 32,
    "embed_dim": 120
  },
  "train": {
    "G_optimizer_lr": 1e-4
  }
}
```

---

## 📁 Output Structure

Training will create the following structure:
```
superresolution/
└── swinir_sr_classical_patch48_x2/
    ├── models/
    │   ├── 5000_G.pth          # Model checkpoints
    │   ├── 10000_G.pth
    │   └── ...
    ├── images/
    │   └── (test images during training)
    ├── log/
    │   └── train.log           # Training log
    └── options/
        └── train.json          # Copy of your config
```

---

## 🐛 Common Issues

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size, patch size, or model size

### Issue: "No training data found"
**Solution**: Check `dataroot_H` path is correct and contains images

### Issue: Training is too slow
**Solution**: 
- Increase `dataloader_num_workers`
- Check data is on fast storage (SSD)
- Enable distributed training

### Issue: Loss is NaN
**Solution**: 
- Reduce learning rate (e.g., `2e-4` → `1e-4`)
- Check data normalization
- Reduce batch size

---

## 📝 Summary: Minimum Changes Needed

To train on your own data, modify these in the config file:

1. **Dataset paths**: `dataroot_H` in both train and test sections
2. **GPU settings**: `gpu_ids` and `dist` based on your hardware
3. **Scale factor**: `scale`, `netG.upscale`, and `netG.img_size` (img_size = H_size/scale)
4. **Batch size**: `dataloader_batch_size` based on GPU memory

Everything else can use default values!
