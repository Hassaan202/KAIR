# KAIR Preprocessing Pipeline

A fully configurable preprocessing pipeline for generating training data for super-resolution models. Supports two operational modes and four state-of-the-art degradation models.

## Features

- **Two Pipeline Modes**:
  - **HR-only mode**: Preprocess HR images → apply degradation → generate LR
  - **HR/LR pair mode**: Preprocess existing HR/LR pairs (no degradation)

- **Four Degradation Models**:
  - BSRGAN (ICCV 2021)
  - Real-ESRGAN (ICCVW 2021)
  - BSRGAN-Plus (extended pipeline)
  - Satellite-Optimized (for remote sensing)

- **Preprocessing Options**:
  - Percentile-based normalization
  - Sentinel-2 cloud masking (via s2cloudless)

- **Fully Configurable**: All parameters exposed in JSON config

---

## Usage

### Basic Command

```bash
python preprocessing_pipeline/run_pipeline.py --config preprocessing_pipeline/config.json
```

---

## Pipeline Modes

### 1. HR-only Mode (Default)

Preprocesses HR images, then degrades them to generate LR images.

**Use case**: Generate training data from high-quality images.

**Config**:
```json
{
  "pipeline_mode": "hr_only",
  "degradation_type": "bsrgan",  // or "real_esrgan", "bsrgan_plus", "satellite"
  "scale": 4,
  "input_hr_dir": "trainsets/HR",
  "input_lr_dir": null,  // not used in hr_only mode
  "output_hr_dir": "trainsets/HR_processed",
  "output_lr_dir": "trainsets/LR_processed"
}
```

**Processing flow**:
```
HR image → normalize (optional) → cloud mask (optional) → degrade → LR image
          ↓
          save processed HR
```

---

### 2. HR/LR Pair Mode

Preprocesses existing HR/LR pairs with the **same** operations (no degradation).

**Use case**: Normalize or clean existing training pairs.

**Config**:
```json
{
  "pipeline_mode": "hr_lr_pair",
  "scale": 4,  // not used for degradation, just metadata
  "input_hr_dir": "trainsets/HR_raw",
  "input_lr_dir": "trainsets/LR_raw",  // REQUIRED for pair mode
  "output_hr_dir": "trainsets/HR_processed",
  "output_lr_dir": "trainsets/LR_processed"
}
```

**Processing flow**:
```
HR image → normalize (optional) → cloud mask (optional) → save
LR image → normalize (optional) → cloud mask (optional) → save
```

**Important**: HR and LR images **must have matching filenames** (e.g., `image_001.png` in both directories).

---

## Configuration Reference

### Pipeline Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline_mode` | string | `"hr_only"` | Mode: `"hr_only"` or `"hr_lr_pair"` |
| `degradation_type` | string | `"bsrgan"` | Degradation model (hr_only mode only) |
| `scale` | int | `4` | Downscale factor: 2, 3, 4, or 8 |
| `n_channels` | int | `3` | 1 for grayscale, 3 for RGB |
| `seed` | int/null | `42` | Random seed (null for random) |
| `num_workers` | int | `1` | Parallel workers (1 = sequential) |

### Paths

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_hr_dir` | string | ✅ | Source directory of HR images |
| `input_lr_dir` | string | ⚠️ | Source directory of LR images (required for `hr_lr_pair` mode) |
| `output_hr_dir` | string | ✅ | Where to save processed HR images |
| `output_lr_dir` | string | ✅ | Where to save processed/degraded LR images |
| `supported_extensions` | array | `[".png", ".jpg", ...]` | Image file extensions to process |

### Preprocessing: Normalization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize_enabled` | bool | `false` | Enable percentile normalization |
| `normalize_low_percentile` | float | `2` | Lower percentile for clipping |
| `normalize_high_percentile` | float | `98` | Upper percentile for clipping |

### Preprocessing: Cloud Masking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cloud_mask_enabled` | bool | `false` | Enable s2cloudless masking |
| `cloud_mask_threshold` | float | `0.4` | Cloud probability threshold |
| `cloud_mask_average_over` | int | `4` | Smoothing window |
| `cloud_mask_dilation_size` | int | `2` | Dilation in pixels |
| `cloud_mask_nodata` | float | `0.0` | Value for masked pixels |
| `cloud_mask_auto_scale` | bool | `true` | Auto-scale raw S2 data |

**Note**: Cloud masking requires 10-band Sentinel-2 images and the `s2cloudless` library:
```bash
pip install s2cloudless
```

### Degradation Models

Each degradation model has its own configuration section in `config.json`:

#### BSRGAN
```json
"bsrgan": {
  "jpeg_prob": 0.9,
  "scale2_prob": 0.25,
  "isp_prob": 0.25,
  "noise_level1": 2,
  "noise_level2": 25
}
```

#### Real-ESRGAN
```json
"real_esrgan": {
  "blur_prob_1": 1.0,
  "resize_prob_1": 1.0,
  "gaussian_noise_prob_1": 0.5,
  // ... 19 total parameters (see config.json)
}
```

#### BSRGAN-Plus
```json
"bsrgan_plus": {
  "shuffle_prob": 0.5,
  "use_sharp": false,
  "sharpening_weight": 0.5,
  // ... 10 total parameters
}
```

#### Satellite
```json
"satellite": {
  "blur_prob_1": 1.0,
  "blur_type_1": "mtf",
  "poisson_prob_1": 0.75,
  // ... 22 total parameters including MTF ranges
}
```

See `config.json` for complete parameter lists and documentation.

---

## Examples

### Example 1: Generate Training Data from HR Images

```json
{
  "task": "generate_bsrgan_x4",
  "pipeline_mode": "hr_only",
  "degradation_type": "bsrgan",
  "scale": 4,
  "n_channels": 3,
  "input_hr_dir": "trainsets/DIV2K/HR",
  "output_hr_dir": "trainsets/DIV2K/HR_processed",
  "output_lr_dir": "trainsets/DIV2K/LR_bsrgan_x4",
  "normalize_enabled": false,
  "cloud_mask_enabled": false
}
```

Run:
```bash
python preprocessing_pipeline/run_pipeline.py --config my_config.json
```

---

### Example 2: Normalize Existing HR/LR Pairs

```json
{
  "task": "normalize_satellite_pairs",
  "pipeline_mode": "hr_lr_pair",
  "scale": 4,
  "n_channels": 3,
  "input_hr_dir": "trainsets/satellite/HR_raw",
  "input_lr_dir": "trainsets/satellite/LR_raw",
  "output_hr_dir": "trainsets/satellite/HR_normalized",
  "output_lr_dir": "trainsets/satellite/LR_normalized",
  "normalize_enabled": true,
  "normalize_low_percentile": 2,
  "normalize_high_percentile": 98,
  "cloud_mask_enabled": false
}
```

---

### Example 3: Satellite Data with Cloud Masking

```json
{
  "task": "satellite_sr_x4",
  "pipeline_mode": "hr_only",
  "degradation_type": "satellite",
  "scale": 4,
  "n_channels": 3,
  "input_hr_dir": "trainsets/sentinel2/HR",
  "output_hr_dir": "trainsets/sentinel2/HR_processed",
  "output_lr_dir": "trainsets/sentinel2/LR_satellite_x4",
  "normalize_enabled": true,
  "normalize_low_percentile": 2,
  "normalize_high_percentile": 98,
  "cloud_mask_enabled": true,
  "cloud_mask_threshold": 0.4,
  "satellite": {
    "blur_type_1": "mtf",
    "blur_type_2": "mtf",
    "noise_level1": 0.8,
    "noise_level2": 10.0
  }
}
```

---

## Output

After processing, you'll have:

### HR-only mode:
```
output_hr_dir/
  image_001.png  (mod-cropped HR)
  image_002.png
  ...

output_lr_dir/
  image_001.png  (degraded LR)
  image_002.png
  ...
```

### HR/LR pair mode:
```
output_hr_dir/
  image_001.png  (preprocessed HR)
  image_002.png
  ...

output_lr_dir/
  image_001.png  (preprocessed LR)
  image_002.png
  ...
```

---

## Integration with KAIR Training

The output directories can be directly used with KAIR's training scripts:

```json
// options/swinir/train_swinir_custom.json
{
  "datasets": {
    "train": {
      "dataset_type": "sr",
      "dataroot_H": "trainsets/DIV2K/HR_processed",
      "dataroot_L": "trainsets/DIV2K/LR_bsrgan_x4"
    }
  }
}
```

Then train:
```bash
python main_train_swinir.py --opt options/swinir/train_swinir_custom.json
```

---

## Performance Tips

### Parallel Processing

Enable multi-worker processing for large datasets:

```json
{
  "num_workers": 8  // use 8 CPU cores
}
```

**Note**: Each worker gets its own random seed for reproducibility.

### Memory Considerations

- Large images + degradation can use significant RAM
- If OOM errors occur, reduce `num_workers` to 1
- Consider processing in batches

---

## Troubleshooting

### HR/LR filenames don't match

**Error**: `[WARN] X HR images have no matching LR image`

**Solution**: Ensure HR and LR images have identical filenames (extensions can differ if both are in `supported_extensions`).

### Cloud masking fails

**Error**: `ImportError: s2cloudless is required`

**Solution**: Install the library:
```bash
pip install s2cloudless
```

### Images have wrong number of bands

**Error**: `[WARN] Cloud masking enabled but image does not have 10 bands`

**Solution**: Cloud masking only works with 10-band Sentinel-2 imagery. Disable it for RGB images.

---

## References

- **BSRGAN**: Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution", ICCV 2021
- **Real-ESRGAN**: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data", ICCVW 2021
- **KAIR**: Zhang, "KAIR: A Collection of Image Restoration Methods", GitHub 2021

---

## License

This preprocessing pipeline is part of the KAIR project and follows the same license.

