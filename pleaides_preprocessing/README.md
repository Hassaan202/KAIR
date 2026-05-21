Pleiades Preprocessing
================================

This document explains the preprocessing pipeline and verification tools in this folder.

Files
-----
- `pipeline3.py` — main preprocessing pipeline (modules: config, load, coregistration A/B/C, scaling, radiometric regression, patch extraction).
- `verify_coregistration.py` — diagnostic suite to verify HR/LR patch alignment.
- `esrgan_mapping_optuna.py` — Optuna-based learning of satellite-specific ESRGAN degradation parameters (`degrade_satellite()` + `load_config()`).
- `apply_esrgan_degradation.py` — apply learned degradation to generate synthetic LR images.

1) pipeline3.py — overview and usage
-----------------------------------
Purpose:
- Preprocess paired HR/LR Pleiades GeoTIFF/JP2 images into matched 8-bit RGB patch pairs suitable for SR training.

Key modules (high-level):
- Configuration: `CONFIG` defaults are in-file; optional `config.json` next to the script overrides keys.
- Data loading: `load_rgb_bands()` reads specified bands (uint16 → H×W×3).
- Spatial coregistration (Module 3): three stages
  - Stage A (ORB + RANSAC homography) — coarse global alignment.
  - Stage B (phase cross-correlation) — sub-pixel global translation.
  - Stage C (ECC) — patch-wise local refinement applied during patch extraction.
- Scaling (Module 4): `scale_to_uint8()` — per-channel percentile stretch (16-bit → 8-bit).
- Radiometric normalisation (Module 5): `fit_and_apply_radiometric_regression()` — linear least-squares fit + optional histogram matching.
- Patch extraction (Module 6): `extract_and_save_patches()` — sliding-window extraction, Stage C ECC per patch, quality filters (nodata, variance, ECC score, SSIM), and save to `OUTPUT_DIR/hr` & `OUTPUT_DIR/lr`.

Run (from repository root):
```bash
python pleaides_preprocessing/pipeline3.py
```
Options:
- Edit `CONFIG` inside `pipeline3.py` or place a `config.json` alongside the script with any keys to override.
- Important config keys to inspect:
  - `HR_IMAGE_PATH`, `LR_IMAGE_PATH`, `OUTPUT_DIR`
  - `HR_RGB_BANDS`, `LR_RGB_BANDS`
  - Coregistration toggles: `COREG_A_ENABLED`, `COREG_B_ENABLED`, `COREG_C_ENABLED`
  - Quality gates: `MIN_ECC_SCORE`, `MIN_SSIM`, `MIN_VARIANCE`, `MAX_NODATA_FRACTION`.

Outputs:
- Patches saved under `<OUTPUT_DIR>/hr/` and `<OUTPUT_DIR>/lr/` plus a timestamped pipeline log file in `<OUTPUT_DIR>`.

2) Verifying coregistration
---------------------------
Script: `verify_coregistration.py` — run against the `OUTPUT_DIR` produced by the pipeline.

Basic usage examples:
```bash
# use pipeline config OUTPUT_DIR (default)
python pleaides_preprocessing/verify_coregistration.py

# override output dir and sample size
python pleaides_preprocessing/verify_coregistration.py --output Lahore_4 --n 50
```

What it does:
- Produces diagnostic outputs in `<OUTPUT_DIR>/verification/`:
  - `blinker/` GIFs toggling HR_down ↔ LR (good for visually spotting sub-pixel shifts)
  - `diff/` per-patch absolute-difference heatmaps and stats (MAE, RMSE, p95, max)
  - `checkerboard/` interleaved tiles to inspect edge continuity
  - `ssim/` SSIM heatmaps and quadrant metrics
  - `summary/` grids / combined GIFs and `report.txt` summarising per-patch and global stats

Interpretation (short):
- Mean SSIM ≳ 0.72 and MAE small → good alignment.
- Blinker: straight static edges → excellent; breathing or jitter → residual shift.
- Diff / SSIM heatmaps: localized red/white at building edges → parallax; global shift shows uniform offset across edges.

3) ESRGAN mapping (`esrgan_mapping_optuna.py`)
---------------------------------------------
Purpose:
- Learn a satellite-aware degradation model (Gaussian blur, resize stages, Gaussian/Poisson noise probabilities) using Optuna to match real LR images.

Key functions:
- `degrade_satellite(img, **params)` — reproduce learned degradation on an HR float image (returns LR float and mod-cropped HR).
- `optimise_degradation_params(...)` — run Optuna study and write results to JSON (default `best_degradation.json`).
- `load_config(path)` / `save_config(...)` — load/save learned params.

Run example (to learn params):
```bash
python pleaides_preprocessing/esrgan_mapping_optuna.py
# or import and call optimise_degradation_params() from Python
```
Notes:
- The script expects paired HR/LR folders (configured at top of the file) and uses multiple image-quality metrics to optimise a composite objective.
- Check `best_degradation.json` produced by the script — it contains `degradation_params` and `meta`.

4) Applying learned degradation (`apply_esrgan_degradation.py`)
---------------------------------------------------------------
Purpose:
- Apply `best_degradation.json` parameters to a directory of HR images to create synthetic LR images suitable for training ESRGAN/Real-ESRGAN-style models.

Usage (edit `CONFIG` or pass values by editing the file):
```bash
python pleaides_preprocessing/apply_esrgan_degradation.py
```
Key options (file `CONFIG`):
- `hr_dir` — source HR images
- `lr_dir` — destination for synthetic LR
- `params_path` — JSON file from the Optuna study (default `best_degradation.json`)
- `sf`, `seed`, `overwrite`, `recursive`, `save_ext`

Programmatic usage example:
```python
from pleaides_preprocessing.esrgan_mapping_optuna import load_config, degrade_satellite
params = load_config('pleaides_preprocessing/best_degradation.json')
# apply degrade_satellite to an hr float32 image (H,W,3) in [0,1]
lr, _ = degrade_satellite(hr_img, **params)
```

5) Quick checklist to verify a good preprocessing run
----------------------------------------------------
- Run `pipeline3.py` and ensure `<OUTPUT_DIR>/hr` and `<OUTPUT_DIR>/lr` contain many patch PNGs.
- Run `verify_coregistration.py --output <OUTPUT_DIR> --n 50` and inspect `<OUTPUT_DIR>/verification/report.txt` and the summary grids.
- Confirm mean SSIM and MAE values are within acceptable ranges (see `verify_coregistration.py` interpretation guide inside the report).
- If global shift is present, try:
  - Re-enable / tune Stage A (ORB) parameters in `CONFIG` (`COREG_A_*`).
  - Increase `COREG_A_MAX_FEATURES` or relax `COREG_A_RANSAC_THRESH` if too strict.
  - If local errors persist, examine `COREG_C_*` ECC settings.
