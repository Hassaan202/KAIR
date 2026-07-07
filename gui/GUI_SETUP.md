# GUI Setup Guide

This guide covers how to install, run, and use the KAIR Super-Resolution Studio — a web GUI for training, inference, and preprocessing using SwinIR.

---

## Prerequisites

- Python ≥ 3.9 with the KAIR virtualenv already set up (see [USAGE.md](USAGE.md))
- Node.js ≥ 18 and npm ≥ 9

> For a complete end-to-end reference including pipeline flowcharts, config tables, and code-level explanations, see [USER_MANUAL.md](USER_MANUAL.md).

---

## 1. Install Backend Dependencies

Activate your KAIR virtual environment first, then install the extra GUI packages:

```bash
# Activate venv (if not already active)
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate.bat       # Windows

# Install GUI backend dependencies
pip install -r gui/backend/requirements.txt
```

---

## 2. Install Frontend Dependencies

```bash
cd gui/frontend
npm install
cd ../..
```

---

## 3. Run the Application

Open **two terminals** from the KAIR project root.

### Terminal 1 — Start the FastAPI backend

```bash
source .venv/bin/activate

# Run from the project root (so relative paths like superresolution/ resolve correctly)
uvicorn gui.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Terminal 2 — Start the React frontend

```bash
cd gui/frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser.

> The Vite dev server automatically proxies all `/api/*` requests to `http://localhost:8000`, so no CORS issues during development.

---

## 4. Pages Overview

### Training (`/training`)

| Field | Description |
|---|---|
| Mode tabs | **PSNR Training** → `main_train_swinir.py` · **GAN Training** → `main_train_swinir_gan.py` |
| Task name | Becomes the output folder under `superresolution/` |
| Scale | Upscaling factor: ×2 / ×3 / ×4 / ×8 |
| Load preset config | Load defaults from any `options/swinir/*.json` |
| Dataset section | HR/LR train & test dirs, batch size, workers, patch size |
| Network section | `embed_dim`, `depths`, `num_heads`, `upsampler`, `resi_connection` |
| Hyperparameters | Loss type, LR, EMA decay, milestones, checkpoint intervals |
| GAN section | Discriminator, perceptual loss, GAN type (only shown in GAN mode) |
| Recent runs | Live list of `superresolution/` task dirs with iteration count |

A temporary config JSON is written to `gui/backend/tmp/` and passed to the training script via `--opt`. The real `options/swinir/` configs are never modified.

---

### Inference (`/inference`)

Features a 3-tab layout:

**Tab 1 — Patched Images (`main_test_swinir_config.py`)**
Runs batch inference on a directory of already-aligned LR patches against an HR ground truth directory.
- Model selection (latest from task or custom `.pth`) with autofilled `MODEL_CONFIG`
- Input/Output dir paths
- Reports 8 metrics across the batch (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER)

**Tab 2 — Raw HR+LR Inference (`raw_inference.py`)**
Performs end-to-end inference on a paired raw satellite image.
- Configure band mappings per sensor (HR and LR band indices) using the sensor profile selector
- Automatically coregisters (ORB + Phase Correlation) and radiometrically matches LR to HR
- Enforces an exact integer scaling factor (e.g. ×2) regardless of sensor resolution differences
- Patches the full image, runs SR, stitches with a Hann-window blend, and computes metrics
- **Band selection**: When a valid image path is entered, the GUI fetches band count and dimensions from the backend and displays a `MetaPill` (e.g. "4 bands · 1024×1024 px · geospatial"). Band selection switches from a text array editor to toggle buttons — click to add/remove bands; the order you click sets the display channel order (badges show 1/2/3 position).
- Resulting SR/LR/HR RGB composites are displayed directly in the GUI
- **Per-band images**: Below the composite viewer, a spectral band grid shows each individual grayscale channel for LR, SR, and HR (click any thumbnail to open full-size). Labelled "Spectral N" using the configured band indices.
- **Per-band metrics table**: PSNR and SSIM broken down per output channel, labelled by the actual spectral band number from the band selection config.

**Tab 3 — LR-Only Inference (`raw_inference.py`)**
Runs SR on a single unlabelled LR image (no HR ground truth).
- Full image patching and stitching
- Band selection with metadata auto-load (same `MetaPill` + toggle-button UX as Tab 2)
- Per-band grayscale SR images displayed below the RGB composite result
- Result SR image is displayed directly in the GUI

**Model Architecture card** — shown in the right column for all three tabs (outside the form, always visible while configuring bands and paths).

**Band Classes reference** — a "Band Classes" button in the sidebar opens a modal listing spectral band profiles for common satellite sensors (Panchromatic, RGB, Multispectral 4-band, Multispectral 8-band, Sentinel-2). Each profile shows band index, name, wavelength range, and description. Use this as a quick reference when setting `lr_bands` / `hr_bands`.

---

### Preprocessing (`/preprocessing`)

#### Tab A — Pleiades Pipeline (`pipeline3.py`)

For **paired HR + LR GeoTIFF / JP2** satellite imagery. Performs:
1. 3-stage coregistration: ORB → Phase Correlation → ECC (each stage independently toggleable)
2. Radiometric regression (linear LR→HR normalisation)
3. Optional histogram matching
4. Sliding-window patch extraction with quality filters (variance, SSIM, ECC score, nodata)
5. Optional post-processing **train/test split** (configurable ratio)

A temporary config JSON is written to `gui/backend/tmp/pipeline3_config.json` and passed to `pipeline3.py` via `--config`. The script's inline `CONFIG` dict and `pleaides_preprocessing/config.json` are **not modified**.

#### Tab B — HR Degradation Pipeline (`run_pipeline.py`)

For **HR-only** images (generates synthetic LR) or **existing HR+LR pairs** (preprocessing only):

| Mode | Description |
|---|---|
| `hr_only` | Reads HR, applies normalisation, degrades to LR |
| `hr_lr_pair` | Reads both HR and LR, preprocesses without degradation |

**Degradation types:**
- `bsrgan` — BSRGAN-style (13-step degradation shuffle)
- `real_esrgan` — Real-ESRGAN two-stage pipeline
- `bsrgan_plus` — Extended BSRGAN with USM sharpening
- `satellite` — MTF-based optics/sensor/atmospheric degradation (recommended for satellite data)

Optional **train/test split** moves a configurable fraction of output images to separate `*_test` directories.

**Degradation file browser** — at the top of the Degradation section:
- **↑ Load from file**: loads a previously saved `.json` degradation config and merges all degradation parameters into the current form state (useful for reusing a config learned by `esrgan_mapping_optuna.py`).
- **↓ Save config**: downloads the current degradation form values as a named `.json` file for reuse or version control.

#### Tab C — Step Preview

Displays image previews emitted by Pipeline A jobs as they run. Each preprocessing stage that produces a preview image (e.g. coregistration overlays, patch quality visualisations) appears as a labelled thumbnail grid. Thumbnails are updated in real time as the job progresses. Click any thumbnail to view it full-size.

---

## 5. Live Logs

All jobs stream stdout/stderr in real time via **Server-Sent Events (SSE)**. The log panel:
- Auto-scrolls to the latest line
- Color-codes warnings (amber), errors (red), completions (green), training metrics (blue)
- Shows a pulsing green dot while the job is running
- Has a **Stop Job** button that sends `SIGTERM` to the subprocess

---

## 6. File Locations

| Path | Purpose |
|---|---|
| `gui/backend/main.py` | FastAPI app entry point |
| `gui/backend/routers/` | API route handlers (training, inference, preprocessing) |
| `gui/backend/services/` | Job manager, config reader, model scanner |
| `gui/backend/schemas/` | Pydantic request/response models |
| `gui/backend/tmp/` | Temporary JSON configs written by the GUI (auto-created) |
| `gui/frontend/src/pages/` | React page components |
| `gui/frontend/src/components/` | Reusable UI components |
| `gui/frontend/src/api/client.js` | Axios API client |

---

## 7. Production Build (optional)

To serve the frontend directly from the FastAPI server (no separate Vite process):

```bash
cd gui/frontend
npm run build           # outputs to gui/frontend/dist/
cd ../..
uvicorn gui.backend.main:app --host 0.0.0.0 --port 8000
```

The FastAPI app automatically detects and serves `gui/frontend/dist/` as static files when the `dist/` directory exists.

---

## 8. API Reference

Interactive Swagger UI: `http://localhost:8000/docs`

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/training/configs` | GET | List available swinir config files |
| `/api/training/config/{name}` | GET | Load a config file |
| `/api/training/runs` | GET | List training runs in `superresolution/` |
| `/api/training/start` | POST | Launch training subprocess |
| `/api/training/stream/{job_id}` | GET | SSE log stream |
| `/api/training/stop/{job_id}` | POST | Cancel training |
| `/api/inference/tasks` | GET | List trained tasks |
| `/api/inference/latest-model/{task}` | GET | Get latest model + autofilled config |
| `/api/inference/config-from-options/{name}` | GET | Extract model config from an options JSON file |
| `/api/inference/config-from-path` | GET | Auto-detect model config from a checkpoint path |
| `/api/inference/start` | POST | Launch patched-image inference (`main_test_swinir_config.py`) |
| `/api/inference/stream/{job_id}` | GET | SSE log stream |
| `/api/inference/stop/{job_id}` | POST | Cancel inference |
| `/api/inference/raw-paired/start` | POST | Launch raw paired inference (`raw_inference.py`, `mode=paired`) |
| `/api/inference/lr-only/start` | POST | Launch LR-only inference (`raw_inference.py`, `mode=lr_only`) |
| `/api/inference/raw/result/{job_id}/{filename}` | GET | Serve a result PNG (display composite or per-band grayscale) |
| `/api/inference/raw/metrics/{job_id}` | GET | Return `metrics.json` from a paired inference job |
| `/api/inference/image-info` | GET | Return band count and pixel dimensions for any image path (`?path=...`) |
| `/api/preprocessing/pipeline3/start` | POST | Launch pipeline3.py |
| `/api/preprocessing/run-pipeline/start` | POST | Launch run_pipeline.py |
| `/api/preprocessing/stream/{job_id}` | GET | SSE log stream |
| `/api/preprocessing/stop/{job_id}` | POST | Cancel preprocessing job |

---

## 9. Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` in backend | Make sure you activated the KAIR venv before running `uvicorn` |
| `npm install` fails (SSL) | `npm config set strict-ssl false` then retry |
| Frontend not connecting to backend | Confirm backend is running on port 8000; check Vite proxy config in `vite.config.js` |
| Training fails immediately | Check `gui/backend/tmp/train_<task>.json` — the generated config may have a path error |
| pipeline3.py config not applied | Confirm the script was patched with `--config` support (see diff in `pleaides_preprocessing/pipeline3.py`) |
| `CUDA out of memory` | Reduce `dataloader_batch_size`, `H_size`, or `embed_dim` in the training form |
| Band selector not appearing (stays as text input) | The `/api/inference/image-info` call returned an error — check that the image path is accessible from the backend's working directory and that `rasterio` or `opencv-python` is installed in the venv |
| Per-band images missing after inference | Ensure `raw_inference.py` at project root is the updated version; the script must save `lr_band_N.png` / `sr_band_N.png` files alongside the display composites |
| Degradation load/save buttons not visible | Scroll to the bottom of the collapsed Degradation section and expand it — the buttons appear at the top of the expanded panel |
| Step Preview tab empty after pipeline run | Preview images are generated only by Pipeline A jobs that emit `PREVIEW_READY` log lines; Pipeline B does not emit previews |
