# GUI Setup Guide

This guide covers how to install, run, and use the KAIR Super-Resolution Studio — a web GUI for training, inference, and preprocessing using SwinIR.

---

## Prerequisites

- Python ≥ 3.9 with the KAIR virtualenv already set up (see [USAGE.md](USAGE.md))
- Node.js ≥ 18 and npm ≥ 9

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

| Field | Description |
|---|---|
| Latest from task | Dropdown of `superresolution/` task names — auto-selects highest `*_E.pth` and autofills `MODEL_CONFIG` from `options/train.json` |
| Custom path | Manual `.pth` path and manual `MODEL_CONFIG` form |
| LR/HR/SR dirs | Input/output paths |
| Tile size | Leave blank for full-image; set to a multiple of `window_size` for tiled inference on large images |
| MODEL_CONFIG | `upscale`, `in_chans`, `img_size`, `window_size`, `embed_dim`, `depths`, `num_heads`, `upsampler`, `resi_connection` |

Metrics reported: PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER.

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
| `/api/training/configs` | GET | List available swinir config files |
| `/api/training/config/{name}` | GET | Load a config file |
| `/api/training/runs` | GET | List training runs in `superresolution/` |
| `/api/training/start` | POST | Launch training subprocess |
| `/api/training/stream/{job_id}` | GET | SSE log stream |
| `/api/training/stop/{job_id}` | POST | Cancel training |
| `/api/inference/tasks` | GET | List trained tasks |
| `/api/inference/latest-model/{task}` | GET | Get latest model + autofilled config |
| `/api/inference/start` | POST | Launch inference |
| `/api/inference/stream/{job_id}` | GET | SSE log stream |
| `/api/preprocessing/pipeline3/start` | POST | Launch pipeline3.py |
| `/api/preprocessing/run-pipeline/start` | POST | Launch run_pipeline.py |
| `/api/preprocessing/stream/{job_id}` | GET | SSE log stream |
| `/api/health` | GET | Health check |

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
