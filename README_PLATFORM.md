# SUPARCO Super-Resolution Lab

Full-stack satellite image super-resolution platform — preprocessing, training, and inference in one place.

## Architecture

```
KAIR/
├── frontend/          Next.js 14 · TypeScript · Tailwind · Zustand · React Query
├── backend/           FastAPI · Celery · Redis · SSE log streaming
│   ├── main.py        API entry point
│   ├── schemas.py     Pydantic models for every config param
│   ├── metrics.py     All 8 metrics (PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER)
│   ├── routers/       preprocessing · training · inference · status
│   └── tasks/         Celery tasks wrapping existing pipeline scripts
└── docker-compose.yml Single-command local deploy
```

## Quick start (local, recommended for GPU training)

### 1. Backend

```bash
# Install Python deps (assumes CUDA PyTorch already installed)
pip install -r backend/requirements.txt

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Start FastAPI
cd KAIR
uvicorn backend.main:app --reload --port 8000

# Start Celery worker (separate terminal)
celery -A backend.tasks.celery_app worker --loglevel=info --concurrency=1
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

Set the backend URL if not on localhost:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://your-server:8000
```

## Docker Compose (all services)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

> **GPU note:** The Dockerfile uses CPU PyTorch. For GPU training, replace the pip install line in `backend/Dockerfile` with your CUDA wheel URL, e.g.:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

## Deployment (Vercel + server backend)

1. Push `frontend/` to a GitHub repo, connect to Vercel
2. Set `NEXT_PUBLIC_API_URL=https://your-backend-server:8000` in Vercel environment variables
3. Run the backend + Celery on your GPU server (Oracle Cloud, RunPod, etc.)

## Pipeline scripts used

| UI module | Script |
|---|---|
| Preprocessing → Simple | `preprocessing_pipeline/run_pipeline.py` |
| Preprocessing → Complete | `preprocessing_pipeline/complete_pipeline.py` |
| Training → PSNR | `main_train_swinir.py` |
| Training → GAN | `main_train_swinir_gan.py` |
| Inference | `main_test_swinir_config.py` |

## Metrics (from best_degradation.json)

| Metric | Weight | Direction |
|---|---|---|
| PSNR | 20% | higher = better |
| SSIM | 20% | higher = better |
| SAM | 15% | lower = better |
| FSIM | 15% | higher = better |
| UIQI | 10% | higher = better |
| RMSE | 10% | lower = better |
| IT-SSIM | 5% | higher = better |
| SRER | 5% | higher = better |
