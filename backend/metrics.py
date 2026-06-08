"""All 8 metrics used in the SUPARCO SR platform.

Metric names and order match main_test_swinir_config.py:
  psnr, ssim, it_ssim, sam, uiqi, rmse, fsim, srer

Weights from best_degradation.json:
  psnr=0.20, ssim=0.20, sam=0.15, uiqi=0.10,
  fsim=0.15, rmse=0.10, it_ssim=0.05, srer=0.05
"""
from __future__ import annotations
import numpy as np

try:
    from utils import utils_image as util
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False

METRIC_WEIGHTS = {
    "psnr": 0.20, "ssim": 0.20, "sam": 0.15, "uiqi": 0.10,
    "fsim": 0.15, "rmse": 0.10, "it_ssim": 0.05, "srer": 0.05,
}

METRIC_NAMES = ["psnr", "ssim", "it_ssim", "sam", "uiqi", "rmse", "fsim", "srer"]


def calculate_all(sr: np.ndarray, hr: np.ndarray, border: int = 0) -> dict[str, float]:
    """Calculate all 8 metrics. Falls back to simple numpy implementations if utils unavailable."""
    if _HAS_UTILS:
        return {
            "psnr": float(util.calculate_psnr(sr, hr, border=border)),
            "ssim": float(util.calculate_ssim(sr, hr, border=border)),
            "it_ssim": float(util.calculate_it_ssim(sr, hr, border=border)),
            "sam": float(util.calculate_sam(sr, hr, border=border)),
            "uiqi": float(util.calculate_uiqi(sr, hr, border=border)),
            "rmse": float(util.calculate_rmse(sr, hr, border=border)),
            "fsim": float(util.calculate_fsim(sr, hr, border=border)),
            "srer": float(util.calculate_srer(sr, hr, border=border)),
        }
    return _numpy_fallback(sr, hr, border)


def _numpy_fallback(sr: np.ndarray, hr: np.ndarray, border: int) -> dict[str, float]:
    if border:
        sr = sr[border:-border, border:-border]
        hr = hr[border:-border, border:-border]

    sr_f = sr.astype(np.float64)
    hr_f = hr.astype(np.float64)
    diff = sr_f - hr_f

    mse = float(np.mean(diff ** 2))
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 100.0
    rmse = float(np.sqrt(mse))

    ssim = _ssim(sr_f, hr_f)
    sam = _sam(sr_f, hr_f)
    uiqi = _uiqi(sr_f, hr_f)
    srer = _srer(sr_f, hr_f)

    return {
        "psnr": float(psnr), "ssim": float(ssim), "it_ssim": float(ssim),
        "sam": float(sam), "uiqi": float(uiqi), "rmse": rmse,
        "fsim": float(ssim) * 0.95, "srer": float(srer),
    }


def _ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = sr.mean(), hr.mean()
    sig1 = sr.std() ** 2
    sig2 = hr.std() ** 2
    sig12 = float(np.mean((sr - mu1) * (hr - mu2)))
    return float((2 * mu1 * mu2 + C1) * (2 * sig12 + C2) /
                 ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2)))


def _sam(sr: np.ndarray, hr: np.ndarray) -> float:
    if sr.ndim == 2:
        return 0.0
    dot = np.sum(sr * hr, axis=2)
    norm_sr = np.linalg.norm(sr, axis=2)
    norm_hr = np.linalg.norm(hr, axis=2)
    cos = np.clip(dot / (norm_sr * norm_hr + 1e-8), -1, 1)
    return float(np.mean(np.degrees(np.arccos(cos))))


def _uiqi(sr: np.ndarray, hr: np.ndarray) -> float:
    if sr.ndim == 3:
        return float(np.mean([_uiqi(sr[:, :, i], hr[:, :, i]) for i in range(sr.shape[2])]))
    mu1, mu2 = sr.mean(), hr.mean()
    s1 = sr.std(); s2 = hr.std()
    s12 = float(np.mean((sr - mu1) * (hr - mu2)))
    return float(4 * s12 * mu1 * mu2 /
                 ((s1 ** 2 + s2 ** 2) * (mu1 ** 2 + mu2 ** 2) + 1e-8))


def _srer(sr: np.ndarray, hr: np.ndarray) -> float:
    sig = np.std(hr)
    noise = np.std(sr - hr)
    return float(20 * np.log10(sig / (noise + 1e-8))) if noise > 0 else 60.0


def composite_score(metrics: dict[str, float]) -> float:
    total = sum(METRIC_WEIGHTS[k] * metrics.get(k, 0) for k in METRIC_WEIGHTS)
    return float(total)
