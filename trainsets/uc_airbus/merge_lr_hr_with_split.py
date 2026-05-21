"""
merge_lr_hr_with_split.py

Merge multiple paired HR/LR datasets into one unified dataset with train/test split.
Each copied file is prefixed with a dataset identifier so provenance is preserved.

Run:
  python merge_lr_hr_with_split.py
"""

import json
import os
import random
import shutil
import logging
from typing import Dict, List, Tuple

import cv2


# ===========================================================================
# JSON CONFIG (edit values below)
# ===========================================================================
CONFIG_JSON = r"""
{
  "datasets": [
    {
      "name": "airbus",
      "hr_dir": "E:/FAST/SUPARCO/KAIR/pleaides_preprocessing/Lahore_3/hr",
      "lr_dir": "E:/FAST/SUPARCO/KAIR/pleaides_preprocessing/Lahore_3/lr"
    },
    {
      "name": "ucmerced",
      "hr_dir": "E:/FAST/SUPARCO/KAIR/trainsets/UCMerced_LandUse/hr_all",
      "lr_dir": "E:/FAST/SUPARCO/KAIR/trainsets/UCMerced_LandUse/lr_esrgan"
    }
  ],
    "output_train_root": "E:/FAST/SUPARCO/KAIR/trainsets/uc_airbus",
    "output_test_root": "E:/FAST/SUPARCO/KAIR/testsets/uc_airbus",
  "split": {
    "train_ratio": 0.8,
    "seed": 42
  },
    "pair_filter": {
        "scale_factor": 2,
          "exact_scale_only": true,
        "min_hr_size": [256, 256],
        "min_lr_size": [128, 128]
    },
  "copy": {
    "recursive": false,
    "overwrite": false,
        "clean_output_dirs": true,
    "image_exts": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
  }
}
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config() -> Dict:
    cfg = json.loads(CONFIG_JSON)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict) -> None:
    required_top = {"datasets", "output_train_root", "output_test_root", "split", "pair_filter", "copy"}
    missing = required_top - set(cfg.keys())
    if missing:
        raise KeyError(f"Missing required config keys: {sorted(missing)}")

    datasets = cfg["datasets"]
    if not isinstance(datasets, list) or len(datasets) == 0:
        raise ValueError("datasets must be a non-empty list")

    seen_names = set()
    for d in datasets:
        for key in ["name", "hr_dir", "lr_dir"]:
            if key not in d:
                raise KeyError(f"Dataset entry missing key: {key}")
        name = str(d["name"]).strip()
        if not name:
            raise ValueError("Dataset name must be non-empty")
        if name in seen_names:
            raise ValueError(f"Duplicate dataset name found: {name}")
        seen_names.add(name)

    train_ratio = float(cfg["split"]["train_ratio"])
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("split.train_ratio must be in (0, 1)")

    scale_factor = int(cfg["pair_filter"]["scale_factor"])
    if scale_factor <= 0:
        raise ValueError("pair_filter.scale_factor must be a positive integer")
    exact_scale_only = bool(cfg["pair_filter"].get("exact_scale_only", True))
    if not exact_scale_only:
        raise ValueError("pair_filter.exact_scale_only must be true for strict dataset creation")

    for k in ["min_hr_size", "min_lr_size"]:
        v = cfg["pair_filter"].get(k)
        if v is not None:
            if (not isinstance(v, list)) or len(v) != 2:
                raise ValueError(f"pair_filter.{k} must be null or [height, width]")
            if int(v[0]) <= 0 or int(v[1]) <= 0:
                raise ValueError(f"pair_filter.{k} values must be positive")

    image_exts = cfg["copy"]["image_exts"]
    if not isinstance(image_exts, list) or len(image_exts) == 0:
        raise ValueError("copy.image_exts must be a non-empty list")
    for ext in image_exts:
        if not isinstance(ext, str) or not ext.startswith("."):
            raise ValueError(f"Invalid extension in copy.image_exts: {ext}")


def _list_images(root: str, recursive: bool, image_exts: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    if recursive:
        for base, _, files in os.walk(root):
            for f in files:
                if f.lower().endswith(image_exts):
                    paths.append(os.path.join(base, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and f.lower().endswith(image_exts):
                paths.append(p)
    return sorted(paths)


def _build_stem_map(paths: List[str]) -> Dict[str, str]:
    stem_map: Dict[str, str] = {}
    duplicates = 0
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in stem_map:
            duplicates += 1
            continue
        stem_map[stem] = p
    if duplicates:
        logger.warning("Found %d duplicate stems in a folder; extras were ignored.", duplicates)
    return stem_map


def _validate_output_pair_dirs(
    hr_dir: str,
    lr_dir: str,
    scale_factor: int,
    split_name: str,
    min_hr_h: int,
    min_hr_w: int,
    min_lr_h: int,
    min_lr_w: int,
) -> Dict[str, int]:
    """Validate generated output folders contain stem-aligned HR/LR pairs with correct scale."""
    hr_files = _list_images(hr_dir, recursive=False, image_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    lr_files = _list_images(lr_dir, recursive=False, image_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    hr_map = _build_stem_map(hr_files)
    lr_map = _build_stem_map(lr_files)

    only_hr = sorted(set(hr_map) - set(lr_map))
    only_lr = sorted(set(lr_map) - set(hr_map))
    if only_hr or only_lr:
        raise RuntimeError(
            f"Output {split_name} mismatch: only_hr={len(only_hr)}, only_lr={len(only_lr)}"
        )

    bad_scale = 0
    bad_min_size = 0
    unreadable = 0
    for stem in sorted(set(hr_map) & set(lr_map)):
        hr_path = hr_map[stem]
        lr_path = lr_map[stem]
        try:
            if not _is_scale_match(hr_path, lr_path, scale_factor):
                bad_scale += 1
            if not _is_min_size_match(hr_path, lr_path, min_hr_h, min_hr_w, min_lr_h, min_lr_w):
                bad_min_size += 1
        except Exception:
            unreadable += 1

    if bad_scale or bad_min_size or unreadable:
        raise RuntimeError(
            f"Output {split_name} failed validation: bad_scale={bad_scale}, bad_min_size={bad_min_size}, unreadable={unreadable}"
        )

    return {
        "pairs": len(hr_map),
        "bad_scale": bad_scale,
        "bad_min_size": bad_min_size,
        "unreadable": unreadable,
    }


def _clear_dir_images(root: str, image_exts: Tuple[str, ...]) -> int:
    removed = 0
    if not os.path.isdir(root):
        return removed
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isfile(p) and name.lower().endswith(image_exts):
            os.remove(p)
            removed += 1
    return removed


def _copy_pair(
    hr_src: str,
    lr_src: str,
    dst_hr_dir: str,
    dst_lr_dir: str,
    out_stem: str,
    overwrite: bool,
) -> bool:
    hr_ext = os.path.splitext(hr_src)[1]
    lr_ext = os.path.splitext(lr_src)[1]
    hr_dst = os.path.join(dst_hr_dir, out_stem + hr_ext)
    lr_dst = os.path.join(dst_lr_dir, out_stem + lr_ext)

    if (not overwrite) and (os.path.exists(hr_dst) or os.path.exists(lr_dst)):
        return False

    shutil.copy2(hr_src, hr_dst)
    shutil.copy2(lr_src, lr_dst)
    return True


def _read_hw(path: str) -> Tuple[int, int]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    return h, w


def _is_scale_match(hr_path: str, lr_path: str, scale_factor: int) -> bool:
    h_hr, w_hr = _read_hw(hr_path)
    h_lr, w_lr = _read_hw(lr_path)
    # Strict exact-ratio check with no tolerance.
    if h_lr <= 0 or w_lr <= 0:
        return False
    if (h_hr % h_lr) != 0 or (w_hr % w_lr) != 0:
        return False
    return (h_hr // h_lr) == scale_factor and (w_hr // w_lr) == scale_factor


def _is_min_size_match(
    hr_path: str,
    lr_path: str,
    min_hr_h: int,
    min_hr_w: int,
    min_lr_h: int,
    min_lr_w: int,
) -> bool:
    h_hr, w_hr = _read_hw(hr_path)
    h_lr, w_lr = _read_hw(lr_path)
    return (h_hr >= min_hr_h and w_hr >= min_hr_w and h_lr >= min_lr_h and w_lr >= min_lr_w)


def merge_and_split() -> None:
    cfg = _load_config()

    datasets = cfg["datasets"]
    output_train_root = cfg["output_train_root"]
    output_test_root = cfg["output_test_root"]
    train_ratio = float(cfg["split"]["train_ratio"])
    seed = int(cfg["split"]["seed"])
    scale_factor = int(cfg["pair_filter"]["scale_factor"])
    exact_scale_only = bool(cfg["pair_filter"].get("exact_scale_only", True))
    min_hr_h, min_hr_w = [int(x) for x in cfg["pair_filter"].get("min_hr_size", [1, 1])]
    min_lr_h, min_lr_w = [int(x) for x in cfg["pair_filter"].get("min_lr_size", [1, 1])]
    recursive = bool(cfg["copy"]["recursive"])
    overwrite = bool(cfg["copy"]["overwrite"])
    clean_output_dirs = bool(cfg["copy"].get("clean_output_dirs", False))
    image_exts = tuple(e.lower() for e in cfg["copy"]["image_exts"])

    out_train_hr = os.path.join(output_train_root, "hr")
    out_train_lr = os.path.join(output_train_root, "lr")
    out_test_hr = os.path.join(output_test_root, "hr")
    out_test_lr = os.path.join(output_test_root, "lr")
    os.makedirs(out_train_hr, exist_ok=True)
    os.makedirs(out_train_lr, exist_ok=True)
    os.makedirs(out_test_hr, exist_ok=True)
    os.makedirs(out_test_lr, exist_ok=True)

    if clean_output_dirs:
        removed = 0
        removed += _clear_dir_images(out_train_hr, image_exts)
        removed += _clear_dir_images(out_train_lr, image_exts)
        removed += _clear_dir_images(out_test_hr, image_exts)
        removed += _clear_dir_images(out_test_lr, image_exts)
        logger.info("Cleaned %d existing output image(s) before copy.", removed)

    dataset_to_pairs: Dict[str, List[Tuple[str, str, str, str]]] = {}
    # tuple: (dataset_name, stem, hr_path, lr_path)
    total_discarded_scale = 0
    total_discarded_small = 0

    for d in datasets:
        name = str(d["name"]).strip()
        hr_dir = d["hr_dir"]
        lr_dir = d["lr_dir"]

        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR dir not found for {name}: {hr_dir}")
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR dir not found for {name}: {lr_dir}")

        if os.path.abspath(hr_dir) == os.path.abspath(lr_dir):
            raise ValueError(f"HR and LR dirs must differ for dataset {name}")

        hr_files = _list_images(hr_dir, recursive=recursive, image_exts=image_exts)
        lr_files = _list_images(lr_dir, recursive=recursive, image_exts=image_exts)

        if not hr_files:
            logger.warning("No HR images found in dataset %s (%s)", name, hr_dir)
        if not lr_files:
            logger.warning("No LR images found in dataset %s (%s)", name, lr_dir)

        hr_map = _build_stem_map(hr_files)
        lr_map = _build_stem_map(lr_files)
        common_stems = sorted(set(hr_map) & set(lr_map))

        if not common_stems:
            logger.warning("No matched HR/LR stems found for dataset: %s", name)
            continue

        logger.info("Dataset %s: matched by filename %d pairs", name, len(common_stems))
        dataset_pairs: List[Tuple[str, str, str, str]] = []
        discarded_scale = 0
        discarded_small = 0
        for stem in common_stems:
            hr_path = hr_map[stem]
            lr_path = lr_map[stem]
            try:
                if not _is_scale_match(hr_path, lr_path, scale_factor):
                    discarded_scale += 1
                    continue

                if not _is_min_size_match(hr_path, lr_path, min_hr_h, min_hr_w, min_lr_h, min_lr_w):
                    discarded_small += 1
                    continue

                dataset_pairs.append((name, stem, hr_path, lr_path))
            except Exception as e:
                discarded_scale += 1
                logger.warning("Discarded unreadable/mismatch pair (%s, %s): %s", hr_path, lr_path, e)

        total_discarded_scale += discarded_scale
        total_discarded_small += discarded_small
        logger.info(
            "Dataset %s: kept %d pairs after filters (discarded scale=%d, small=%d, sf=%d, min_hr=%dx%d, min_lr=%dx%d)",
            name,
            len(dataset_pairs),
            discarded_scale,
            discarded_small,
            scale_factor,
            min_hr_h,
            min_hr_w,
            min_lr_h,
            min_lr_w,
        )

        if not dataset_pairs:
            logger.warning("Dataset %s has no valid pairs after scale filtering.", name)
            continue
        dataset_to_pairs[name] = dataset_pairs

    if not dataset_to_pairs:
        raise RuntimeError("No paired samples found across all datasets.")

    rng = random.Random(seed)
    train_pairs: List[Tuple[str, str, str, str]] = []
    test_pairs: List[Tuple[str, str, str, str]] = []
    per_dataset_split: Dict[str, Dict[str, int]] = {}

    # Stratified split: apply train_ratio within each dataset independently.
    for dataset_name, pairs in dataset_to_pairs.items():
        local_pairs = list(pairs)
        rng.shuffle(local_pairs)
        n_local_total = len(local_pairs)
        n_local_train = int(n_local_total * train_ratio)
        local_train = local_pairs[:n_local_train]
        local_test = local_pairs[n_local_train:]

        train_pairs.extend(local_train)
        test_pairs.extend(local_test)

        per_dataset_split[dataset_name] = {
            "total_pairs": n_local_total,
            "train_pairs": len(local_train),
            "test_pairs": len(local_test),
        }

    # Shuffle final train/test buckets to avoid grouped ordering by dataset.
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    n_total = len(train_pairs) + len(test_pairs)

    logger.info("Total pairs: %d | Train: %d | Test: %d", n_total, len(train_pairs), len(test_pairs))

    copied_train = 0
    copied_test = 0
    skipped = 0

    for dataset_name, stem, hr_src, lr_src in train_pairs:
        out_stem = f"{dataset_name}_{stem}"
        ok = _copy_pair(hr_src, lr_src, out_train_hr, out_train_lr, out_stem, overwrite)
        if ok:
            copied_train += 1
        else:
            skipped += 1

    for dataset_name, stem, hr_src, lr_src in test_pairs:
        out_stem = f"{dataset_name}_{stem}"
        ok = _copy_pair(hr_src, lr_src, out_test_hr, out_test_lr, out_stem, overwrite)
        if ok:
            copied_test += 1
        else:
            skipped += 1

    # Robust post-checks to ensure generated dataset is consistent.
    train_validation = _validate_output_pair_dirs(
        out_train_hr, out_train_lr, scale_factor, "train", min_hr_h, min_hr_w, min_lr_h, min_lr_w
    )
    test_validation = _validate_output_pair_dirs(
        out_test_hr, out_test_lr, scale_factor, "test", min_hr_h, min_hr_w, min_lr_h, min_lr_w
    )

    split_manifest = {
        "total_pairs": n_total,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "copied_train": copied_train,
        "copied_test": copied_test,
        "skipped": skipped,
        "discarded_scale_mismatch": total_discarded_scale,
        "discarded_too_small": total_discarded_small,
        "scale_factor": scale_factor,
        "exact_scale_only": exact_scale_only,
        "min_hr_size": [min_hr_h, min_hr_w],
        "min_lr_size": [min_lr_h, min_lr_w],
        "train_ratio": train_ratio,
        "seed": seed,
        "output_train_root": output_train_root,
        "output_test_root": output_test_root,
        "split_mode": "stratified_per_dataset",
        "per_dataset_split": per_dataset_split,
        "output_validation": {
            "train": train_validation,
            "test": test_validation,
        },
        "datasets": datasets,
    }

    manifest_path = os.path.join(output_train_root, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    test_manifest_path = os.path.join(output_test_root, "split_manifest.json")
    with open(test_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info("Done.")
    logger.info("Train copied: %d | Test copied: %d | Skipped: %d", copied_train, copied_test, skipped)
    logger.info("Train output root: %s", output_train_root)
    logger.info("Test output root: %s", output_test_root)
    logger.info("Train manifest: %s", manifest_path)
    logger.info("Test manifest: %s", test_manifest_path)


if __name__ == "__main__":
    merge_and_split()
