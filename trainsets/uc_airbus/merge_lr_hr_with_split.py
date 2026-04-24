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
  "copy": {
    "recursive": false,
    "overwrite": false,
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
    train_ratio = float(cfg["split"]["train_ratio"])
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("split.train_ratio must be in (0, 1)")
    return cfg


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


def merge_and_split() -> None:
    cfg = _load_config()

    datasets = cfg["datasets"]
    output_train_root = cfg["output_train_root"]
    output_test_root = cfg["output_test_root"]
    train_ratio = float(cfg["split"]["train_ratio"])
    seed = int(cfg["split"]["seed"])
    recursive = bool(cfg["copy"]["recursive"])
    overwrite = bool(cfg["copy"]["overwrite"])
    image_exts = tuple(e.lower() for e in cfg["copy"]["image_exts"])

    out_train_hr = os.path.join(output_train_root, "hr")
    out_train_lr = os.path.join(output_train_root, "lr")
    out_test_hr = os.path.join(output_test_root, "hr")
    out_test_lr = os.path.join(output_test_root, "lr")
    os.makedirs(out_train_hr, exist_ok=True)
    os.makedirs(out_train_lr, exist_ok=True)
    os.makedirs(out_test_hr, exist_ok=True)
    os.makedirs(out_test_lr, exist_ok=True)

    dataset_to_pairs: Dict[str, List[Tuple[str, str, str, str]]] = {}
    # tuple: (dataset_name, stem, hr_path, lr_path)

    for d in datasets:
        name = str(d["name"]).strip()
        hr_dir = d["hr_dir"]
        lr_dir = d["lr_dir"]

        if not os.path.isdir(hr_dir):
            raise FileNotFoundError(f"HR dir not found for {name}: {hr_dir}")
        if not os.path.isdir(lr_dir):
            raise FileNotFoundError(f"LR dir not found for {name}: {lr_dir}")

        hr_files = _list_images(hr_dir, recursive=recursive, image_exts=image_exts)
        lr_files = _list_images(lr_dir, recursive=recursive, image_exts=image_exts)

        hr_map = _build_stem_map(hr_files)
        lr_map = _build_stem_map(lr_files)
        common_stems = sorted(set(hr_map) & set(lr_map))

        if not common_stems:
            logger.warning("No matched HR/LR stems found for dataset: %s", name)
            continue

        logger.info("Dataset %s: matched %d pairs", name, len(common_stems))
        dataset_pairs: List[Tuple[str, str, str, str]] = []
        for stem in common_stems:
            dataset_pairs.append((name, stem, hr_map[stem], lr_map[stem]))
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

    split_manifest = {
        "total_pairs": n_total,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "copied_train": copied_train,
        "copied_test": copied_test,
        "skipped": skipped,
        "train_ratio": train_ratio,
        "seed": seed,
        "output_train_root": output_train_root,
        "output_test_root": output_test_root,
        "split_mode": "stratified_per_dataset",
        "per_dataset_split": per_dataset_split,
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
