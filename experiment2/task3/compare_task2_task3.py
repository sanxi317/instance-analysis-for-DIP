"""Compare Task 2 restoration with Task 3 controls for Experiment 2.

Example:
  python experiment2/task3/compare_task2_task3.py
  python experiment2/task3/compare_task2_task3.py --task2-dir /path/to/task2

This script never modifies Task 1 or Task 2 results. All metrics are recomputed
from saved uint8 PNGs on the same GT, so PNG quantization is consistent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DIV2K_NAMES = ["div2k_15x15_reflect", "div2k_19x19_reflect"]
REAL_CONFIGS = [
    "Test_Size_15x15_sigma3",
    "Test_Size_35x35_sigma3",
    "Test_Sigma_35x35_sigma5",
    "Test_Sigma_35x35_sigma7",
]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def calculate(gt: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    if gt.shape != img.shape:
        raise ValueError(f"shape mismatch: GT={gt.shape}, result={img.shape}")
    g = gt.astype(np.float32) / 255.0
    p = img.astype(np.float32) / 255.0
    return (float(peak_signal_noise_ratio(g, p, data_range=1.0)),
            float(structural_similarity(g, p, data_range=1.0)))


def find_task2_image(task2_root: Path, expected_subdir: str,
                     name: str) -> Path | None:
    """根据 Task 2 实际目录定位复原结果。"""

    folder_map = {
        "div2k_15x15_reflect":
            Path("algorithm/picture_fixed/singles/div2k_15x15_reflect"),
        "div2k_19x19_reflect":
            Path("algorithm/picture_fixed/singles/div2k_19x19_reflect"),
        "Test_Size_15x15_sigma3":
            Path("realblur/realblur_fixed/singles/Test_Size_15x15_sigma3"),
        "Test_Size_35x35_sigma3":
            Path("realblur/realblur_fixed/singles/Test_Size_35x35_sigma3"),
        "Test_Sigma_35x35_sigma5":
            Path("realblur/realblur_fixed/singles/Test_Sigma_35x35_sigma5"),
        "Test_Sigma_35x35_sigma7":
            Path("realblur/realblur_fixed/singles/Test_Sigma_35x35_sigma7"),
    }

    if expected_subdir not in folder_map:
        print(f"[WARN] 未知实验配置: {expected_subdir}")
        return None

    path = task2_root / folder_map[expected_subdir] / name

    if path.is_file():
        return path

    return None

def optional_load(path: Path | None, description: str) -> np.ndarray | None:
    if path is None or not path.is_file():
        print(f"[MISS] {description}: {path or '(not found)'}")
        return None
    return read_gray(path)


def add_metric(rows: list[dict], dataset: str, method: str,
               img: np.ndarray, gt: np.ndarray, source: str,
               psf_guess: str = "-") -> bool:
    if img.shape != gt.shape:
        print(f"[SKIP] {dataset}/{method}: shape {img.shape} != GT {gt.shape}")
        return False
    p, s = calculate(gt, img)
    rows.append({"Dataset": dataset, "PSF_Guess": psf_guess,
                 "Method": method, "PSNR": p, "SSIM": s,
                 "Source": source})
    return True


def common_crop(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    h, w = img.shape
    left = min(max(0, x), max(0, w - size))
    top = min(max(0, y), max(0, h - size))
    return img[top:min(top + size, h), left:min(left + size, w)]


def save_panel(items: list[tuple[str, np.ndarray]], path: Path,
               crop_size: int = 256, local: bool = True) -> None:
    if not items:
        return
    h, w = items[0][1].shape
    x, y = max(0, (w - crop_size) // 2), max(0, (h - crop_size) // 2)
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, img) in zip(axes.flat, items):
        if img.shape != (h, w):
            print(f"[SKIP] panel element {label}: mismatched shape")
            continue
        view = common_crop(img, x, y, crop_size) if local else img
        ax.imshow(view, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_div2k(task1: Path, data: Path, task2: Path, task3: Path,
                  output: Path, rows: list[dict], crop: int) -> None:
    gt = optional_load(data / "div2k_gt.png", "DIV2K GT")
    if gt is None:
        return
    for name in DIV2K_NAMES:
        lq = optional_load(task1 / f"{name}.png", f"{name} LQ")
        if lq is None:
            continue
        if lq.shape != gt.shape:
            print(f"[SKIP] {name}: GT/LQ shapes mismatch")
            continue
        items = [("GT", gt), ("LQ", lq)]
        add_metric(rows, name, "LQ", lq, gt, str(task1 / f"{name}.png"))

        # These exact names are generated by the provided Task 2 DIV2K script.
        representative_d0 = 30 if "15x15" in name else 20
        for file_name, method in [
            ("Full_Inverse.png", "Full Inverse"),
            (f"Inv_D0_{representative_d0}.png", f"Limited Inverse D0={representative_d0}"),
            ("Wiener_Filter.png", "Wiener"),
            ("CLS_Filter.png", "CLS"),
        ]:
            p = find_task2_image(task2, name, file_name)
            img = optional_load(p, f"Task2 DIV2K {name}/{file_name}")
            if img is not None and add_metric(rows, name, method, img, gt, str(p)):
                items.append((method, img))

        downup_path = task3 / "bicubic" / name / "bicubic_down_up.png"
        downup = optional_load(downup_path, f"Task3 DIV2K {name} Down-Up")
        if downup is not None and add_metric(rows, name, "Bicubic Down-Up", downup, gt, str(downup_path)):
            items.append(("Bicubic Down-Up", downup))

        save_panel(items, output / f"{name}_crop.png", crop, local=True)
        save_panel(items, output / f"{name}_full.png", crop, local=False)


def process_realblur(task1: Path, task2: Path, task3: Path, output: Path,
                     rows: list[dict], crop: int) -> None:
    gt = optional_load(task1 / "realblur_gt_gray.png", "RealBlur GT")
    lq = optional_load(task1 / "realblur_blur_gray.png", "RealBlur LQ")
    if gt is None or lq is None or gt.shape != lq.shape:
        print("[SKIP] RealBlur GT/LQ missing or mismatched")
        return
    add_metric(rows, "RealBlur", "LQ", lq, gt, str(task1 / "realblur_blur_gray.png"))

    shared = [("GT", gt), ("LQ", lq)]
    downup_path = task3 / "bicubic" / "RealBlur" / "bicubic_down_up.png"
    downup = optional_load(downup_path, "Task3 RealBlur Down-Up")
    if downup is not None and add_metric(rows, "RealBlur", "Bicubic Down-Up", downup, gt, str(downup_path)):
        shared.append(("Bicubic Down-Up", downup))

    for h in [8, 13, 20]:
        p = task3 / "nlm" / f"nlm_h{h}.png"
        img = optional_load(p, f"Task3 RealBlur NLM h={h}")
        if img is not None and add_metric(rows, "RealBlur", f"NLM h={h}", img, gt, str(p)):
            shared.append((f"NLM h={h}", img))

    # Task 2 RealBlur has FOUR guessed PSFs, each with its own Wiener and CLS.
    for config in REAL_CONFIGS:
        items = shared.copy()
        for method, filename in [("Full Inverse", "Full_Inverse.png"),
                                 ("Limited Inverse D0=15", "Inv_D0_15.png"),
                                 ("Wiener", "Wiener.png"),
                                 ("CLS", "CLS.png")]:
            p = find_task2_image(task2, config, filename)
            img = optional_load(p, f"Task2 RealBlur {config}/{filename}")
            if img is not None and add_metric(rows, "RealBlur", method, img, gt, str(p), config):
                items.append((f"{method}\n{config}", img))
        save_panel(items, output / f"realblur_{config}_crop.png", crop, local=True)
        # One full image comparison for the canonical 15x15 sigma=3 configuration
        if config == REAL_CONFIGS[0]:
            save_panel(items, output / "realblur_primary_full.png", crop, local=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task2-dir", type=Path,
                        help="Task2 results root (default: experiment2/task2)")
    parser.add_argument("--task1-dir", type=Path,
                        help="Task1 output folder (default: experiment2/task1/output)")
    parser.add_argument("--out-dir", type=Path,
                        help="Comparison output folder (default: experiment2/task3/output/task2_comparison)")
    parser.add_argument("--crop", type=int, default=256,
                        help="Centered crop side length (default: 256)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    task1 = args.task1_dir or root / "task1" / "output"
    task2 = args.task2_dir or root / "task2"
    task3 = root / "task3" / "output"
    output = args.out_dir or task3 / "task2_comparison"
    if not task2.is_dir():
        print(f"[WARN] Task2 folder not found: {task2}")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    process_div2k(task1, root / "data", task2, task3, output, rows, args.crop)
    process_realblur(task1, task2, task3, output, rows, args.crop)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output / "task2_task3_metrics_recomputed.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"\nSaved {len(df)} metric rows: {csv_path}")
        for dataset, part in df.groupby("Dataset", sort=False):
            print("\n===", dataset, "===")
            print(part[["PSF_Guess", "Method", "PSNR", "SSIM"]].to_string(index=False,
                       float_format=lambda x: f"{x:.5f}"))
    else:
        print("[WARN] No metric rows generated. Check source paths.")
    print(f"\nFigures in: {output}")
    print("Note: bicubic direct x2 is excluded from PSNR/SSIM (size mismatch).")
    print("Down-Up contains additional downsampling; do not interpret as deblurring.")
    print("Metrics are recomputed from PNGs; minor differences from Task2 CSVs are normal.")


if __name__ == "__main__":
    main()
