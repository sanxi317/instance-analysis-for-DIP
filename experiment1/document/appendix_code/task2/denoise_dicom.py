#!/usr/bin/env python3
"""任务二：直接基于配对 DICOM 的传统 CT 去噪与参数比较。"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
import SimpleITK as sitk
import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


TASK_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = TASK_DIR.parent
RESULTS_DIR = TASK_DIR / "results"
FIGURES_DIR = TASK_DIR / "figures"
IMAGES_DIR = TASK_DIR / "images"

FDCT_PATH = (
    EXPERIMENT_DIR
    / "full_3mm/L067/full_3mm"
    / "L067_FD_3_1.CT.0002.0001.2015.12.22.18.12.07.5968.358090401.IMA"
)
LDCT_PATH = (
    EXPERIMENT_DIR
    / "quarter_3mm/L067/quarter_3mm"
    / "L067_QD_3_1.CT.0004.0001.2015.12.22.18.12.56.428910.358293547.IMA"
)

# DICOM 中提供的肺窗预设：窗位 -600 HU，窗宽 1500 HU。
WINDOW_CENTER = -600.0
WINDOW_WIDTH = 1500.0
WINDOW_MIN = WINDOW_CENTER - WINDOW_WIDTH / 2.0
WINDOW_MAX = WINDOW_CENTER + WINDOW_WIDTH / 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dicom_hu(path: Path) -> tuple[np.ndarray, sitk.Image]:
    """读取单张 DICOM，并返回二维 HU 数组。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 DICOM 文件：{path}")
    image = sitk.ReadImage(str(path))
    array = np.squeeze(sitk.GetArrayFromImage(image)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"期望二维切片，实际数组形状为 {array.shape}")
    return array, image


def metadata(image: sitk.Image, key: str, default: str = "N/A") -> str:
    return image.GetMetaData(key).strip() if image.HasMetaDataKey(key) else default


def window_to_uint8(image_hu: np.ndarray) -> np.ndarray:
    """使用固定肺窗截断，并线性归一化为 0~255 灰度。"""
    clipped = np.clip(image_hu, WINDOW_MIN, WINDOW_MAX)
    normalized = (clipped - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return np.rint(normalized * 255.0).astype(np.uint8)


def evaluate(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """以同窗宽归一化的 FDCT 为参考，计算全图 PSNR 与 SSIM。"""
    psnr_value = peak_signal_noise_ratio(reference, candidate, data_range=255)
    ssim_value = structural_similarity(reference, candidate, data_range=255)
    return float(psnr_value), float(ssim_value)


def adaptive_median_filter(image: np.ndarray, max_window: int = 7) -> np.ndarray:
    """真正的自适应中值滤波：窗口从 3×3 逐级增大到 max_window。"""
    if max_window < 3 or max_window % 2 == 0:
        raise ValueError("max_window 必须是大于等于 3 的奇数")

    source = image.astype(np.uint8)
    output = source.copy()
    unresolved = np.ones(source.shape, dtype=bool)
    last_median = source.copy()

    for window in range(3, max_window + 1, 2):
        kernel = np.ones((window, window), dtype=np.uint8)
        local_min = cv2.erode(source, kernel, borderType=cv2.BORDER_REFLECT)
        local_max = cv2.dilate(source, kernel, borderType=cv2.BORDER_REFLECT)
        local_median = cv2.medianBlur(source, window)
        last_median = local_median

        stage_a = (local_median > local_min) & (local_median < local_max)
        newly_resolved = unresolved & stage_a
        stage_b = (source > local_min) & (source < local_max)

        keep_original = newly_resolved & stage_b
        use_median = newly_resolved & ~stage_b
        output[keep_original] = source[keep_original]
        output[use_median] = local_median[use_median]
        unresolved &= ~stage_a

    output[unresolved] = last_median[unresolved]
    return output


def wavelet_denoise(
    image: np.ndarray,
    threshold: float,
    wavelet: str = "db1",
    level: int = 2,
) -> np.ndarray:
    """二维离散小波软阈值去噪。"""
    coeffs = pywt.wavedec2(image.astype(np.float32), wavelet, level=level)
    thresholded = [coeffs[0]]
    for horizontal, vertical, diagonal in coeffs[1:]:
        thresholded.append(
            tuple(
                pywt.threshold(detail, threshold, mode="soft")
                for detail in (horizontal, vertical, diagonal)
            )
        )
    reconstructed = pywt.waverec2(thresholded, wavelet)
    reconstructed = reconstructed[: image.shape[0], : image.shape[1]]
    return np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)


def save_grayscale(path: Path, image: np.ndarray) -> None:
    """保存纯 512×512 灰度图，不添加标题、坐标轴或白边。"""
    if not cv2.imwrite(str(path), image):
        raise OSError(f"图片保存失败：{path}")


def save_figure(fig: plt.Figure, basename: str) -> None:
    fig.savefig(
        FIGURES_DIR / f"{basename}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        FIGURES_DIR / f"{basename}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def parameter_text(parameters: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in parameters.items())


def main() -> None:
    for directory in (RESULTS_DIR, FIGURES_DIR, IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    fdct_hu, fdct_dicom = read_dicom_hu(FDCT_PATH)
    ldct_hu, ldct_dicom = read_dicom_hu(LDCT_PATH)

    if fdct_hu.shape != ldct_hu.shape:
        raise ValueError(f"配对图像尺寸不一致：FDCT={fdct_hu.shape}, LDCT={ldct_hu.shape}")

    pairing_tags = {
        "Patient ID": "0010|0020",
        "Image Position Patient": "0020|0032",
        "Image Orientation Patient": "0020|0037",
        "Pixel Spacing": "0028|0030",
        "Slice Thickness": "0018|0050",
        "Convolution Kernel": "0018|1210",
        "Instance Number": "0020|0013",
    }
    pairing_metadata: dict[str, str] = {}
    for label, tag in pairing_tags.items():
        fdct_value = metadata(fdct_dicom, tag)
        ldct_value = metadata(ldct_dicom, tag)
        if fdct_value != ldct_value:
            raise ValueError(
                f"配对图像的 {label} 不一致：FDCT={fdct_value}, LDCT={ldct_value}"
            )
        pairing_metadata[label] = fdct_value
    fdct_position = pairing_metadata["Image Position Patient"]

    fdct = window_to_uint8(fdct_hu)
    ldct = window_to_uint8(ldct_hu)

    search_rows: list[dict[str, object]] = []
    best: dict[str, dict[str, object]] = {}

    def record(
        method: str,
        parameters: dict[str, object],
        candidate: np.ndarray,
        elapsed_seconds: float,
    ) -> None:
        psnr_value, ssim_value = evaluate(fdct, candidate)
        row = {
            "method": method,
            "parameters": parameter_text(parameters),
            "psnr_db": psnr_value,
            "ssim": ssim_value,
            "runtime_seconds": elapsed_seconds,
        }
        search_rows.append(row)
        previous = best.get(method)
        # 医学图像以结构保持为主：SSIM 优先，PSNR 作为同分决胜项。
        if previous is None or (ssim_value, psnr_value) > (
            float(previous["ssim"]),
            float(previous["psnr_db"]),
        ):
            best[method] = {**row, "image": candidate.copy(), "parameter_dict": parameters}

    for kernel_size in (3, 5, 7):
        start = time.perf_counter()
        candidate = cv2.blur(ldct, (kernel_size, kernel_size))
        record("Mean", {"kernel": kernel_size}, candidate, time.perf_counter() - start)

    for kernel_size, sigma in (
        (3, 0.5),
        (3, 1.0),
        (5, 0.5),
        (5, 1.0),
        (5, 1.5),
        (7, 1.0),
        (7, 1.5),
    ):
        start = time.perf_counter()
        candidate = cv2.GaussianBlur(
            ldct,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        record(
            "Gaussian",
            {"kernel": kernel_size, "sigma": sigma},
            candidate,
            time.perf_counter() - start,
        )

    for max_window in (3, 5, 7):
        start = time.perf_counter()
        candidate = adaptive_median_filter(ldct, max_window=max_window)
        record(
            "AdaptiveMedian",
            {"max_window": max_window},
            candidate,
            time.perf_counter() - start,
        )

    for h_value in (1, 2, 3, 4, 5, 7, 10, 15):
        start = time.perf_counter()
        candidate = cv2.fastNlMeansDenoising(
            ldct,
            None,
            h=float(h_value),
            templateWindowSize=7,
            searchWindowSize=21,
        )
        record("NLM", {"h": h_value}, candidate, time.perf_counter() - start)

    for threshold in (1, 2, 3, 4, 5, 8, 10, 15, 20):
        start = time.perf_counter()
        candidate = wavelet_denoise(ldct, threshold=float(threshold))
        record(
            "Wavelet",
            {"wavelet": "db1", "level": 2, "threshold": threshold},
            candidate,
            time.perf_counter() - start,
        )

    baseline_psnr, baseline_ssim = evaluate(fdct, ldct)
    method_order = ["Mean", "Gaussian", "AdaptiveMedian", "NLM", "Wavelet"]
    final_rows: list[dict[str, object]] = [
        {
            "method": "LDCT",
            "parameters": "none",
            "psnr_db": baseline_psnr,
            "ssim": baseline_ssim,
            "delta_psnr_db": 0.0,
            "delta_ssim": 0.0,
        }
    ]
    for method in method_order:
        chosen = best[method]
        final_rows.append(
            {
                "method": method,
                "parameters": chosen["parameters"],
                "psnr_db": chosen["psnr_db"],
                "ssim": chosen["ssim"],
                "delta_psnr_db": float(chosen["psnr_db"]) - baseline_psnr,
                "delta_ssim": float(chosen["ssim"]) - baseline_ssim,
            }
        )

    best_overall = max(
        (row for row in final_rows if row["method"] != "LDCT"),
        key=lambda row: (float(row["ssim"]), float(row["psnr_db"])),
    )

    with (RESULTS_DIR / "parameter_search.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    save_grayscale(IMAGES_DIR / "fdct_ground_truth.png", fdct)
    save_grayscale(IMAGES_DIR / "ldct_input.png", ldct)
    output_names = {
        "Mean": "mean.png",
        "Gaussian": "gaussian.png",
        "AdaptiveMedian": "adaptive_median.png",
        "NLM": "nlm.png",
        "Wavelet": "wavelet.png",
    }
    for method, filename in output_names.items():
        save_grayscale(IMAGES_DIR / filename, np.asarray(best[method]["image"]))

    panels = [
        ("FDCT ground truth", fdct),
        ("LDCT input", ldct),
        *[(method, np.asarray(best[method]["image"])) for method in method_order],
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), constrained_layout=True)
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    summary_axis = axes.flat[-1]
    summary_axis.axis("off")
    summary_axis.text(
        0.06,
        0.82,
        "Selected result",
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.06,
        0.66,
        f"Method: {best_overall['method']}\n"
        f"PSNR: {float(best_overall['psnr_db']):.3f} dB\n"
        f"SSIM: {float(best_overall['ssim']):.5f}",
        fontsize=9,
        linespacing=1.45,
        va="top",
        transform=summary_axis.transAxes,
    )
    summary_axis.text(
        0.06,
        0.30,
        "Selection rule:\nmaximize SSIM,\nthen PSNR",
        fontsize=8,
        color="#555555",
        linespacing=1.4,
        va="top",
        transform=summary_axis.transAxes,
    )
    fig.suptitle("DICOM-based denoising comparison (lung window)", fontsize=14)
    save_figure(fig, "denoising_comparison")

    # 肺内血管区域局部放大，所有方法使用相同 ROI 与显示尺度。
    y0, y1, x0, x1 = 150, 370, 45, 265
    detail_panels = [
        ("FDCT", fdct[y0:y1, x0:x1]),
        ("LDCT", ldct[y0:y1, x0:x1]),
        ("NLM", np.asarray(best["NLM"]["image"])[y0:y1, x0:x1]),
        ("Wavelet", np.asarray(best["Wavelet"]["image"])[y0:y1, x0:x1]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), constrained_layout=True)
    for axis, (title, image) in zip(axes, detail_panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle("Same lung ROI at native pixel scale", fontsize=13)
    save_figure(fig, "detail_comparison")

    labels = [str(row["method"]) for row in final_rows]
    psnr_values = [float(row["psnr_db"]) for row in final_rows]
    ssim_values = [float(row["ssim"]) for row in final_rows]
    colors = ["#7f7f7f", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    psnr_bars = axes[0].bar(labels, psnr_values, color=colors)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(0, max(psnr_values) * 1.18)
    axes[0].set_title("Pixel fidelity")
    axes[0].bar_label(psnr_bars, fmt="%.2f", fontsize=8)
    axes[0].tick_params(axis="x", rotation=25)
    ssim_bars = axes[1].bar(labels, ssim_values, color=colors)
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Structural similarity")
    axes[1].bar_label(ssim_bars, fmt="%.4f", fontsize=8)
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Denoising metrics against paired FDCT", fontsize=14)
    save_figure(fig, "metrics_comparison")

    summary_lines = [
        "任务二：基于原始配对 DICOM 的传统滤波去噪",
        "=" * 60,
        f"FDCT: {FDCT_PATH}",
        f"LDCT: {LDCT_PATH}",
        f"图像位置: {fdct_position}",
        f"图像尺寸: {fdct.shape}",
        f"像素间距: {pairing_metadata['Pixel Spacing']} mm",
        f"层厚: {pairing_metadata['Slice Thickness']} mm",
        f"重建核: {pairing_metadata['Convolution Kernel']}",
        f"肺窗: center={WINDOW_CENTER:g} HU, width={WINDOW_WIDTH:g} HU, "
        f"range=[{WINDOW_MIN:g}, {WINDOW_MAX:g}] HU",
        "预处理: HU 截断后统一线性映射为 uint8 0~255",
        "参数选择: SSIM 优先，PSNR 作为同分决胜项",
        "评价范围: 完整 512x512 图像（无标题、坐标轴和白边）",
        "",
        "最终指标",
        "-" * 60,
        "Method            PSNR(dB)    SSIM      Delta PSNR   Delta SSIM",
    ]
    for row in final_rows:
        summary_lines.append(
            f"{str(row['method']):17s} "
            f"{float(row['psnr_db']):9.3f}  "
            f"{float(row['ssim']):8.5f}  "
            f"{float(row['delta_psnr_db']):10.3f}  "
            f"{float(row['delta_ssim']):10.5f}"
        )
    summary_lines.extend(
        [
            "",
            "最优参数",
            "-" * 60,
            *[f"{method}: {best[method]['parameters']}" for method in method_order],
            "",
            f"按 SSIM 优先规则，综合最优方法: {best_overall['method']}",
            f"PSNR={float(best_overall['psnr_db']):.3f} dB, "
            f"SSIM={float(best_overall['ssim']):.5f}",
            "",
            "边界说明: 本次只验证 L067 的一张严格配对切片；不能据此推出所有患者均相同。",
        ]
    )
    (RESULTS_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    manifest = {
        "command": f"{sys.executable} {Path(__file__).name}",
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "fdct": {"path": str(FDCT_PATH), "sha256": sha256(FDCT_PATH)},
            "ldct": {"path": str(LDCT_PATH), "sha256": sha256(LDCT_PATH)},
        },
        "window": {"center_hu": WINDOW_CENTER, "width_hu": WINDOW_WIDTH},
        "pairing_checks": pairing_metadata,
        "packages": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "SimpleITK": sitk.Version_VersionString(),
            "matplotlib": matplotlib.__version__,
            "PyWavelets": pywt.__version__,
            "scikit-image": skimage.__version__,
        },
        "selection_rule": "maximize SSIM; break ties with PSNR",
        "best_parameters": {
            method: best[method]["parameter_dict"] for method in method_order
        },
    }
    (RESULTS_DIR / "reproducibility.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print(f"\n结果目录：{TASK_DIR}")


if __name__ == "__main__":
    main()
