#!/usr/bin/env python3
"""任务4：以配对 FDCT 为 Ground Truth 评价去噪与 4 倍插值重建。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------- 中文支持 ----------
plt.rcParams["font.sans-serif"] = [
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
    "SimSun",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


TASK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TASK_DIR.parent
RESULTS_DIR = TASK_DIR / "results"

WINDOW_MIN = -1350.0
WINDOW_MAX = 150.0

# 可选 ROI：左上角坐标及宽高。没有可靠标注时保持 None。
ROI_X: int | None = None
ROI_Y: int | None = None
ROI_W: int | None = None
ROI_H: int | None = None

FDCT_FILENAME = "L067_FD_3_1.CT.0002.0001.2015.12.22.18.12.07.5968.358090401.IMA"
LDCT_FILENAME = "L067_QD_3_1.CT.0004.0001.2015.12.22.18.12.56.428910.358293547.IMA"

TASK2_IMAGE_PATHS = {
    "FDCT Ground Truth": PROJECT_DIR / "task2" / "images" / "fdct_ground_truth.png",
    "Original LDCT": PROJECT_DIR / "task2" / "images" / "ldct_input.png",
    "Mean": PROJECT_DIR / "task2" / "images" / "mean.png",
    "Gaussian": PROJECT_DIR / "task2" / "images" / "gaussian.png",
    "Adaptive Median": PROJECT_DIR / "task2" / "images" / "adaptive_median.png",
    "NLM": PROJECT_DIR / "task2" / "images" / "nlm.png",
    "Wavelet": PROJECT_DIR / "task2" / "images" / "wavelet.png",
}

TASK3_IMAGE_PATHS = {
    "NLM + Nearest": PROJECT_DIR / "task3" / "output" / "nearest_image.png",
    "NLM + Bilinear": PROJECT_DIR / "task3" / "output" / "bilinear_image.png",
    "NLM + Bicubic": PROJECT_DIR / "task3" / "output" / "bicubic_image.png",
}


def read_gray(path: Path) -> np.ndarray:
    """用 Pillow 稳健读取含中文路径的单通道 uint8 灰度图。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{path}")
    try:
        with Image.open(path) as image:
            if image.mode not in {"L", "I;16", "I", "F"}:
                raise ValueError(f"要求单通道灰度图，实际 mode={image.mode}：{path}")
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取有效灰度图：{path}；原因：{exc}") from exc
    if array.ndim != 2:
        raise ValueError(f"要求二维单通道图，实际 shape={array.shape}：{path}")
    if array.dtype != np.uint8:
        raise ValueError(f"要求 uint8 图像，实际 dtype={array.dtype}：{path}")
    return np.ascontiguousarray(array)


def normalize_to_uint8(image_hu: np.ndarray) -> np.ndarray:
    """按固定肺窗截断、归一化，并转换为 uint8。"""
    clipped = np.clip(np.asarray(image_hu, dtype=np.float32), WINDOW_MIN, WINDOW_MAX)
    normalized = (clipped - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return np.rint(normalized * 255.0).astype(np.uint8)


def read_dicom_and_preprocess(path: Path) -> tuple[np.ndarray, Any]:
    """使用 SimpleITK 读取单张 DICOM/IMA 并完成统一肺窗预处理。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到 DICOM/IMA：{path}")
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("DICOM 回退或逐像素核验需要安装 SimpleITK。") from exc
    dicom = sitk.ReadImage(str(path))
    array = np.squeeze(sitk.GetArrayFromImage(dicom))
    if array.ndim != 2:
        raise ValueError(f"DICOM 应为二维切片，实际 shape={array.shape}：{path}")
    return normalize_to_uint8(array), dicom


def find_unique_file(filename: str) -> Path:
    """按 task2 记录的精确文件名，在真实目录中定位唯一 DICOM。"""
    matches = [path for path in PROJECT_DIR.rglob(filename) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"无法唯一定位 {filename}，实际匹配 {len(matches)} 个：{matches}"
        )
    return matches[0]


def validate_image(name: str, image: np.ndarray, path: Path) -> str:
    """检查并记录图像的路径、shape、dtype 和灰度范围。"""
    if image.ndim != 2:
        raise ValueError(f"{name} 不是二维灰度图：shape={image.shape}，路径={path}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} 不是 uint8：dtype={image.dtype}，路径={path}")
    line = (
        f"{name}: path={path}; shape={image.shape}; dtype={image.dtype}; "
        f"min={int(image.min())}; max={int(image.max())}"
    )
    print(line)
    return line


def calculate_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """以 GT 为第一个参数计算 PSNR 和 SSIM。"""
    if reference.shape != candidate.shape:
        raise ValueError(
            "不允许对尺寸不同的图像直接计算指标："
            f"GT={reference.shape}, processed={candidate.shape}"
        )
    psnr = peak_signal_noise_ratio(reference, candidate, data_range=255)
    ssim = structural_similarity(reference, candidate, data_range=255)
    return float(psnr), float(ssim)


def verify_pairing(fdct_dicom: Any, ldct_dicom: Any) -> list[str]:
    """核对两张 DICOM 是否为同一位置的配对切片。"""
    tags = {
        "Patient ID": "0010|0020",
        "Image Position Patient": "0020|0032",
        "Image Orientation Patient": "0020|0037",
        "Pixel Spacing": "0028|0030",
        "Slice Thickness": "0018|0050",
        "Convolution Kernel": "0018|1210",
        "Instance Number": "0020|0013",
    }
    lines: list[str] = []
    for label, tag in tags.items():
        fd_value = fdct_dicom.GetMetaData(tag).strip() if fdct_dicom.HasMetaDataKey(tag) else "N/A"
        ld_value = ldct_dicom.GetMetaData(tag).strip() if ldct_dicom.HasMetaDataKey(tag) else "N/A"
        if fd_value != ld_value:
            raise ValueError(f"配对核验失败：{label}: FDCT={fd_value}, LDCT={ld_value}")
        lines.append(f"{label}: {fd_value}")
    return lines


def load_reference_and_ldct() -> tuple[dict[str, np.ndarray], dict[str, Path], list[str]]:
    """优先使用规范 task2 PNG；不合格时回退读取原始 DICOM。"""
    paths = {
        "FDCT Ground Truth": TASK2_IMAGE_PATHS["FDCT Ground Truth"],
        "Original LDCT": TASK2_IMAGE_PATHS["Original LDCT"],
    }
    try:
        images = {name: read_gray(path) for name, path in paths.items()}
        if images["FDCT Ground Truth"].shape != images["Original LDCT"].shape:
            raise ValueError("task2 的 FDCT/LDCT PNG 尺寸不一致")
        if images["FDCT Ground Truth"].shape != (512, 512):
            raise ValueError(f"task2 PNG 不是预期 512x512：{images['FDCT Ground Truth'].shape}")

        fdct_path = find_unique_file(FDCT_FILENAME)
        ldct_path = find_unique_file(LDCT_FILENAME)
        try:
            fdct_check, fdct_dicom = read_dicom_and_preprocess(fdct_path)
            ldct_check, ldct_dicom = read_dicom_and_preprocess(ldct_path)
            if not np.array_equal(images["FDCT Ground Truth"], fdct_check):
                raise ValueError("task2 FDCT PNG 与统一肺窗 DICOM 像素不一致")
            if not np.array_equal(images["Original LDCT"], ldct_check):
                raise ValueError("task2 LDCT PNG 与统一肺窗 DICOM 像素不一致")
            pairing_lines = verify_pairing(fdct_dicom, ldct_dicom)
            pairing_lines.insert(0, "PNG 与 DICOM 统一预处理结果逐像素一致。")
        except RuntimeError:
            pairing_lines = [
                "当前解释器未安装 SimpleITK；运行时使用通过规格检查的 task2 纯灰度 PNG。",
                "DICOM 配对依据 task2 的 reproducibility.json 与 summary.txt。",
            ]
        paths["FDCT DICOM"] = fdct_path
        paths["LDCT DICOM"] = ldct_path
        return images, paths, pairing_lines
    except (FileNotFoundError, ValueError) as png_error:
        print(f"规范 PNG 不可用，回退到原始 DICOM：{png_error}")
        fdct_path = find_unique_file(FDCT_FILENAME)
        ldct_path = find_unique_file(LDCT_FILENAME)
        fdct, fdct_dicom = read_dicom_and_preprocess(fdct_path)
        ldct, ldct_dicom = read_dicom_and_preprocess(ldct_path)
        pairing_lines = verify_pairing(fdct_dicom, ldct_dicom)
        return (
            {"FDCT Ground Truth": fdct, "Original LDCT": ldct},
            {"FDCT DICOM": fdct_path, "LDCT DICOM": ldct_path},
            pairing_lines,
        )


def rebuild_sr_from_nlm(nlm: np.ndarray) -> dict[str, np.ndarray]:
    """当 task3 不是 512 输出时，建立 128→512 公平评价分支。"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("重建分支需要安装 opencv-python。") from exc
    low_resolution = cv2.resize(nlm, (128, 128), interpolation=cv2.INTER_AREA)
    return {
        "NLM + Nearest": cv2.resize(low_resolution, (512, 512), interpolation=cv2.INTER_NEAREST),
        "NLM + Bilinear": cv2.resize(low_resolution, (512, 512), interpolation=cv2.INTER_LINEAR),
        "NLM + Bicubic": cv2.resize(low_resolution, (512, 512), interpolation=cv2.INTER_CUBIC),
    }


def load_all_images() -> tuple[dict[str, np.ndarray], dict[str, Path], list[str], str]:
    images, paths, pairing_lines = load_reference_and_ldct()
    gt_shape = images["FDCT Ground Truth"].shape

    missing = [str(path) for name, path in TASK2_IMAGE_PATHS.items() if name not in images and not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 task2 去噪结果：\n" + "\n".join(missing))
    for name, path in TASK2_IMAGE_PATHS.items():
        if name not in images:
            images[name] = read_gray(path)
            paths[name] = path
        if images[name].shape != gt_shape:
            raise ValueError(
                f"{name} 尺寸与 FDCT 不一致，不静默 resize："
                f"{images[name].shape} vs {gt_shape}，路径={path}"
            )

    missing_sr = [str(path) for path in TASK3_IMAGE_PATHS.values() if not path.is_file()]
    if missing_sr:
        raise FileNotFoundError("缺少 task3 插值结果：\n" + "\n".join(missing_sr))
    sr_images = {name: read_gray(path) for name, path in TASK3_IMAGE_PATHS.items()}
    sr_shapes = {image.shape for image in sr_images.values()}
    if sr_shapes == {gt_shape}:
        sr_mode = "Task3 实际流程为 512→128→512；直接使用三种 512×512 输出与 FDCT 比较。"
        images.update(sr_images)
        paths.update(TASK3_IMAGE_PATHS)
    else:
        sr_mode = (
            f"Task3 输出尺寸为 {sorted(sr_shapes)}，不能直接与 FDCT {gt_shape} 比较；"
            "已在 task4 内从 NLM 建立 INTER_AREA 128×128→三种插值 512×512 的评价分支。"
        )
        print(sr_mode)
        images.update(rebuild_sr_from_nlm(images["NLM"]))
        for name in TASK3_IMAGE_PATHS:
            paths[name] = Path("task4 内存重建（未覆盖 task3）")
    return images, paths, pairing_lines, sr_mode


def create_metrics_table(images: dict[str, np.ndarray]) -> list[dict[str, str]]:
    gt = images["FDCT Ground Truth"]
    specifications = [
        ("Original", "LDCT", "Original LDCT"),
        ("Denoising", "Mean", "Mean"),
        ("Denoising", "Gaussian", "Gaussian"),
        ("Denoising", "Adaptive Median", "Adaptive Median"),
        ("Denoising", "NLM", "NLM"),
        ("Denoising", "Wavelet", "Wavelet"),
        ("Super Resolution", "NLM + Nearest", "NLM + Nearest"),
        ("Super Resolution", "NLM + Bilinear", "NLM + Bilinear"),
        ("Super Resolution", "NLM + Bicubic", "NLM + Bicubic"),
    ]
    rows: list[dict[str, str]] = []
    for category, method, key in specifications:
        psnr, ssim = calculate_metrics(gt, images[key])
        rows.append(
            {"Category": category, "Method": method, "PSNR_dB": f"{psnr:.4f}", "SSIM": f"{ssim:.5f}"}
        )
    rows.append({"Category": "Reference", "Method": "FDCT", "PSNR_dB": "GT", "SSIM": "GT"})
    return rows


def metrics_lookup(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    return {
        row["Method"]: (row["PSNR_dB"], row["SSIM"])
        for row in rows
        if row["Category"] != "Reference"
    }


def panel_title(label: str, method: str, lookup: dict[str, tuple[str, str]]) -> str:
    if method == "FDCT":
        return f"{label}\n参考真值"
    psnr, ssim = lookup[method]
    return f"{label}\nPSNR {psnr} dB | SSIM {ssim}"


def plot_panels(
    panels: list[tuple[str, str, np.ndarray]],
    rows: list[dict[str, str]],
    output: Path,
    columns: int,
    figure_title: str,
    empty_note: str | None = None,
) -> None:
    lookup = metrics_lookup(rows)
    nrows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        nrows,
        columns,
        figsize=(3.3 * columns, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (label, method, image) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(panel_title(label, method, lookup), fontsize=7.5)
        axis.axis("off")
    unused_axes = list(axes.flat[len(panels) :])
    if empty_note and unused_axes:
        unused_axes[0].text(
            0.5,
            0.5,
            empty_note,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.5,
            transform=unused_axes[0].transAxes,
        )
        unused_axes[0].axis("off")
        unused_axes = unused_axes[1:]
    for axis in unused_axes:
        axis.axis("off")
    fig.suptitle(figure_title, fontsize=10)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_denoising_comparison(images: dict[str, np.ndarray], rows: list[dict[str, str]]) -> None:
    panels = [
        ("原始 LDCT", "LDCT", images["Original LDCT"]),
        ("均值滤波", "Mean", images["Mean"]),
        ("高斯滤波", "Gaussian", images["Gaussian"]),
        ("自适应中值滤波", "Adaptive Median", images["Adaptive Median"]),
        ("NLM", "NLM", images["NLM"]),
        ("小波软阈值", "Wavelet", images["Wavelet"]),
        ("FDCT 真值", "FDCT", images["FDCT Ground Truth"]),
    ]
    best = max(
        (row for row in rows if row["Category"] == "Denoising"),
        key=lambda row: (float(row["SSIM"]), float(row["PSNR_dB"])),
    )
    note = (
        "最优去噪\n"
        f"{best['Method']}\n\n"
        f"PSNR {best['PSNR_dB']} dB\n"
        f"SSIM {best['SSIM']}\n\n"
        "参考：配对 FDCT"
    )
    plot_panels(
        panels,
        rows,
        RESULTS_DIR / "comparison_denoising.png",
        4,
        "五种去噪方法与 FDCT 真值对比",
        empty_note=note,
    )


def plot_sr_comparison(images: dict[str, np.ndarray], rows: list[dict[str, str]]) -> None:
    panels = [
        ("NLM", "NLM", images["NLM"]),
        ("NLM + 最近邻", "NLM + Nearest", images["NLM + Nearest"]),
        ("NLM + 双线性", "NLM + Bilinear", images["NLM + Bilinear"]),
        ("NLM + 双三次", "NLM + Bicubic", images["NLM + Bicubic"]),
        ("FDCT 真值", "FDCT", images["FDCT Ground Truth"]),
    ]
    plot_panels(
        panels,
        rows,
        RESULTS_DIR / "comparison_sr.png",
        5,
        "三种插值 4 倍重建与 FDCT 真值对比",
    )


def plot_all_comparison(images: dict[str, np.ndarray], rows: list[dict[str, str]]) -> None:
    panels = [
        ("原始 LDCT", "LDCT", images["Original LDCT"]),
        ("均值滤波", "Mean", images["Mean"]),
        ("高斯滤波", "Gaussian", images["Gaussian"]),
        ("自适应中值滤波", "Adaptive Median", images["Adaptive Median"]),
        ("NLM", "NLM", images["NLM"]),
        ("小波软阈值", "Wavelet", images["Wavelet"]),
        ("NLM + 最近邻", "NLM + Nearest", images["NLM + Nearest"]),
        ("NLM + 双线性", "NLM + Bilinear", images["NLM + Bilinear"]),
        ("NLM + 双三次", "NLM + Bicubic", images["NLM + Bicubic"]),
        ("FDCT 真值", "FDCT", images["FDCT Ground Truth"]),
    ]
    plot_panels(
        panels,
        rows,
        RESULTS_DIR / "comparison_all.png",
        5,
        "去噪与超分综合对比",
    )


def plot_roi_comparison(images: dict[str, np.ndarray], rows: list[dict[str, str]]) -> bool:
    values = (ROI_X, ROI_Y, ROI_W, ROI_H)
    if all(value is None for value in values):
        print("ROI 未设置：不生成 comparison_roi.png。")
        return False
    if any(value is None for value in values):
        raise ValueError("ROI_X、ROI_Y、ROI_W、ROI_H 必须全部设置或全部为 None")
    x, y, width, height = (int(value) for value in values if value is not None)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"ROI 参数无效：x={x}, y={y}, w={width}, h={height}")
    image_height, image_width = images["FDCT Ground Truth"].shape
    if x + width > image_width or y + height > image_height:
        raise ValueError(f"ROI 超出图像范围：ROI=({x},{y},{width},{height}), image={(image_height,image_width)}")

    best_denoising = max(
        (row for row in rows if row["Category"] == "Denoising"),
        key=lambda row: (float(row["SSIM"]), float(row["PSNR_dB"])),
    )["Method"]
    keys = [
        ("LDCT", "LDCT", "Original LDCT"),
        (f"最优：{best_denoising}", best_denoising, best_denoising),
        ("最近邻", "NLM + Nearest", "NLM + Nearest"),
        ("双线性", "NLM + Bilinear", "NLM + Bilinear"),
        ("双三次", "NLM + Bicubic", "NLM + Bicubic"),
        ("FDCT 真值", "FDCT", "FDCT Ground Truth"),
    ]
    panels = [(label, method, images[key][y : y + height, x : x + width]) for label, method, key in keys]
    plot_panels(panels, rows, RESULTS_DIR / "comparison_roi.png", 6, "同一 ROI 局部结构对比")
    return True


def write_metrics(rows: list[dict[str, str]]) -> str:
    with (RESULTS_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["Category", "Method", "PSNR_dB", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'Category':<18} {'Method':<20} {'PSNR (dB)':>12} {'SSIM':>10}"
    lines = [
        "Task4 metrics (reference: paired FDCT, data_range=255)",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row['Category']:<18} {row['Method']:<20} {row['PSNR_dB']:>12} {row['SSIM']:>10}"
        )
    table = "\n".join(lines)
    (RESULTS_DIR / "metrics.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return table


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"项目目录：{PROJECT_DIR}")
    print("开始检查输入；不会修改 task1、task2、task3。")

    images, paths, pairing_lines, sr_mode = load_all_images()
    validation_lines = ["Task4 输入检查", "=" * 80]
    for name, image in images.items():
        validation_lines.append(validate_image(name, image, paths.get(name, Path("内存图像"))))
    validation_lines.extend(["", "DICOM/PNG 配对核验", "-" * 80, *pairing_lines])
    validation_lines.extend(["", "Task3 方案", "-" * 80, sr_mode])
    validation_lines.extend(
        [
            "",
            "原始 DICOM 路径",
            "-" * 80,
            f"FDCT: {paths['FDCT DICOM']}",
            f"LDCT: {paths['LDCT DICOM']}",
        ]
    )
    (RESULTS_DIR / "input_validation.txt").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8"
    )

    rows = create_metrics_table(images)
    write_metrics(rows)
    plot_denoising_comparison(images, rows)
    plot_sr_comparison(images, rows)
    plot_all_comparison(images, rows)
    roi_created = plot_roi_comparison(images, rows)

    expected = [
        RESULTS_DIR / "metrics.csv",
        RESULTS_DIR / "metrics.txt",
        RESULTS_DIR / "input_validation.txt",
        RESULTS_DIR / "comparison_denoising.png",
        RESULTS_DIR / "comparison_sr.png",
        RESULTS_DIR / "comparison_all.png",
        RESULTS_DIR / "comparison_denoising.pdf",
        RESULTS_DIR / "comparison_sr.pdf",
        RESULTS_DIR / "comparison_all.pdf",
    ]
    if roi_created:
        expected.append(RESULTS_DIR / "comparison_roi.png")
        expected.append(RESULTS_DIR / "comparison_roi.pdf")
    missing_outputs = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise OSError("输出生成失败：\n" + "\n".join(missing_outputs))
    print("\nTask4 完成。已生成：")
    for path in expected:
        print(f"- {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nTask4 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise