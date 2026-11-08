
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)

# ==================================================
# 1. 路径与参数
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TASK1_OUT = ROOT / "task1" / "output"
OUT_DIR = Path(__file__).resolve().parent / "output"

DIV2K_GT = DATA_DIR / "div2k_gt.png"
DIV2K_FILES = [
    "div2k_15x15_reflect.png",
    "div2k_19x19_reflect.png",
]

REAL_GT = TASK1_OUT / "realblur_gt_gray.png"
REAL_LQ = TASK1_OUT / "realblur_blur_gray.png"

# 双三次插值参数
SCALE = 2
CROP_SIZE = 256

# NLM 参数
NLM_H_VALUES = list(range(2, 21, 2))
NLM_TEMPLATE = 7
NLM_SEARCH = 21


# ==================================================
# 2. 基础函数
# ==================================================

def read_gray(path):
    """读取 uint8 灰度图像。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def save_gray(path, img):
    """保存灰度图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if not cv2.imwrite(str(path), img):
        raise IOError(f"保存图像失败: {path}")


def calculate_metrics(gt, pred):
    """计算 PSNR、SSIM，输入为 uint8 灰度图。"""
    if gt.shape != pred.shape:
        raise ValueError(
            f"图像尺寸不一致: GT={gt.shape}, pred={pred.shape}"
        )

    gt_f = gt.astype(np.float32) / 255.0
    pred_f = pred.astype(np.float32) / 255.0

    return (
        psnr(gt_f, pred_f, data_range=1.0),
        ssim(gt_f, pred_f, data_range=1.0),
    )


def central_crop(img, crop_size=CROP_SIZE):
    """裁剪中心区域，便于观察细节。"""
    h, w = img.shape

    ch = min(crop_size, h)
    cw = min(crop_size, w)

    y = (h - ch) // 2
    x = (w - cw) // 2

    return img[y:y + ch, x:x + cw]


def save_comparison(images, titles, path, figsize=None):
    """保存并排比较图。"""
    n = len(images)

    if figsize is None:
        figsize = (4 * n, 4)

    fig, axes = plt.subplots(
        1, n, figsize=figsize, squeeze=False
    )

    for ax, img, title in zip(axes[0], images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# 3. 双三次插值
# ==================================================

def bicubic_upscale(img, scale=SCALE):
    """直接进行双三次插值放大。"""
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def bicubic_down_up(img, factor=SCALE):
    """
    先下采样，再双三次插值恢复原尺寸。

    下采样使用 INTER_AREA，减轻混叠；
    上采样使用 INTER_CUBIC。
    """
    h, w = img.shape

    down_w = max(1, w // factor)
    down_h = max(1, h // factor)

    down = cv2.resize(
        img,
        (down_w, down_h),
        interpolation=cv2.INTER_AREA,
    )

    restored = cv2.resize(
        down,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    return down, restored


def run_bicubic(name, lq_path, gt_path):
    """执行双三次插值及补充重采样实验。"""
    print(f"\n========== Bicubic: {name} ==========")

    lq = read_gray(lq_path)
    gt = read_gray(gt_path)

    if lq.shape != gt.shape:
        raise ValueError(f"{name}: GT 与 LQ 尺寸不一致")

    out = OUT_DIR / "bicubic" / name
    out.mkdir(parents=True, exist_ok=True)

    # ---------- A. 直接放大 2 倍 ----------
    enlarged = bicubic_upscale(lq)
    save_gray(out / "bicubic_x2.png", enlarged)

    # 对应局部区域，统一显示为两倍大小
    gt_crop = central_crop(gt)
    lq_crop = central_crop(lq)

    # GT 用双三次放大作为显示参考
    gt_view = bicubic_upscale(gt_crop)

    # LQ 最近邻放大，仅用于展示原始像素结构
    lq_view = cv2.resize(
        lq_crop,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_NEAREST,
    )

    bicubic_view = bicubic_upscale(lq_crop)

    save_comparison(
        [gt_view, lq_view, bicubic_view],
        [
            "GT (display x2)",
            "LQ (nearest x2)",
            "LQ (bicubic x2)",
        ],
        out / "bicubic_x2_comparison.png",
    )

    # ---------- B. 下采样再双三次插值 ----------
    down, restored = bicubic_down_up(lq)

    save_gray(out / "downsampled.png", down)
    save_gray(out / "bicubic_down_up.png", restored)

    # 保留原尺寸 LQ 作为基线
    lq_p, lq_s = calculate_metrics(gt, lq)
    res_p, res_s = calculate_metrics(gt, restored)

    save_comparison(
        [gt, lq, restored],
        [
            "GT",
            f"LQ (PSNR={lq_p:.2f})",
            f"Down-Up (PSNR={res_p:.2f})",
        ],
        out / "down_up_comparison.png",
    )

    # 也保存局部对比，方便观察细节
    save_comparison(
        [
            central_crop(gt),
            central_crop(lq),
            central_crop(restored),
        ],
        [
            "GT crop",
            "LQ crop",
            "Down-Up crop",
        ],
        out / "down_up_crop.png",
    )

    print(f"LQ:      PSNR={lq_p:.3f}, SSIM={lq_s:.5f}")
    print(f"Down-Up: PSNR={res_p:.3f}, SSIM={res_s:.5f}")

    return [
        {
            "Dataset": name,
            "Method": "LQ",
            "Parameter": "-",
            "PSNR": lq_p,
            "SSIM": lq_s,
        },
        {
            "Dataset": name,
            "Method": "Bicubic_DownUp",
            "Parameter": "factor=2",
            "PSNR": res_p,
            "SSIM": res_s,
        },
    ]


# ==================================================
# 4. RealBlur-J NLM 去噪
# ==================================================

def nlm_denoise(img, h):
    """OpenCV 非局部均值去噪。"""
    return cv2.fastNlMeansDenoising(
        img,
        None,
        h=float(h),
        templateWindowSize=NLM_TEMPLATE,
        searchWindowSize=NLM_SEARCH,
    )


def run_nlm():
    print("\n========== RealBlur-J NLM ==========")

    gt = read_gray(REAL_GT)
    lq = read_gray(REAL_LQ)

    if gt.shape != lq.shape:
        raise ValueError("RealBlur-J GT 与 LQ 尺寸不一致")

    out = OUT_DIR / "nlm"
    out.mkdir(parents=True, exist_ok=True)

    baseline_p, baseline_s = calculate_metrics(gt, lq)

    rows = [{
        "Dataset": "RealBlur",
        "Method": "LQ",
        "Parameter": "-",
        "PSNR": baseline_p,
        "SSIM": baseline_s,
    }]

    images = {}
    tested = {}

    def evaluate_h(h):
        """运行一个参数，防止重复计算。"""
        if h in tested:
            return

        result = nlm_denoise(lq, h)
        p, s = calculate_metrics(gt, result)

        images[h] = result
        tested[h] = (p, s)

        rows.append({
            "Dataset": "RealBlur",
            "Method": "NLM",
            "Parameter": f"h={h}",
            "PSNR": p,
            "SSIM": s,
        })

        save_gray(out / f"nlm_h{h}.png", result)

        print(f"h={h:2d}: PSNR={p:.3f}, SSIM={s:.5f}")

    # ---------- A. 粗搜索 ----------
    for h in NLM_H_VALUES:
        evaluate_h(h)

    # ---------- B. PSNR / SSIM 最优值附近细搜索 ----------
    best_p_h = max(tested, key=lambda h: tested[h][0])
    best_s_h = max(tested, key=lambda h: tested[h][1])

    fine_values = set()

    for center in [best_p_h, best_s_h]:
        fine_values.update(
            range(max(1, center - 2), center + 3)
        )

    for h in sorted(fine_values):
        evaluate_h(h)

    # ---------- C. 最优参数 ----------
    best_p_h = max(tested, key=lambda h: tested[h][0])
    best_s_h = max(tested, key=lambda h: tested[h][1])

    save_gray(out / "best_psnr.png", images[best_p_h])
    save_gray(out / "best_ssim.png", images[best_s_h])

    print(f"\nLQ 基线: PSNR={baseline_p:.3f}, "
          f"SSIM={baseline_s:.5f}")

    print(f"PSNR 最优: h={best_p_h}, "
          f"PSNR={tested[best_p_h][0]:.3f}")

    print(f"SSIM 最优: h={best_s_h}, "
          f"SSIM={tested[best_s_h][1]:.5f}")

    # ---------- D. 指标表 ----------
    df = pd.DataFrame(rows)
    df.to_csv(out / "nlm_metrics.csv", index=False)

    nlm_df = df[df["Method"] == "NLM"].copy()
    nlm_df["h"] = (
        nlm_df["Parameter"]
        .str.replace("h=", "", regex=False)
        .astype(int)
    )
    nlm_df = nlm_df.sort_values("h")

    # ---------- E. 参数曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(
        nlm_df["h"], nlm_df["PSNR"], "o-",
        label="NLM"
    )
    axes[0].axhline(
        baseline_p, linestyle="--", color="gray",
        label="LQ baseline"
    )
    axes[0].set_xlabel("NLM h")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        nlm_df["h"], nlm_df["SSIM"], "o-",
        label="NLM"
    )
    axes[1].axhline(
        baseline_s, linestyle="--", color="gray",
        label="LQ baseline"
    )
    axes[1].set_xlabel("NLM h")
    axes[1].set_ylabel("SSIM")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        out / "nlm_parameter_curve.png", dpi=200
    )
    plt.close(fig)

    # ---------- F. 整图对比 ----------
    save_comparison(
        [gt, lq, images[best_p_h], images[best_s_h]],
        [
            "GT",
            "LQ",
            f"NLM best PSNR (h={best_p_h})",
            f"NLM best SSIM (h={best_s_h})",
        ],
        out / "nlm_comparison.png",
    )

    # ---------- G. 局部对比 ----------
    save_comparison(
        [
            central_crop(gt),
            central_crop(lq),
            central_crop(images[best_p_h]),
            central_crop(images[best_s_h]),
        ],
        [
            "GT crop",
            "LQ crop",
            f"NLM PSNR-best h={best_p_h}",
            f"NLM SSIM-best h={best_s_h}",
        ],
        out / "nlm_crop_comparison.png",
    )

    return rows


# ==================================================
# 5. 运行全部 Task 3 实验
# ==================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # DIV2K：与 Task 2 使用的两组保持一致
    for filename in DIV2K_FILES:
        rows = run_bicubic(
            name=Path(filename).stem,
            lq_path=TASK1_OUT / filename,
            gt_path=DIV2K_GT,
        )
        all_rows.extend(rows)

    # RealBlur-J：双三次插值
    rows = run_bicubic(
        name="RealBlur",
        lq_path=REAL_LQ,
        gt_path=REAL_GT,
    )
    all_rows.extend(rows)

    # RealBlur-J：NLM
    all_rows.extend(run_nlm())

    # 统一指标汇总
    df = pd.DataFrame(all_rows)
    df.to_csv(
        OUT_DIR / "task3_metrics_summary.csv",
        index=False,
    )

    print("\n===================================")
    print("Task 3 全部完成！")
    print(f"结果目录: {OUT_DIR}")
    print("===================================")


if __name__ == "__main__":
    main()
