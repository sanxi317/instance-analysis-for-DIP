"""任务4：定性 + 定量结果评价（统一交付脚本）

按《案例二》任务4的三条要求统一交付：
  1. 主观可视化：并排展示 退化图 / 逆滤波 / 维纳 / CLS / 对照组 / GT（全图 + 细节放大）；
  2. 定量指标：在同一分支的归一化灰度 GT 上重算 PSNR、SSIM；
  3. 汇总表格：DIV2K、RealBlur 两分支各方案指标。找到选做实验的 PSF 时加入 Radon 结果。

输出目录：experiment2/task4/
  - task4_metrics_summary.csv  完整汇总表（全参数）
  - div2k_*_full.png / _crop.png   DIV2K 分支并排对比图
  - realblur_full.png / _crop.png  RealBlur 分支并排对比图
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task2.algorithm.code import (
    CLS_GAMMA,
    DIV2K_CUTOFFS,
    DIV2K_MODES,
    WIENER_K,
    calculate_metrics as metrics,
    cls_filter,
    get_gaussian_psf,
    inverse_filter_full,
    inverse_filter_limited,
    wiener_filter,
)
from task3.task3 import SCALE, bicubic_down_up, nlm_denoise, read_gray

OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"
T1 = ROOT / "task1" / "output"
RADON_PSF = ROOT / "task3_extra" / "output" / "realblur" / "motion_psf.npy"

# 图中展示反射组中 PSNR 最高的受限逆滤波半径，指标表保留所有候选值。
DIV2K_REP_D0 = {15: 150, 19: 140}
GAUSS_GUESSES = [(15, 3.0), (35, 3.0), (35, 5.0), (35, 7.0)]   # RealBlur 猜核
GUESS_D0_LIST = [15, 30, 50, 70, 90]
GUESS_REP_D0 = 30
NLM_H_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 7, 9, 11, 13]
RADON_K = 0.10           # task3_extra 参数扫描中的 PSNR 峰值
RADON_SSIM_K = 0.15      # task3_extra 参数扫描中的 SSIM 峰值
RADON_GAMMA = 0.01


# ==================================================
# 1. 评价与可视化
# ==================================================
def normalize(img):
    return img.astype(np.float64) / 255.0


def center_crop_slice(img, size=256):
    """与 task2/param/param_fixed/5_wiener_visual_crop.png 一致：
    图像中心 size×size 固定窗口（h//2 - size//2 : h//2 + size//2）。"""
    h, w = img.shape
    r = size // 2
    return (slice(h // 2 - r, h // 2 + r), slice(w // 2 - r, w // 2 + r))


def artifacts_diagnostics(gt, out_float, border=16):
    """计算越界像素比例、内外区域 PSNR 和相对 GT 的 Sobel 梯度能量。

    梯度能量与越界比例用于辅助读图，噪声和真实边缘都可能抬高梯度能量。
    """
    h, w = gt.shape
    pred = np.clip(out_float, 0, 1)
    mask_b = np.zeros_like(gt, dtype=bool)
    mask_b[:border, :] = True
    mask_b[-border:, :] = True
    mask_b[:, :border] = True
    mask_b[:, -border:] = True
    inner = (slice(border, h - border), slice(border, w - border))

    def grad_energy(im):
        return np.mean(cv2.Sobel(im, cv2.CV_64F, 1, 0) ** 2
                       + cv2.Sobel(im, cv2.CV_64F, 0, 1) ** 2)

    ge_gt = grad_energy(gt)
    return {
        "clip_frac": float(np.mean((out_float < 0) | (out_float > 1))),
        "border_psnr": float(psnr(gt[mask_b], pred[mask_b], data_range=1.0)),
        "inner_psnr": float(psnr(gt[inner], pred[inner], data_range=1.0)),
        "grad_ratio": float(grad_energy(pred) / ge_gt),
    }


def save_figure(images, tag, crop_slices, suptitle):
    """images: [(name, img)]；crop_slices 用中心 256×256 窗口。
    布局与 task2/param/param_fixed/*_visual_crop.png 一致：横排、dpi=250。"""
    n = len(images)

    def draw(slices, suffix):
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
        axes = np.atleast_1d(axes).ravel()
        for ax, (name, img) in zip(axes, images):
            view = img if slices is None else img[slices]
            ax.imshow(np.clip(view, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(name, fontsize=11)
            ax.axis("off")
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"{tag}_{suffix}.png", dpi=250)
        plt.close(fig)

    draw(None, "full")
    draw(crop_slices, "crop")


def save_boundary_figure(size, arts):
    """3×3 边界模式对照图：行 = LQ / Wiener / CLS，列 = zero / reflect / wrap。
    全图视角（边界伪影是本图的观察对象，不裁 crop）。"""
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    rows = [("LQ (blurred)", "lq"), (f"Wiener K={WIENER_K}", "wien"),
            (f"CLS gamma={CLS_GAMMA}", "cls")]
    for i, (rname, key) in enumerate(rows):
        for j, mode in enumerate(DIV2K_MODES):
            axes[i, j].imshow(np.clip(arts[mode][key], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{rname}  [{mode}]", fontsize=12)
            axes[i, j].axis("off")
    fig.suptitle(f"DIV2K {size}x{size}: boundary-mode comparison (zero / reflect / wrap)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"div2k_{size}x{size}_boundary_compare.png", dpi=250)
    plt.close(fig)


# ==================================================
# 2. 汇总表
# ==================================================
rows = []   # 每行: Dataset, PSF_Guess, Method, Parameter, PSNR, SSIM
diag_rows = []   # 每行: Dataset, Method, clip_frac, border_psnr, inner_psnr, grad_ratio

def add(dataset, guess, method, param, p, s):
    rows.append({"Dataset": dataset, "PSF_Guess": guess, "Method": method,
                 "Parameter": param, "PSNR": f"{p:.4f}", "SSIM": f"{s:.5f}"})


def add_diag(dataset, method, gt, out_float):
    d = artifacts_diagnostics(gt, out_float)
    diag_rows.append({"Dataset": dataset, "Method": method,
                      "clip_frac": f"{d['clip_frac']:.5f}",
                      "border_psnr": f"{d['border_psnr']:.3f}",
                      "inner_psnr": f"{d['inner_psnr']:.3f}",
                      "grad_ratio": f"{d['grad_ratio']:.4f}"})


# ==================================================
# 3. DIV2K 分支（非盲复原 + Bicubic 对照组，覆盖三种边界模式）
# ==================================================
gt_div = normalize(read_gray(DATA / "div2k_gt.png"))
for size, d0_list in DIV2K_CUTOFFS.items():
    rep_d0 = DIV2K_REP_D0[size]
    psf = np.load(T1 / f"psf_{size}x{size}.npy")
    mode_arts = {}

    for mode in DIV2K_MODES:
        tag = f"div2k_{size}x{size}_{mode}"
        lq_gray = read_gray(T1 / f"{tag}.png")
        lq = normalize(lq_gray)

        inv_full = inverse_filter_full(lq, psf)
        inv_d0 = {d0: inverse_filter_limited(lq, psf, d0) for d0 in d0_list}
        wien = wiener_filter(lq, psf, WIENER_K)
        cls = cls_filter(lq, psf, CLS_GAMMA)
        _, bicub_gray = bicubic_down_up(lq_gray)
        bicub = normalize(bicub_gray)
        mode_arts[mode] = dict(lq=lq, inv_full=inv_full, inv_d0=inv_d0,
                               wien=wien, cls=cls, bicub=bicub)

        add(tag, "-", "LQ", "-", *metrics(gt_div, lq))
        add(tag, "-", "Full Inverse", "eps=1e-3", *metrics(gt_div, inv_full))
        for d0 in d0_list:
            add(tag, "-", f"Limited Inverse D0={d0}", f"D0={d0}",
                *metrics(gt_div, inv_d0[d0]))
        add(tag, "-", "Wiener", f"K={WIENER_K}", *metrics(gt_div, wien))
        add(tag, "-", "CLS", f"gamma={CLS_GAMMA}", *metrics(gt_div, cls))
        add(tag, "-", "Bicubic Down-Up", f"factor={SCALE}", *metrics(gt_div, bicub))

        for name, out_f in [("LQ", lq), ("Full Inverse", inv_full),
                            (f"Limited Inv D0={rep_d0}", inv_d0[rep_d0]),
                            ("Wiener", wien), ("CLS", cls), ("Bicubic Down-Up", bicub)]:
            add_diag(tag, name, gt_div, out_f)

    # 主对比图（reflect，与 task2 参数分析图同窗口同风格）
    r = mode_arts["reflect"]
    tag = f"div2k_{size}x{size}_reflect"
    slices = center_crop_slice(gt_div, 256)
    save_figure(
        [
            ("LQ (blurred)", r["lq"]),
            ("Full Inverse", r["inv_full"]),
            (f"Limited Inv D0={rep_d0}", r["inv_d0"][rep_d0]),
            (f"Wiener K={WIENER_K}", r["wien"]),
            (f"CLS gamma={CLS_GAMMA}", r["cls"]),
            ("Bicubic Down-Up (control)", r["bicub"]),
            ("GT (reference)", gt_div),
        ],
        tag, slices,
        f"DIV2K {size}x{size}, reflect: restoration and controls",
    )

    # 三种边界模式对照图
    save_boundary_figure(size, mode_arts)

    print(f"[DIV2K {size}x{size}] "
          + " | ".join(f"{m}: LQ={metrics(gt_div, mode_arts[m]['lq'])[0]:.2f} "
                       f"Wiener={metrics(gt_div, mode_arts[m]['wien'])[0]:.2f} "
                       f"CLS={metrics(gt_div, mode_arts[m]['cls'])[0]:.2f}"
                       for m in DIV2K_MODES))

# ==================================================
# 4. RealBlur 分支（盲复原核猜测 + 对照组 + Radon 加分项）
# ==================================================
gt_rb = normalize(read_gray(T1 / "realblur_gt_gray.png"))
lq_rb_gray = read_gray(T1 / "realblur_blur_gray.png")
lq_rb = normalize(lq_rb_gray)

_, bicub_rb_gray = bicubic_down_up(lq_rb_gray)
bicub_rb = normalize(bicub_rb_gray)
nlm_rb = {h: normalize(nlm_denoise(lq_rb_gray, h)) for h in NLM_H_LIST}

add("RealBlur", "-", "LQ", "-", *metrics(gt_rb, lq_rb))
add("RealBlur", "-", "Bicubic Down-Up", f"factor={SCALE}", *metrics(gt_rb, bicub_rb))
for h in NLM_H_LIST:
    add("RealBlur", "-", "NLM", f"h={h}", *metrics(gt_rb, nlm_rb[h]))

guess_arts = {}
for size, sigma in GAUSS_GUESSES:
    gname = f"Gaussian {size}x{size} sigma={sigma}"
    psf = get_gaussian_psf(size, sigma)
    inv_full = inverse_filter_full(lq_rb, psf)
    inv_d0 = {d0: inverse_filter_limited(lq_rb, psf, d0) for d0 in GUESS_D0_LIST}
    wien = wiener_filter(lq_rb, psf, WIENER_K)
    cls = cls_filter(lq_rb, psf, CLS_GAMMA)
    guess_arts[gname] = dict(inv_full=inv_full, inv_d0=inv_d0, wien=wien, cls=cls)

    add("RealBlur", gname, "Full Inverse", "eps=1e-3", *metrics(gt_rb, inv_full))
    for d0 in GUESS_D0_LIST:
        add("RealBlur", gname, f"Limited Inverse D0={d0}", f"D0={d0}",
            *metrics(gt_rb, inv_d0[d0]))
    add("RealBlur", gname, "Wiener", f"K={WIENER_K}", *metrics(gt_rb, wien))
    add("RealBlur", gname, "CLS", f"gamma={CLS_GAMMA}", *metrics(gt_rb, cls))

# 代表面板使用固定 K 下 PSNR 最高的高斯 Wiener 配置，并补充所选核的逆滤波和 CLS。
rep_guess = "Gaussian 15x15 sigma=3.0"
rep_arts = guess_arts[rep_guess]
slices = center_crop_slice(gt_rb, 256)
realblur_panels = [
    ("LQ (blurred)", lq_rb),
    ("Full Inverse", rep_arts["inv_full"]),
    (f"Limited Inv D0={GUESS_REP_D0}", rep_arts["inv_d0"][GUESS_REP_D0]),
    (f"Wiener K={WIENER_K}", rep_arts["wien"]),
    (f"CLS gamma={CLS_GAMMA}", rep_arts["cls"]),
    ("Bicubic Down-Up (control)", bicub_rb),
    ("NLM h=8 (control)", nlm_rb[8]),
]

# 定性诊断
for name, out_f in [("LQ", lq_rb),
                    (f"Full Inverse ({rep_guess})", rep_arts["inv_full"]),
                    (f"Limited Inv D0={GUESS_REP_D0} ({rep_guess})", rep_arts["inv_d0"][GUESS_REP_D0]),
                    (f"Wiener ({rep_guess})", rep_arts["wien"]),
                    (f"CLS ({rep_guess})", rep_arts["cls"]),
                    ("Bicubic Down-Up", bicub_rb),
                    ("NLM h=8", nlm_rb[8])]:
    add_diag("RealBlur", name, gt_rb, out_f)

if RADON_PSF.is_file():
    radon_psf = np.load(RADON_PSF)
    radon_label = "Radon Motion"
    rad_wien_best = wiener_filter(lq_rb, radon_psf, RADON_K)
    add("RealBlur", radon_label, "Wiener", f"K={RADON_K}", *metrics(gt_rb, rad_wien_best))
    for k in (WIENER_K, RADON_SSIM_K):
        add("RealBlur", radon_label, "Wiener", f"K={k}",
            *metrics(gt_rb, wiener_filter(lq_rb, radon_psf, k)))
    add("RealBlur", radon_label, "CLS", f"gamma={RADON_GAMMA}",
        *metrics(gt_rb, cls_filter(lq_rb, radon_psf, RADON_GAMMA)))
    radon_title = f"Radon Wiener K={RADON_K:.2f} (optional)"
    realblur_panels.append((radon_title, rad_wien_best))
    add_diag("RealBlur", radon_title, gt_rb, rad_wien_best)
    print(f"[RealBlur optional] Radon Wiener={metrics(gt_rb, rad_wien_best)[0]:.2f} dB")

realblur_panels.append(("GT (reference)", gt_rb))
save_figure(
    realblur_panels, "realblur", slices,
    "RealBlur-J: restoration and controls (Gaussian guess 15x15, sigma=3)",
)
print(f"[RealBlur] LQ={metrics(gt_rb, lq_rb)[0]:.2f}  "
      f"NLM h=8={metrics(gt_rb, nlm_rb[8])[0]:.2f}")

# ==================================================
# 5. 写出汇总表
# ==================================================
csv_path = OUT / "task4_metrics_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Dataset", "PSF_Guess", "Method",
                                           "Parameter", "PSNR", "SSIM"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\n汇总表已写出: {csv_path}  (共 {len(rows)} 行)")

diag_path = OUT / "task4_artifacts_diagnostics.csv"
with open(diag_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Dataset", "Method", "clip_frac",
                                           "border_psnr", "inner_psnr", "grad_ratio"])
    writer.writeheader()
    writer.writerows(diag_rows)
print(f"定性诊断表已写出: {diag_path}  (共 {len(diag_rows)} 行)")
