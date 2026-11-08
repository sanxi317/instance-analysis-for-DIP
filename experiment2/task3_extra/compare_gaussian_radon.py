
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. 路径和实验参数
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
EXTRA = Path(__file__).resolve().parent

LQ_PATH = ROOT / "task1/output/realblur_blur_gray.png"
GT_PATH = ROOT / "task1/output/realblur_gt_gray.png"
RADON_PSF_PATH = EXTRA / "output/realblur/motion_psf.npy"

OUT = EXTRA / "output/gaussian_radon_comparison"
OUT.mkdir(parents=True, exist_ok=True)

K_LIST = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

# 用于视觉比较的统一 K
DISPLAY_K = 0.1


# ==========================================
# 2. 基础函数
# ==========================================

def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img.astype(np.float64) / 255.0


def get_gaussian_psf(size, sigma):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def pad_psf(psf, shape):
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    if kh > shape[0] or kw > shape[1]:
        raise ValueError("PSF 尺寸超过图像尺寸")
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return padded


def wiener_filter(img, psf, K):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2(np.conj(H) * G / (np.abs(H)**2 + K)))


def calculate_metrics(gt, img):
    img = np.clip(img, 0, 1)
    return (
        psnr(gt, img, data_range=1.0),
        ssim(gt, img, data_range=1.0)
    )


def save_gray(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


# ==========================================
# 3. 主实验：相同 K 对比不同 PSF
# ==========================================

def main():
    gt = read_gray(GT_PATH)
    lq = read_gray(LQ_PATH)
    radon_psf = np.load(RADON_PSF_PATH)

    if gt.shape != lq.shape:
        raise ValueError("GT 和 LQ 图像尺寸不一致")

    if radon_psf.ndim != 2 or not np.all(np.isfinite(radon_psf)):
        raise ValueError("Radon PSF 格式错误")

    if not np.isclose(radon_psf.sum(), 1.0):
        raise ValueError("Radon PSF 未归一化")

    # 与 Task 2 的四组假设高斯 PSF 保持一致
    psf_configs = {
        "Gaussian 15x15 sigma3": get_gaussian_psf(15, 3.0),
        "Gaussian 35x35 sigma3": get_gaussian_psf(35, 3.0),
        "Gaussian 35x35 sigma5": get_gaussian_psf(35, 5.0),
        "Gaussian 35x35 sigma7": get_gaussian_psf(35, 7.0),
        "Radon Motion PSF": radon_psf
    }

    lq_psnr, lq_ssim = calculate_metrics(gt, lq)
    print(f"LQ: PSNR={lq_psnr:.3f}, SSIM={lq_ssim:.5f}\n")

    records = []
    display_images = {"GT": gt, "LQ": lq}

    for name, psf in psf_configs.items():
        print(f"===== {name} =====")

        for K in K_LIST:
            restored = wiener_filter(lq, psf, K)
            p, s = calculate_metrics(gt, restored)

            records.append({
                "PSF": name,
                "K": K,
                "PSNR": p,
                "SSIM": s
            })

            print(f"K={K:<5} PSNR={p:.3f}, SSIM={s:.5f}")

            if np.isclose(K, DISPLAY_K):
                display_images[name] = restored
                filename = name.lower().replace(" ", "_")
                save_gray(OUT / f"{filename}_K_{K}.png", restored)

        print()

    # ======================================
    # 4. 保存完整指标和最优参数
    # ======================================

    df = pd.DataFrame(records)
    df.to_csv(OUT / "all_metrics.csv", index=False)

    best_psnr = df.loc[df.groupby("PSF")["PSNR"].idxmax()]
    best_ssim = df.loc[df.groupby("PSF")["SSIM"].idxmax()]

    best_psnr.to_csv(OUT / "best_psnr.csv", index=False)
    best_ssim.to_csv(OUT / "best_ssim.csv", index=False)

    print("===== 各 PSF 的最高 PSNR =====")
    print(best_psnr[["PSF", "K", "PSNR", "SSIM"]].to_string(index=False))

    print("\n===== 各 PSF 的最高 SSIM =====")
    print(best_ssim[["PSF", "K", "PSNR", "SSIM"]].to_string(index=False))

    # ======================================
    # 5. PSNR / SSIM 参数曲线
    # ======================================

    for metric in ["PSNR", "SSIM"]:
        plt.figure(figsize=(9, 5))

        for name, group in df.groupby("PSF", sort=False):
            plt.plot(
                group["K"],
                group[metric],
                marker="o",
                label=name
            )

        if metric == "PSNR":
            plt.axhline(
                lq_psnr,
                color="gray",
                linestyle="--",
                label="LQ baseline"
            )
        else:
            plt.axhline(
                lq_ssim,
                color="gray",
                linestyle="--",
                label="LQ baseline"
            )

        plt.xlabel("Wiener K")
        plt.ylabel(metric)
        plt.title(f"Gaussian vs Radon PSF - {metric}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / f"{metric.lower()}_vs_k.png", dpi=200)
        plt.close()

    # ======================================
    # 6. 固定 K 的视觉对比
    # ======================================

    ncols = 3
    nrows = int(np.ceil(len(display_images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))

    axes = np.array(axes).reshape(-1)

    for ax, (name, img) in zip(axes, display_images.items()):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    for ax in axes[len(display_images):]:
        ax.axis("off")

    fig.suptitle(f"Gaussian vs Radon PSF (Wiener K={DISPLAY_K})")
    plt.tight_layout()
    plt.savefig(OUT / "comparison_fixed_k.png", dpi=200)
    plt.close()

    print(f"\n全部完成，输出目录: {OUT}")


if __name__ == "__main__":
    main()
