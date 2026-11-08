
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# 路径
ROOT = Path(__file__).resolve().parents[1]
EXTRA_DIR = Path(__file__).resolve().parent

LQ_PATH = ROOT / "task1/output/realblur_blur_gray.png"
GT_PATH = ROOT / "task1/output/realblur_gt_gray.png"
PSF_PATH = EXTRA_DIR / "output/realblur/motion_psf.npy"

OUT = EXTRA_DIR / "output/realblur/wiener_k_test"
OUT.mkdir(parents=True, exist_ok=True)

# 固定 PSF，仅调整 Wiener 的 K
K_LIST = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return img.astype(np.float64) / 255.0


def pad_psf(psf, shape):
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return padded


def wiener_filter(img, psf, K):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    F_hat = np.conj(H) * G / (np.abs(H)**2 + K)
    return np.real(ifft2(F_hat))


def calculate_metrics(gt, img):
    img = np.clip(img, 0, 1)
    return (
        psnr(gt, img, data_range=1.0),
        ssim(gt, img, data_range=1.0)
    )


def save_image(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def main():
    gt = read_gray(GT_PATH)
    lq = read_gray(LQ_PATH)
    psf = np.load(PSF_PATH)

    if gt.shape != lq.shape:
        raise ValueError("GT 和 LQ 尺寸不一致")

    if not np.isclose(psf.sum(), 1.0):
        raise ValueError("读取的 PSF 未归一化")

    results = []
    images = [("GT", gt), ("LQ", lq)]

    p, s = calculate_metrics(gt, lq)
    results.append(["LQ", "-", p, s])

    print(f"LQ: PSNR={p:.3f}, SSIM={s:.5f}")

    for K in K_LIST:
        restored = wiener_filter(lq, psf, K)

        p, s = calculate_metrics(gt, restored)

        results.append(["Wiener", K, p, s])
        images.append((f"Wiener K={K}", restored))

        save_image(OUT / f"wiener_K_{K}.png", restored)

        print(
            f"K={K:<6} "
            f"PSNR={p:.3f}, SSIM={s:.5f}"
        )

    # 保存指标
    with (OUT / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "K", "PSNR", "SSIM"])
        writer.writerows(results)

    # 保存视觉对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (name, img) in zip(axes.flat, images):
        ax.imshow(np.clip(img, 0, 1), cmap="gray",
                  vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")

    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "comparison.png", dpi=200)
    plt.close()

    print(f"\n输出目录: {OUT}")


if __name__ == "__main__":
    main()
