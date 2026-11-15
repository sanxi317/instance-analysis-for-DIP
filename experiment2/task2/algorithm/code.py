import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


WIENER_K = 0.01
CLS_GAMMA = 0.01
DIV2K_CUTOFFS = {15: (30, 60, 100, 150, 200), 19: (20, 50, 80, 140, 180)}
DIV2K_MODES = ("zero", "reflect", "wrap")


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img / 255.0

# ==========================================
# 1. 核心功能函数定义
# ==========================================
def get_gaussian_psf(size, sigma=3.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def pad_psf(psf, img_shape):
    psf_padded = np.zeros(img_shape)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -(kh//2), axis=0)
    psf_padded = np.roll(psf_padded, -(kw//2), axis=1)
    return psf_padded

def inverse_filter_full(img, psf, eps=1e-3):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2(G / (H + eps)))

def inverse_filter_limited(img, psf, D0):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    M, N = img.shape
    u = np.fft.fftfreq(M) * M
    v = np.fft.fftfreq(N) * N
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    F_hat = np.zeros_like(G, dtype=complex)
    mask = D <= D0
    F_hat[mask] = G[mask] / (H[mask] + 1e-6)
    return np.real(ifft2(F_hat))

def wiener_filter(img, psf, K=WIENER_K):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + K)) * G))

def cls_filter(img, psf, gamma=CLS_GAMMA):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    P = fft2(pad_psf(laplacian, img.shape))
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2)) * G))

def calculate_metrics(gt, pred):
    pred = np.clip(pred, 0, 1)
    return psnr(gt, pred, data_range=1.0), ssim(gt, pred, data_range=1.0)

def save_comparison(save_dir, name, results, d0_values):
    single_dir = save_dir / "singles" / name
    single_dir.mkdir(parents=True, exist_ok=True)
    plot_entries = [("LQ", "LQ", "LQ (Degraded)"),
                    ("Full Inv", "Full_Inverse", "Full Inverse")]
    plot_entries.extend((f"Inv D0={d0}", f"Inv_D0_{d0}", f"Inv D0={d0}")
                        for d0 in d0_values)
    plot_entries.extend([("Wiener", "Wiener_Filter", "Wiener Filter"),
                         ("CLS", "CLS_Filter", "CLS Filter")])

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Restoration Results (Standard Inverse): {name}.png", fontsize=20)
    for ax, (method, file_stem, title) in zip(axes.flat, plot_entries):
        display = np.clip(results[method], 0, 1)
        img_to_save = (display * 255).astype(np.uint8)
        cv2.imwrite(str(single_dir / f"{file_stem}.png"), img_to_save)
        ax.imshow(display, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_dir / f"{name}_compare.png", dpi=300)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    task1_output = root / "task1" / "output"
    save_dir = Path(__file__).resolve().parent / "picture_fixed"
    save_dir.mkdir(parents=True, exist_ok=True)
    img_gt = read_gray(root / "data" / "div2k_gt.png")

    rows = []
    for size, d0_values in DIV2K_CUTOFFS.items():
        psf = np.load(task1_output / f"psf_{size}x{size}.npy").astype(np.float64)
        for mode in DIV2K_MODES:
            name = f"div2k_{size}x{size}_{mode}"
            filename = f"{name}.png"
            print(f"\n正在处理: {filename}")
            img_lq = read_gray(task1_output / filename)
            results = {
                "LQ": img_lq,
                "Full Inv": inverse_filter_full(img_lq, psf),
                "Wiener": wiener_filter(img_lq, psf),
                "CLS": cls_filter(img_lq, psf),
            }
            for d0 in d0_values:
                results[f"Inv D0={d0}"] = inverse_filter_limited(img_lq, psf, d0)

            for method, img in results.items():
                p, s = calculate_metrics(img_gt, img)
                print(f"  {method:12}: PSNR={p:.2f}, SSIM={s:.4f}")
                rows.append({"Image": filename, "Method": method, "PSNR": p, "SSIM": s})

            if mode == "reflect":
                save_comparison(save_dir, name, results, d0_values)

    pd.DataFrame(rows).to_csv(save_dir / "task2_metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
