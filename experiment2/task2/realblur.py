import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_gaussian_psf(size, sigma=3.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def pad_psf(psf, img_shape):
    psf_padded = np.zeros(img_shape)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    return psf_padded

def inverse_filter_full(img, psf, eps=1e-3):
    H = fft2(pad_psf(psf, img.shape)); G = fft2(img)
    return np.real(ifft2(G / (H + eps)))

def inverse_filter_limited(img, psf, D0):
    H = fft2(pad_psf(psf, img.shape)); G = fft2(img)
    M, N = img.shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    D = np.sqrt(U**2 + V**2)
    F_hat = np.zeros_like(G, dtype=complex)
    mask = D <= D0
    F_hat[mask] = G[mask] / (H[mask] + 1e-6)
    return np.real(ifft2(F_hat))

def wiener_filter(img, psf, K=0.01):
    H = fft2(pad_psf(psf, img.shape)); G = fft2(img)
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + K)) * G))

def cls_filter(img, psf, gamma=0.01):
    H = fft2(pad_psf(psf, img.shape)); G = fft2(img)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    P = fft2(pad_psf(laplacian, img.shape))
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2)) * G))

def calculate_metrics(gt, pred):
    pred = np.clip(pred, 0, 1)
    return psnr(gt, pred, data_range=1.0), ssim(gt, pred, data_range=1.0)

# ==========================================
# 2. RealBlur-J 分支实验
# ==========================================
if __name__ == "__main__":
    lq_path = r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\task1\output\realblur_blur_gray.png"
    gt_path = r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\task1\output\realblur_gt_gray.png"

    save_dir = r"C:\Users\Sueye\Desktop\DIP\example 2\task2\realblur_out"
    os.makedirs(save_dir, exist_ok=True)

    img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255.0
    img_lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE) / 255.0
    if img_gt is None or img_lq is None:
        print("错误：找不到图像，请检查路径。")
        exit()

    print(f"图像尺寸: {img_gt.shape}")

    psf_configs = [
        {"name": "Gaussian_15x15_sigma3", "size": 15, "sigma": 3.0,
         "D0_list": [20, 40, 60, 80, 100]},
        {"name": "Gaussian_19x19_sigma3", "size": 19, "sigma": 3.0,
         "D0_list": [15, 30, 50, 70, 90]},
    ]

    results_list = []

    for cfg in psf_configs:
        print(f"\n===== 使用假设核: {cfg['name']} =====")
        psf = get_gaussian_psf(cfg["size"], cfg["sigma"])
        D0_list = cfg["D0_list"]

        # 跑算法
        img_inv_full = inverse_filter_full(img_lq, psf)
        img_inv_list = [inverse_filter_limited(img_lq, psf, d0) for d0 in D0_list]
        img_wiener = wiener_filter(img_lq, psf, K=0.01)
        img_cls = cls_filter(img_lq, psf, gamma=0.01)

        # 指标
        results = {
            "Full Inv": (img_inv_full, calculate_metrics(img_gt, img_inv_full)),
            "Wiener":   (img_wiener,   calculate_metrics(img_gt, img_wiener)),
            "CLS":      (img_cls,      calculate_metrics(img_gt, img_cls)),
        }
        for i, d0 in enumerate(D0_list):
            results[f"Inv D0={d0}"] = (img_inv_list[i], calculate_metrics(img_gt, img_inv_list[i]))

        # 打印 + 记录
        for method, (img, m) in results.items():
            print(f"  {method.ljust(12)}: PSNR={m[0]:.2f}, SSIM={m[1]:.4f}")
            results_list.append({
                "PSF_Guess": cfg["name"],
                "Method": method,
                "PSNR": m[0],
                "SSIM": m[1]
            })

        # ===== 可视化: 2行4列 (LQ, 全逆, 两个受限逆, 维纳, CLS, GT) =====
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"RealBlur-J Restoration (Guessed PSF: {cfg['name']})", fontsize=18)

        titles = ["LQ (Real Motion Blur)", "Full Inverse",
                  f"Inv D0={D0_list[0]}", f"Inv D0={D0_list[2]}",
                  "Wiener (K=0.01)", "CLS (γ=0.01)",
                  "GT (Sharp)", "—"]
        images = [img_lq, img_inv_full, img_inv_list[0], img_inv_list[2],
                  img_wiener, img_cls, img_gt, np.zeros_like(img_gt)]

        for i, ax in enumerate(axes.flat):
            if i == 7:
                ax.axis("off")
                continue
            ax.imshow(np.clip(images[i], 0, 1), cmap="gray")
            ax.set_title(titles[i])
            ax.axis("off")

        plt.tight_layout()
        save_fig = os.path.join(save_dir, f"realblur_{cfg['name']}_compare.png")
        plt.savefig(save_fig, dpi=250)
        plt.close()
        print(f"  对比图已保存: {save_fig}")

    # 导出指标表
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(save_dir, "realblur_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n全部完成！指标表已保存至: {csv_path}")