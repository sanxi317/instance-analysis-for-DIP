import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. 核心功能函数定义
# ==========================================
def get_gaussian_psf(size, sigma=3.0):
    """生成高斯模糊核 (PSF)"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def pad_psf(psf, img_shape):
    """将 PSF 填充到与原图同样大小"""
    psf_padded = np.zeros(img_shape)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    return psf_padded

def inverse_filter_full(img, psf, eps=1e-3):
    """全逆滤波"""
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2(G / (H + eps)))

def inverse_filter_limited(img, psf, D0):
    """教材标准版：理想低通受限逆滤波 (高频直接置零)"""
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    
    M, N = img.shape
    U, V = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    D = np.sqrt(U**2 + V**2)
    
    F_hat = np.zeros_like(G, dtype=complex)
    mask = D <= D0
    
    # 只在低频区做逆滤波，高频区直接抛弃 (默认就是 0)
    F_hat[mask] = G[mask] / (H[mask] + 1e-6)
    
    return np.real(ifft2(F_hat))

def wiener_filter(img, psf, K=0.01):
    """维纳滤波"""
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + K)) * G))

def cls_filter(img, psf, gamma=0.01):
    """约束最小二乘复原 (CLS)"""
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    P = fft2(pad_psf(laplacian, img.shape))
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2)) * G))

def calculate_metrics(gt, pred):
    """计算 PSNR 和 SSIM"""
    pred = np.clip(pred, 0, 1)
    return psnr(gt, pred, data_range=1.0), ssim(gt, pred, data_range=1.0)

# ==========================================
# 2. 实验执行代码
# ==========================================
if __name__ == "__main__":
    # 路径设置
    gt_path = r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\data\div2k_gt.png"
    lq_paths = [
        r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\task1\output\div2k_15x15_reflect.png",
        r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\task1\output\div2k_19x19_reflect.png"
    ]
    
    save_dir = r"C:\Users\Sueye\Desktop\DIP\example 2\task2\out"
    os.makedirs(save_dir, exist_ok=True)

    # 读取 GT
    img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255.0
    if img_gt is None:
        print(f"错误：找不到 GT 图像 {gt_path}")
        exit()

    results_list = []

    for lq_path in lq_paths:
        filename = os.path.basename(lq_path)
        print(f"\n正在处理: {filename} ...")

        # 设定不同的截止半径 D0
        if "15x15" in filename:
            psf_size = 15
            D0_list = [30, 60, 100, 150, 200] # 从小到大
        elif "19x19" in filename:
            psf_size = 19
            D0_list = [20, 50, 80, 140, 180] 
        else:
            continue

        psf = get_gaussian_psf(psf_size, sigma=3.0)
        img_lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # 执行复原算法
        img_inv_full = inverse_filter_full(img_lq, psf)
        img_inv_D0_list = [inverse_filter_limited(img_lq, psf, d0) for d0 in D0_list]
        img_wiener = wiener_filter(img_lq, psf, K=0.01)
        img_cls = cls_filter(img_lq, psf, gamma=0.01)

        # 收集指标
        results = {
            "LQ": (img_lq, None),
            "Full Inv": (img_inv_full, calculate_metrics(img_gt, img_inv_full)),
            "Wiener": (img_wiener, calculate_metrics(img_gt, img_wiener)),
            "CLS": (img_cls, calculate_metrics(img_gt, img_cls)),
            "GT": (img_gt, None)
        }
        for i, d0 in enumerate(D0_list):
            results[f"Inv D0={d0}"] = (img_inv_D0_list[i], calculate_metrics(img_gt, img_inv_D0_list[i]))

        # 打印指标并保存
        for method, (img, metrics) in results.items():
            if metrics is not None:
                print(f"  {method.ljust(12)}: PSNR={metrics[0]:.2f}, SSIM={metrics[1]:.4f}")
                results_list.append({"Image": filename, "Method": method, "PSNR": metrics[0], "SSIM": metrics[1]})

        # ==========================================
        # 3. 可视化对比图 (3行3列)
        # ==========================================
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f"Restoration Results (Standard Inverse): {filename}", fontsize=20)
        
        titles = ['LQ (Degraded)', 'Full Inverse', f'Inv D0={D0_list[0]}', 
                  f'Inv D0={D0_list[1]}', f'Inv D0={D0_list[2]}', f'Inv D0={D0_list[3]}', 
                  f'Inv D0={D0_list[4]}', 'Wiener Filter', 'CLS Filter']
        
        images = [img_lq, img_inv_full, img_inv_D0_list[0], 
                  img_inv_D0_list[1], img_inv_D0_list[2], img_inv_D0_list[3], 
                  img_inv_D0_list[4], img_wiener, img_cls]
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.clip(images[i], 0, 1), cmap='gray')
            ax.set_title(titles[i])
            ax.axis('off')
            
        plt.tight_layout()
        save_fig_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_compare.png")
        plt.savefig(save_fig_path, dpi=300)
        plt.close()

    # 导出表格
    if results_list:
        df = pd.DataFrame(results_list)
        csv_path = os.path.join(save_dir, "task2_metrics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n全部处理完成！对比图已保存至: {save_dir}")
        print(f"指标表格已保存至: {csv_path}")
