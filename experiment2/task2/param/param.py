import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
    psf_padded = np.roll(psf_padded, -kh//2, axis=0)
    psf_padded = np.roll(psf_padded, -kw//2, axis=1)
    return psf_padded

def wiener_filter(img, psf, K):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + K)) * G))

def cls_filter(img, psf, gamma):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    P = fft2(pad_psf(laplacian, img.shape))
    return np.real(ifft2((np.conj(H) / (np.abs(H)**2 + gamma * np.abs(P)**2)) * G))

def calculate_metrics(gt, pred):
    pred = np.clip(pred, 0, 1)
    return psnr(gt, pred, data_range=1.0), ssim(gt, pred, data_range=1.0)

# ==========================================
# 2. 实验与绘图主程序
# ==========================================
if __name__ == "__main__":
    gt_path = r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\data\div2k_gt.png"
    lq_path = r"C:\Users\Sueye\Desktop\DIP\example 2\instance-analysis-for-DIP-main\experiment2\task1\output\div2k_15x15_reflect.png"
    save_dir = r"C:\Users\Sueye\Desktop\DIP\example 2\task2\out\param_tuning"
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) / 255.0
    img_lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE) / 255.0
    if img_gt is None or img_lq is None:
        print("错误：找不到图像，请检查路径。")
        exit()

    psf = get_gaussian_psf(15, 3.0)
    K_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    gamma_list = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    x_labels = [str(k) for k in K_list]
    x_pos = range(len(K_list))

    wiener_psnrs, wiener_ssims, wiener_imgs = [], [], []
    for K in K_list:
        img_w = wiener_filter(img_lq, psf, K)
        p, s = calculate_metrics(img_gt, img_w)
        wiener_psnrs.append(p)
        wiener_ssims.append(s)
        wiener_imgs.append(img_w)

    cls_psnrs, cls_ssims, cls_imgs = [], [], []
    for gamma in gamma_list:
        img_c = cls_filter(img_lq, psf, gamma)
        p, s = calculate_metrics(img_gt, img_c)
        cls_psnrs.append(p)
        cls_ssims.append(s)
        cls_imgs.append(img_c)

    print("指标计算完成，开始生成图表...")

    # ==========================================
    # 3. 保存单张图 (新增)
    # ==========================================
    single_dir_w = os.path.join(save_dir, "singles_wiener")
    os.makedirs(single_dir_w, exist_ok=True)
    for i, img in enumerate(wiener_imgs):
        img_to_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(single_dir_w, f"Wiener_K_{K_list[i]}.png"), img_to_save)
        
    single_dir_c = os.path.join(save_dir, "singles_cls")
    os.makedirs(single_dir_c, exist_ok=True)
    for i, img in enumerate(cls_imgs):
        img_to_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(single_dir_c, f"CLS_gamma_{gamma_list[i]}.png"), img_to_save)

    # ==========================================
    # 4. 绘制 4 张独立的折线图
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, wiener_psnrs, 'b-o', linewidth=2)
    plt.title('维纳滤波：不同 K 值的 PSNR', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.xticks(x_pos, x_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "1_wiener_psnr.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, wiener_ssims, 'r-s', linewidth=2)
    plt.title('维纳滤波：不同 K 值的 SSIM', fontsize=14)
    plt.xlabel('Wiener K value', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.xticks(x_pos, x_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "2_wiener_ssim.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, cls_psnrs, 'g-o', linewidth=2)
    plt.title('CLS：不同 γ 值的 PSNR', fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.xticks(x_pos, x_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "3_cls_psnr.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(x_pos, cls_ssims, 'm-s', linewidth=2)
    plt.title('CLS：不同 γ 值的 SSIM', fontsize=14)
    plt.xlabel('CLS gamma value', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.xticks(x_pos, x_labels)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "4_cls_ssim.png"), dpi=250)
    plt.close()

    # ==========================================
    # 5. 绘制局部放大的视觉对比组合图
    # ==========================================
    h, w = img_gt.shape
    crop_slice = (slice(h//2 - 128, h//2 + 128), slice(w//2 - 128, w//2 + 128))
    
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle('维纳滤波局部放大对比 (K 从极小到偏大)', fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(wiener_imgs[i][crop_slice], 0, 1), cmap='gray')
        ax.set_title(f'K = {K_list[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_wiener_visual_crop.png"), dpi=250)
    plt.close()

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle('CLS 局部放大对比 (γ 从极小到偏大)', fontsize=16)
    for i, ax in enumerate(axes):
        ax.imshow(np.clip(cls_imgs[i][crop_slice], 0, 1), cmap='gray')
        ax.set_title(f'γ = {gamma_list[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "6_cls_visual_crop.png"), dpi=250)
    plt.close()

    print(f"\n全部生成完毕！")
