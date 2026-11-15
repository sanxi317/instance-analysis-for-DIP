import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task2.algorithm.code import calculate_metrics, cls_filter, read_gray, wiener_filter


# ==========================================
# 2. 实验与绘图主程序
# ==========================================
if __name__ == "__main__":
    gt_path = ROOT / "data/div2k_gt.png"
    lq_path = ROOT / "task1/output/div2k_15x15_reflect.png"

    save_dir = Path(__file__).resolve().parent / "param_fixed"
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams['axes.unicode_minus'] = False

    img_gt = read_gray(gt_path)
    img_lq = read_gray(lq_path)

    psf = np.load(ROOT / "task1/output/psf_15x15.npy").astype(np.float64)
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

    rows = []
    for method, param_name, values, psnrs, ssims in (
        ("Wiener", "K", K_list, wiener_psnrs, wiener_ssims),
        ("CLS", "gamma", gamma_list, cls_psnrs, cls_ssims),
    ):
        for value, p, s in zip(values, psnrs, ssims):
            rows.append({
                "Method": method, "Parameter": param_name, "Value": value,
                "PSNR": p, "SSIM": s,
            })
    pd.DataFrame(rows).to_csv(save_dir / "task2_parameter_metrics.csv", index=False)

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
