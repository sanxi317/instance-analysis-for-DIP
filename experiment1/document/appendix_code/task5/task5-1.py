# -*- coding: utf-8 -*-
"""
任务5
"""

import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

LDCT_PATH = r"C:\Users\Sueye\Desktop\quarter_3mm\quarter_3mm\L067\quarter_3mm\L067_QD_3_1.CT.0004.0001.2015.12.22.18.12.56.428910.358293547.IMA"

FDCT_PATH = r"C:\Users\Sueye\Desktop\full_3mm\full_3mm\L067\full_3mm\L067_FD_3_1.CT.0002.0001.2015.12.22.18.12.07.5968.358090401.IMA"

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

SCALE = 4   # 4倍超分


# ============================================================
# 1. 读取 .ima（DICOM）并转 HU
# ============================================================
def read_ima(path):
    """读取 .ima / DICOM 文件，返回 HU 值 2D 数组"""
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    hu = arr * slope + intercept

    if hu.ndim == 3:
        hu = hu[0]  # 多帧时取第一帧
    return hu


# ============================================================
# 2. 肺窗 + 归一化到 0~255
# ============================================================
def lung_window(hu, wl=-600, ww=1500):
    """肺窗：窗位-600，窗宽1500，归一化到0~255 uint8"""
    lo, hi = wl - ww / 2, wl + ww / 2
    hu = np.clip(hu, lo, hi)
    return ((hu - lo) / (hi - lo) * 255).astype(np.uint8)


# ============================================================
# 3. 去噪方法（只保留 NLM，任务五顺序分析用）
# ============================================================
def denoise_nlm(img, h=4):
    # 与队友任务二最优参数保持一致：h=4
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


# ============================================================
# 4. 4倍插值（双三次）
# ============================================================
def up4(img, method='bicubic'):
    interp = {
        'nearest':  cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic':  cv2.INTER_CUBIC,
    }[method]
    return cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=interp)


# ============================================================
# 5. 评估指标
# ============================================================
def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

def calc_ssim(a, b):
    return ssim(a, b, data_range=255)


# ============================================================
# 6. 主流程
# ============================================================
def main():
    # ---- 读取 ----
    ld_hu = read_ima(LDCT_PATH)
    fd_hu = read_ima(FDCT_PATH)

    ld = lung_window(ld_hu)
    fd = lung_window(fd_hu)

    print('LDCT shape:', ld.shape, 'FDCT shape:', fd.shape)

    # ============================================================
    # 6a. 顺序影响对比（GT 为插值放大的 FDCT）
    # ============================================================
    # 方案A：先降噪，后超分
    A = up4(denoise_nlm(ld, h=4), 'bicubic')
    gtA = cv2.resize(fd, (A.shape[1], A.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    pA, sA = psnr(A, gtA), calc_ssim(A, gtA)

    # 方案B：先超分，后降噪
    B_up = up4(ld, 'bicubic')
    B = denoise_nlm(B_up, h=4)
    gtB = cv2.resize(fd, (B.shape[1], B.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    pB, sB = psnr(B, gtB), calc_ssim(B, gtB)

    print('\n=== 顺序影响对比（GT为插值放大的FDCT）===')
    print(f'A 先降噪后超分: PSNR={pA:.2f} dB, SSIM={sA:.4f}')
    print(f'B 先超分后降噪: PSNR={pB:.2f} dB, SSIM={sB:.4f}')

    # ============================================================
    # 7. 可视化
    # ============================================================
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1：顺序对比
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(ld, cmap='gray');   axes[0].set_title('LDCT 原图')
    axes[1].imshow(A, cmap='gray');    axes[1].set_title('A 先降噪后超分')
    axes[2].imshow(B, cmap='gray');    axes[2].set_title('B 先超分后降噪')
    axes[3].imshow(gtA, cmap='gray');  axes[3].set_title('FDCT 真值')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_order.png'), dpi=150)

    print(f'\n图已保存到 {OUT_DIR}/')

    # ============================================================
    # 8. 保存指标表
    # ============================================================
    with open(os.path.join(OUT_DIR, 'metrics_task5.txt'), 'w',
              encoding='utf-8') as f:
        f.write('=== 顺序影响（GT为插值放大的FDCT）===\n')
        f.write(f'A 先降噪后超分\tPSNR={pA:.2f}\tSSIM={sA:.4f}\n')
        f.write(f'B 先超分后降噪\tPSNR={pB:.2f}\tSSIM={sB:.4f}\n')
        f.write('\n注：三种插值对比与各去噪+双三次汇总引用队友任务二、三、四结果，本脚本不重复。\n')

    print('指标已保存到 results/metrics_task5.txt')


if __name__ == '__main__':
    main()