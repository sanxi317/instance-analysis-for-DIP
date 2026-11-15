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

from task2.algorithm.code import (
    calculate_metrics,
    cls_filter,
    get_gaussian_psf,
    inverse_filter_full,
    inverse_filter_limited,
    read_gray,
    wiener_filter,
)


# ==========================================
# 2. RealBlur-J 分支实验（控制变量法）
# ==========================================
if __name__ == "__main__":
    lq_path = ROOT / "task1/output/realblur_blur_gray.png"
    gt_path = ROOT / "task1/output/realblur_gt_gray.png"

    save_dir = Path(__file__).resolve().parent / "realblur_fixed"
    os.makedirs(save_dir, exist_ok=True)

    img_gt = read_gray(gt_path)
    img_lq = read_gray(lq_path)

    print(f"图像尺寸: {img_gt.shape}")

    # 严格的控制变量法配置：
    # 统一的受限逆滤波半径列表
    fixed_D0_list = [15, 30, 50, 70, 90]

    psf_configs = [
        # 【第一组：控制 sigma=3.0 不变，只改变 size】
        {"name": "Test_Size_15x15_sigma3", "size": 15, "sigma": 3.0, "D0_list": fixed_D0_list},
        {"name": "Test_Size_35x35_sigma3", "size": 35, "sigma": 3.0, "D0_list": fixed_D0_list},
         
        # 【第二组：固定 size=35，改变 sigma，与第一组的 35×35 核比较】
        {"name": "Test_Sigma_35x35_sigma5", "size": 35, "sigma": 5.0, "D0_list": fixed_D0_list},
        {"name": "Test_Sigma_35x35_sigma7", "size": 35, "sigma": 7.0, "D0_list": fixed_D0_list}
    ]

    lq_p, lq_s = calculate_metrics(img_gt, img_lq)
    results_list = [{"PSF_Guess": "-", "Method": "LQ", "PSNR": lq_p, "SSIM": lq_s}]

    for cfg in psf_configs:
        print(f"\n===== 使用假设核: {cfg['name']} =====")
        psf = get_gaussian_psf(cfg["size"], cfg["sigma"])
        D0_list = cfg["D0_list"]

        # 执行算法
        img_inv_full = inverse_filter_full(img_lq, psf)
        img_inv_list = [inverse_filter_limited(img_lq, psf, d0) for d0 in D0_list]
        img_wiener = wiener_filter(img_lq, psf)
        img_cls = cls_filter(img_lq, psf)

        # 收集指标
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

        # ==========================================
        # 3. 保存单张图
        # ==========================================
        single_dir = os.path.join(save_dir, "singles", cfg["name"])
        os.makedirs(single_dir, exist_ok=True)
        
        titles = ["LQ"] + ["Full_Inverse"] + [f"Inv_D0_{d0}" for d0 in D0_list] + ["Wiener", "CLS", "GT"]
        images = [img_lq, img_inv_full] + img_inv_list + [img_wiener, img_cls, img_gt]
        
        for t, img in zip(titles, images):
            img_to_save = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(single_dir, f"{t}.png"), img_to_save)

        # ==========================================
        # 4. 可视化组合图 (2行4列)
        # ==========================================
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"RealBlur-J Restoration (Guessed PSF: {cfg['name']})", fontsize=18)

        # 展示全逆、D0=30 和 D0=50 的受限逆、维纳、CLS
        plot_titles = ["LQ (Real Blur)", "Full Inverse",
                       f"Inv D0={D0_list[1]}", f"Inv D0={D0_list[2]}",
                       "Wiener (K=0.01)", "CLS (γ=0.01)",
                       "GT (Sharp)", "—"]
        plot_images = [img_lq, img_inv_full, img_inv_list[1], img_inv_list[2],
                       img_wiener, img_cls, img_gt, np.zeros_like(img_gt)]

        for i, ax in enumerate(axes.flat):
            if i == 7:
                ax.axis("off")
                continue
            ax.imshow(np.clip(plot_images[i], 0, 1), cmap="gray")
            ax.set_title(plot_titles[i])
            ax.axis("off")

        plt.tight_layout()
        save_fig = os.path.join(save_dir, f"realblur_{cfg['name']}_compare.png")
        plt.savefig(save_fig, dpi=250)
        plt.close(fig)
        print(f"  对比图及单张图已保存")

    # ==========================================
    # 5. 导出指标表
    # ==========================================
    df = pd.DataFrame(results_list)
    csv_path = os.path.join(save_dir, "realblur_metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n全部完成！")
