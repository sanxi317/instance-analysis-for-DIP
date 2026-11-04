
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ========== 1. 路径与实验参数 ==========

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIV2K_PATH = DATA_DIR / "div2k_gt.png"
REAL_GT_PATH = DATA_DIR / "realblur_gt.png"
REAL_BLUR_PATH = DATA_DIR / "realblur_blur.png"

KERNEL_SIZES = [15, 19]
PSF_SIGMA = 3.0
NOISE_SIGMA = 0.01
RANDOM_SEED = 42

BORDER_MODES = {
    "zero": "零填充",
    "reflect": "反射延拓",
    "wrap": "周期延拓",
}


# ========== 2. 基础函数 ==========

def read_gray(path):
    """读取灰度图，并归一化到 [0, 1]。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    return img.astype(np.float32) / 255.0


def save_gray(path, img):
    """将 [0, 1] 浮点图像保存为 8-bit PNG。"""
    img_uint8 = np.round(
        np.clip(img, 0, 1) * 255
    ).astype(np.uint8)

    if not cv2.imwrite(str(path), img_uint8):
        raise IOError(f"保存失败: {path}")


def gaussian_psf(size, sigma):
    """构造归一化的二维高斯卷积核。"""
    if size <= 0 or size % 2 == 0:
        raise ValueError("卷积核尺寸必须为正奇数")

    radius = size // 2
    coords = np.arange(-radius, radius + 1)

    xx, yy = np.meshgrid(coords, coords)

    psf = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    psf /= psf.sum()
    return psf.astype(np.float32)


def apply_blur(img, psf, mode):
    """使用指定的边界延拓方式进行卷积。"""

    if mode == "zero":
        return cv2.filter2D(
            img, -1, psf,
            borderType=cv2.BORDER_CONSTANT
        )

    if mode == "reflect":
        return cv2.filter2D(
            img, -1, psf,
            borderType=cv2.BORDER_REFLECT
        )

    if mode == "wrap":
        # filter2D 不直接支持 BORDER_WRAP
        # 先周期填充，再卷积，最后裁剪
        r = psf.shape[0] // 2

        padded = np.pad(
            img,
            ((r, r), (r, r)),
            mode="wrap"
        )

        blurred = cv2.filter2D(
            padded, -1, psf,
            borderType=cv2.BORDER_CONSTANT
        )

        return blurred[
            r:r + img.shape[0],
            r:r + img.shape[1]
        ]

    raise ValueError(f"未知边界模式: {mode}")


# ========== 3. DIV2K：六组退化实验 ==========

def process_div2k():
    gt = read_gray(DIV2K_PATH)

    # 六组实验共用同一份噪声
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(
        loc=0,
        scale=NOISE_SIGMA,
        size=gt.shape
    ).astype(np.float32)

    results = {}

    for size in KERNEL_SIZES:
        psf = gaussian_psf(size, PSF_SIGMA)

        # 保存卷积核，供任务二使用
        np.save(
            OUT_DIR / f"psf_{size}x{size}.npy",
            psf
        )

        for mode, name in BORDER_MODES.items():

            blurred = apply_blur(gt, psf, mode)

            # 添加相同的高斯噪声
            lq = np.clip(
                blurred + noise, 0, 1
            )

            filename = (
                f"div2k_{size}x{size}_{mode}.png"
            )

            save_gray(OUT_DIR / filename, lq)

            results[(size, mode)] = lq

            print(f"已生成: {filename}")

    # 绘制 2×3 对比图
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9)
    )

    for row, size in enumerate(KERNEL_SIZES):
        for col, (mode, name) in enumerate(
            BORDER_MODES.items()
        ):
            ax = axes[row, col]

            ax.imshow(
                results[(size, mode)],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            ax.set_title(
                f"{size}x{size} - {mode}"
            )
            ax.axis("off")

    fig.suptitle(
        "DIV2K: Gaussian Blur + Noise",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "div2k_comparison.png",
        dpi=200
    )
    plt.show()
    plt.close(fig)


# ========== 4. RealBlur-J：灰度化与归一化 ==========

def process_realblur():
    gt = read_gray(REAL_GT_PATH)
    blur = read_gray(REAL_BLUR_PATH)

    if gt.shape != blur.shape:
        raise ValueError(
            f"GT shape {gt.shape} != "
            f"Blur shape {blur.shape}"
        )

    # 保存灰度图像
    save_gray(
        OUT_DIR / "realblur_gt_gray.png", gt
    )
    save_gray(
        OUT_DIR / "realblur_blur_gray.png", blur
    )

    # 保留归一化浮点数据供任务二使用
    np.save(OUT_DIR / "realblur_gt.npy", gt)
    np.save(OUT_DIR / "realblur_blur.npy", blur)

    # 展示 GT 和 Blur
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for ax, img, title in zip(
        axes,
        [gt, blur],
        ["RealBlur-J GT", "RealBlur-J Blur"]
    ):
        ax.imshow(
            img, cmap="gray",
            vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "realblur_comparison.png",
        dpi=200
    )
    plt.show()
    plt.close(fig)

    print("RealBlur-J 预处理完成")


# ========== 5. 主程序 ==========

if __name__ == "__main__":
    process_div2k()
    process_realblur()

    print("\n任务一全部完成！")
