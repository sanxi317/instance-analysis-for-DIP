
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import find_peaks
from skimage.transform import radon
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ==========================================
# 0. 路径和实验参数
# ==========================================

# 当前文件位于 experiment2/task3_extra/radon_motion.py
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "output"

# 7 组合成实验
SYNTHETIC_CONFIGS = [
    (0, 23),
    (30, 23),
    (60, 23),
    (90, 23),
    (30, 12),
    (30, 40),
    (30, 60),
]

# 与 Task 2 保持一致，暂不调参
K = 0.01
GAMMA = 0.01

# Radon 分析的中心裁剪尺寸
ANALYSIS_SIZE = 512

# 长度搜索范围，单位：像素
LENGTH_MIN = 8
LENGTH_MAX = 60

# 是否运行对应实验
RUN_SYNTHETIC = True
RUN_REALBLUR = True


# ==========================================
# 1. 基础图像操作
# ==========================================

def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")

    return img.astype(np.float64) / 255.0


def save_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)

    img_uint8 = np.uint8(
        np.round(np.clip(img, 0, 1) * 255)
    )

    if not cv2.imwrite(str(path), img_uint8):
        raise IOError(f"无法保存图像: {path}")


def calculate_metrics(gt, img):
    if gt.shape != img.shape:
        raise ValueError(
            f"图像尺寸不一致: {gt.shape} vs {img.shape}"
        )

    pred = np.clip(img, 0, 1)

    return (
        float(psnr(gt, pred, data_range=1.0)),
        float(ssim(gt, pred, data_range=1.0))
    )


# ==========================================
# 2. 运动 PSF 构造
# ==========================================

def make_motion_psf(length, angle):
    """
    构造归一化的线性运动 PSF。

    角度约定：
    0°：水平
    90°：竖直
    图像坐标下顺时针为正方向。

    使用双线性权重离散运动轨迹。
    """

    if length <= 0:
        raise ValueError("length 必须大于 0")

    size = int(np.ceil(length)) + 4

    if size % 2 == 0:
        size += 1

    psf = np.zeros((size, size), dtype=np.float64)

    center = size // 2
    angle_rad = np.deg2rad(angle)

    ts = np.linspace(
        -(length - 1) / 2,
        (length - 1) / 2,
        max(80, int(np.ceil(length * 12)))
    )

    for t in ts:
        x = center + t * np.cos(angle_rad)
        y = center + t * np.sin(angle_rad)

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))

        dx = x - x0
        dy = y - y0

        neighbors = [
            (y0, x0, (1-dy)*(1-dx)),
            (y0, x0+1, (1-dy)*dx),
            (y0+1, x0, dy*(1-dx)),
            (y0+1, x0+1, dy*dx),
        ]

        for yy, xx, weight in neighbors:
            if 0 <= yy < size and 0 <= xx < size:
                psf[yy, xx] += weight

    return psf / np.sum(psf)


def pad_psf(psf, img_shape):
    """
    将 PSF 填充到图像尺寸并移动核中心。

    注意使用 -(kh//2)，避免之前 Task 2
    的负数整除偏移错误。
    """

    kh, kw = psf.shape
    h, w = img_shape

    if kh > h or kw > w:
        raise ValueError("PSF 尺寸超过图像尺寸")

    padded = np.zeros(img_shape, dtype=np.float64)
    padded[:kh, :kw] = psf

    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)

    return padded


def circular_blur(img, psf):
    """
    周期卷积生成合成模糊图像。

    与后面的 FFT Wiener/CLS 使用相同的
    卷积边界假设。
    """

    H = fft2(pad_psf(psf, img.shape))
    F = fft2(img)

    return np.real(ifft2(F * H))


# ==========================================
# 3. 模块一：Radon 参数估计
# ==========================================

def estimate_motion(img):
    """
    从 LQ 图像估计运动方向和长度。

    输入：
        img: 二维灰度图像，范围 [0,1]

    返回：
        angle_est: 估计方向（度）
        length_est: 估计长度（像素）
                    无法检测时为 None
        diag: 诊断数据

    注意：
        该方法假设存在近似线性运动模糊。
        GT 不参与估计。
    """

    # --------------------------------------
    # 3.1 选择中心区域
    # --------------------------------------

    n = min(ANALYSIS_SIZE, *img.shape)
    n -= n % 2

    if n < 128:
        raise ValueError("图像尺寸过小，至少需要 128×128")

    h, w = img.shape

    tile = img[
        (h-n)//2:(h-n)//2+n,
        (w-n)//2:(w-n)//2+n
    ]

    # --------------------------------------
    # 3.2 频谱预处理
    # --------------------------------------

    window = np.outer(
        np.hanning(n),
        np.hanning(n)
    )

    F = np.fft.fftshift(
        np.fft.fft2(
            (tile - tile.mean()) * window
        )
    )

    spectrum = np.log1p(np.abs(F))

    background = gaussian_filter(
        spectrum,
        sigma=13
    )

    # 增强相对周围背景较暗的频谱结构
    dark = np.maximum(background - spectrum, 0)

    yy, xx = np.ogrid[:n, :n]

    radius = np.hypot(
        xx - n//2,
        yy - n//2
    )

    # 排除中心强直流区域和外围高频区域
    valid_region = (
        (radius > 10) &
        (radius < 0.40 * n)
    )

    dark *= valid_region

    # --------------------------------------
    # 3.3 Radon 方向估计
    # --------------------------------------

    angles = np.arange(0, 180, 1)

    sinogram = radon(
        dark,
        theta=angles,
        circle=False,
        preserve_range=True
    )

    rho = (
        np.arange(sinogram.shape[0])
        - sinogram.shape[0] // 2
    )

    use = (
        (np.abs(rho) > 7) &
        (np.abs(rho) < 0.38 * n)
    )

    angle_scores = np.std(
        sinogram[use, :],
        axis=0
    )

    radon_angle = float(
        angles[np.argmax(angle_scores)]
    )

    # Radon 坐标转换成图像运动方向
    angle_est = (180.0 - radon_angle) % 180.0

    # --------------------------------------
    # 3.4 沿运动方向提取频谱剖面
    # --------------------------------------

    distances = np.arange(
        0,
        0.38 * n,
        dtype=np.float64
    )

    a = np.deg2rad(angle_est)

    profiles = []

    for offset in (-3, -2, -1, 0, 1, 2, 3):

        xs = (
            n/2
            + distances * np.cos(a)
            - offset * np.sin(a)
        )

        ys = (
            n/2
            + distances * np.sin(a)
            + offset * np.cos(a)
        )

        profile = map_coordinates(
            spectrum,
            [ys, xs],
            order=1,
            mode="nearest"
        )

        profiles.append(profile)

    profile = np.mean(profiles, axis=0)

    # --------------------------------------
    # 3.5 估计运动长度
    # --------------------------------------

    smooth_profile = gaussian_filter(
        profile,
        sigma=9
    )

    notch_strength = smooth_profile - profile

    peaks, properties = find_peaks(
        notch_strength,
        distance=8,
        prominence=0.06
    )

    valid_peaks = (
        (peaks >= n / LENGTH_MAX) &
        (peaks <= n / LENGTH_MIN)
    )

    peaks = peaks[valid_peaks]

    prominences = (
        properties["prominences"][valid_peaks]
    )

    length_est = None
    selected_notch = None

    if len(peaks) > 0:
        # 保留原版的启发式长度估计方法
        # 最显著暗纹不一定是第一个零点
        selected_notch = int(
            peaks[np.argmax(prominences)]
        )

        length_est = float(
            np.clip(
                n / selected_notch,
                LENGTH_MIN,
                LENGTH_MAX
            )
        )

    # --------------------------------------
    # 3.6 保存诊断信息
    # --------------------------------------

    diag = {
        "spectrum": spectrum,
        "dark": dark,
        "sinogram": sinogram,
        "angles": angles,
        "angle_scores": angle_scores,
        "radon_angle": radon_angle,
        "distances": distances,
        "profile": profile,
        "notch_strength": notch_strength,
        "notches": peaks,
        "selected_notch": selected_notch
    }

    return angle_est, length_est, diag


# ==========================================
# 4. 模块二：Wiener / CLS 复原
# ==========================================

def restore(img, psf, method):
    H = fft2(pad_psf(psf, img.shape))
    G = fft2(img)

    if method == "wiener":
        F_hat = (
            np.conj(H) * G
            / (np.abs(H)**2 + K)
        )

    elif method == "cls":

        laplacian = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float64)

        P = fft2(
            pad_psf(laplacian, img.shape)
        )

        F_hat = (
            np.conj(H) * G
            / (
                np.abs(H)**2
                + GAMMA * np.abs(P)**2
            )
        )

    else:
        raise ValueError(
            f"未知复原方法: {method}"
        )

    return np.real(ifft2(F_hat))


# ==========================================
# 5. 模块三：诊断图和实验结果
# ==========================================

def save_diagnostics(diag, out):
    """保存 Radon 方向和长度估计过程。"""

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10)
    )

    axes[0, 0].imshow(
        diag["spectrum"],
        cmap="gray"
    )
    axes[0, 0].set_title("Log FFT Spectrum")

    axes[0, 1].imshow(
        diag["dark"],
        cmap="gray"
    )
    axes[0, 1].set_title("Enhanced Dark Spectrum")

    axes[1, 0].imshow(
        diag["sinogram"],
        cmap="magma",
        aspect="auto",
        extent=[
            0, 179,
            diag["sinogram"].shape[0], 0
        ]
    )
    axes[1, 0].set_title("Radon Sinogram")
    axes[1, 0].set_xlabel("Radon Angle (deg)")

    axes[1, 1].plot(
        diag["angles"],
        diag["angle_scores"]
    )
    axes[1, 1].axvline(
        diag["radon_angle"],
        color="r",
        linestyle="--",
        label=f"Radon = {diag['radon_angle']:.0f} deg"
    )
    axes[1, 1].set_xlabel("Radon Angle (deg)")
    axes[1, 1].set_ylabel("Projection Score")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    for ax in axes.flat[:2]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        out / "direction_diagnostics.png",
        dpi=180
    )
    plt.close(fig)

    # 长度估计图
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        diag["distances"],
        diag["notch_strength"],
        label="Spectral Dip"
    )

    if len(diag["notches"]) > 0:
        ax.scatter(
            diag["notches"],
            diag["notch_strength"][diag["notches"]],
            s=35,
            label="Candidate Notches"
        )

    if diag["selected_notch"] is not None:
        ax.axvline(
            diag["selected_notch"],
            color="r",
            linestyle="--",
            label="Selected Notch"
        )

    ax.set_xlabel("Frequency Distance (FFT bins)")
    ax.set_ylabel("Local Spectral Dip")
    ax.set_title("Motion Length Estimation")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        out / "length_diagnostics.png",
        dpi=180
    )
    plt.close(fig)


def save_comparison(images, out):
    """将 GT、LQ 和各复原方法放到同一张图。"""

    count = len(images)
    cols = 3
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 5 * rows)
    )

    axes = np.array(axes).reshape(-1)

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(
            np.clip(img, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        ax.set_title(name)
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        out / "comparison.png",
        dpi=180
    )
    plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ==========================================
# 6. 单组实验执行
# ==========================================

def run_case(
    name,
    lq,
    gt,
    true_psf=None,
    true_angle=None,
    true_length=None
):
    """
    同时支持：
    - synthetic：有真实 PSF，可验证估计精度
    - realblur：没有真实 PSF，仅评价复原效果
    """

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {name} =====")

    save_gray(out / "lq.png", lq)

    if lq.shape != gt.shape:
        raise ValueError("GT 与 LQ 图像尺寸不一致")

    # --------------------------------------
    # 6.1 Radon 估计
    # --------------------------------------

    angle_est, length_est, diag = estimate_motion(lq)

    save_diagnostics(diag, out)

    print(f"估计方向: {angle_est:.2f} deg")

    if length_est is None:
        print("长度估计失败：没有检测到有效候选暗纹")

        write_csv(
            out / "metrics.csv",
            [{
                "Case": name,
                "Method": "LQ",
                "PSNR": calculate_metrics(gt, lq)[0],
                "SSIM": calculate_metrics(gt, lq)[1],
                "Estimated_Angle": angle_est,
                "Estimated_Length": "",
                "Status": "Length estimation failed"
            }]
        )

        return {
            "Case": name,
            "True_Angle": true_angle,
            "Estimated_Angle": angle_est,
            "Angle_Error": "",
            "True_Length": true_length,
            "Estimated_Length": "",
            "Length_Error": "",
            "Status": "Failed"
        }

    print(f"估计长度: {length_est:.2f} px")

    # --------------------------------------
    # 6.2 构造估计 PSF
    # --------------------------------------

    estimated_psf = make_motion_psf(
        length_est,
        angle_est
    )

    # 保存原始数值，避免 PNG 量化误差
    np.save(
        out / "motion_psf.npy",
        estimated_psf
    )

    # PNG 仅用于可视化
    save_gray(
        out / "motion_psf.png",
        estimated_psf / estimated_psf.max()
    )

    # --------------------------------------
    # 6.3 估计 PSF 复原
    # --------------------------------------

    est_wiener = restore(
        lq, estimated_psf, "wiener"
    )

    est_cls = restore(
        lq, estimated_psf, "cls"
    )

    images = {
        "GT": gt,
        "LQ": lq,
        "Estimated PSF - Wiener": est_wiener,
        "Estimated PSF - CLS": est_cls,
    }

    # --------------------------------------
    # 6.4 合成实验的真实 PSF 对照
    # --------------------------------------

    if true_psf is not None:

        true_wiener = restore(
            lq, true_psf, "wiener"
        )

        true_cls = restore(
            lq, true_psf, "cls"
        )

        images["True PSF - Wiener"] = true_wiener
        images["True PSF - CLS"] = true_cls

        np.save(
            out / "true_psf.npy",
            true_psf
        )

    # --------------------------------------
    # 6.5 保存复原图和指标
    # --------------------------------------

    rows = []

    for method, img in images.items():

        p, s = calculate_metrics(gt, img)

        print(
            f"{method:25s} "
            f"PSNR={p:.3f} "
            f"SSIM={s:.5f}"
        )

        if method != "GT":
            filename = (
                method.lower()
                .replace(" ", "_")
                .replace("-", "")
            )

            save_gray(
                out / f"{filename}.png",
                img
            )

        rows.append({
            "Case": name,
            "Method": method,
            "PSNR": p,
            "SSIM": s,
            "Estimated_Angle": angle_est,
            "Estimated_Length": length_est,
            "Status": "OK"
        })

    write_csv(
        out / "metrics.csv",
        rows
    )

    save_comparison(images, out)

    # --------------------------------------
    # 6.6 计算合成实验的估计误差
    # --------------------------------------

    angle_error = ""
    length_error = ""

    if true_angle is not None:
        diff = abs(angle_est - true_angle) % 180
        angle_error = min(diff, 180 - diff)

        print(
            f"真实角度={true_angle:.1f}, "
            f"角度误差={angle_error:.2f} deg"
        )

    if true_length is not None:
        length_error = abs(length_est - true_length)

        print(
            f"真实长度={true_length}, "
            f"长度误差={length_error:.2f} px"
        )

    return {
        "Case": name,
        "True_Angle": true_angle if true_angle is not None else "",
        "Estimated_Angle": angle_est,
        "Angle_Error": angle_error,
        "True_Length": true_length if true_length is not None else "",
        "Estimated_Length": length_est,
        "Length_Error": length_error,
        "Status": "OK"
    }


# ==========================================
# 7. 主程序
# ==========================================

def main():

    OUT.mkdir(parents=True, exist_ok=True)

    summary = []

    # --------------------------------------
    # 7.1 多组合成验证
    # --------------------------------------

    if RUN_SYNTHETIC:

        gt_syn = read_gray(
            ROOT / "data" / "div2k_gt.png"
        )

        for angle, length in SYNTHETIC_CONFIGS:

            name = f"synthetic/angle_{angle}_length_{length}"

            true_psf = make_motion_psf(
                length,
                angle
            )

            # 修复：使用周期卷积生成模糊
            # 与 FFT 复原使用同一退化模型
            syn_lq = circular_blur(
                gt_syn,
                true_psf
            )

            try:
                result = run_case(
                    name,
                    syn_lq,
                    gt_syn,
                    true_psf=true_psf,
                    true_angle=angle,
                    true_length=length
                )
                summary.append(result)

            except Exception as e:
                print(f"合成实验 {name} 失败: {e}")

                summary.append({
                    "Case": name,
                    "True_Angle": angle,
                    "Estimated_Angle": "",
                    "Angle_Error": "",
                    "True_Length": length,
                    "Estimated_Length": "",
                    "Length_Error": "",
                    "Status": f"Error: {e}"
                })

        write_csv(
            OUT / "synthetic_summary.csv",
            summary
        )

    # --------------------------------------
    # 7.2 RealBlur-J 真实实验
    # --------------------------------------

    if RUN_REALBLUR:

        gt_real = read_gray(
            ROOT / "task1/output/realblur_gt_gray.png"
        )

        lq_real = read_gray(
            ROOT / "task1/output/realblur_blur_gray.png"
        )

        try:
            run_case(
                "realblur",
                lq_real,
                gt_real
            )

        except Exception as e:
            print(f"RealBlur-J 实验失败: {e}")

    print("\n全部实验执行结束")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    main()
