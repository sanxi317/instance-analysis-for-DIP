# Task 3：插值与去噪对照实验

## 1. 任务说明

本任务属于数字图像处理案例二“退化图像复原”，主要完成以下内容：

- **双三次插值（Bicubic）**：对 DIV2K 和 RealBlur-J 退化图像进行 2 倍插值放大，作为不进行显式去模糊处理的对照方法。
- **非局部均值去噪（NLM）**：对 RealBlur-J 退化图像进行去噪，比较不同滤波强度下的处理效果。
- **综合对比**：对 Task 2 和 Task 3 的结果进行统一的 PSNR、SSIM 评价与视觉对比。

## 2. 文件说明

```text
task3/
├── task3.py                  # Bicubic 插值、NLM 去噪及指标计算
├── compare_task2_task3.py    # Task 2 与 Task 3 综合对比
├── README.md
└── output/                   # 实验输出
    └── task2_comparison/     # 综合对比图及指标表
```

`task3.py` 的输出还包括 Bicubic 放大图、Down-Up 对照图、NLM 不同参数的去噪图和指标记录。

## 3. 运行方法

### 3.1 环境依赖

使用 Python 3，主要依赖：

- NumPy
- OpenCV
- scikit-image
- Matplotlib
- pandas

### 3.2 执行实验

在仓库根目录运行：

```bash
python experiment2/task3/task3.py
```

完成 Task 3 后，可以运行综合对比脚本：

```bash
python experiment2/task3/compare_task2_task3.py
```

综合对比脚本依赖 Task 1 的 GT/LQ 图像、Task 2 的复原结果和 Task 3 的处理结果。当前综合对比使用 Task 2 修复后的输出。

综合指标表位于：

```text
experiment2/task3/output/task2_comparison/task2_task3_metrics_recomputed.csv
```

## 4. 实验方法

### 4.1 Bicubic 插值

使用 OpenCV 的 `INTER_CUBIC` 实现双三次插值。

实验包含两种方式：

1. **直接 2 倍放大**：将退化图像放大至原尺寸的 2 倍，用于观察插值后的视觉效果。
2. **Down-Up 对照**：先将退化图像缩小至原尺寸的一半，再采用 Bicubic 插值恢复至原尺寸，以便与 GT 进行同尺寸的 PSNR、SSIM 比较。

需要注意，Down-Up 包含额外的降采样步骤，可能产生平滑效果，其指标变化不能直接解释为去模糊能力的提升。

### 4.2 NLM 去噪

对 RealBlur-J 灰度退化图像使用 OpenCV 的 `fastNlMeansDenoising`。

实验固定模板窗口大小为 7、搜索窗口大小为 21，并调整滤波强度参数 `h`，比较不同参数的 PSNR、SSIM 和视觉效果。

### 4.3 综合对比

将 Task 3 的处理结果与 Task 2 的全逆滤波、受限逆滤波、Wiener 滤波及 CLS 复原结果进行比较。

综合对比脚本从保存的 PNG 图像重新计算指标，因此可能与算法直接输出的浮点图像指标存在小幅差异。

## 5. 主要实验结果

### 5.1 DIV2K

| 方法 | 15×15 PSNR | 15×15 SSIM | 19×19 PSNR | 19×19 SSIM |
|---|---:|---:|---:|---:|
| LQ | 24.861 | 0.64326 | 24.775 | 0.63899 |
| Bicubic Down-Up | 24.939 | 0.66143 | 24.853 | 0.65717 |
| Wiener | 26.812 | 0.77846 | 26.815 | 0.78047 |
| CLS | 26.379 | 0.77171 | 26.408 | 0.77371 |

Bicubic Down-Up 相比 LQ 仅有小幅指标提升。相比之下，使用已知 PSF 的 Wiener 和 CLS 复原取得了更明显的改善。

### 5.2 RealBlur-J

| 方法 | PSNR | SSIM |
|---|---:|---:|
| LQ | 18.92460 | 0.57394 |
| Bicubic Down-Up | 18.94997 | 0.57565 |
| NLM（h=8） | 18.94692 | 0.58211 |
| NLM（h=13） | 18.93324 | 0.58268 |
| NLM（h=20） | 18.89167 | 0.58088 |

在测试的参数中，`h=8` 获得较高的 PSNR，`h=13` 获得较高的 SSIM。NLM 对图像结构相似性有一定改善，但无法直接恢复运动模糊造成的细节损失。

## 6. 注意事项

1. Bicubic 直接 2 倍放大后与原 GT 尺寸不同，因此不直接计算 PSNR、SSIM。
2. Down-Up 是辅助定量对照，不应视为实际的去模糊复原算法。
3. NLM 主要用于噪声抑制，其处理效果受到参数 `h` 的影响。
4. 综合对比采用 Task 2 修复后的结果；受限逆滤波在综合图中展示的是选定的代表参数，并非全部参数的最优结果。
5. 所有实验结论仅针对当前测试图像及参数配置。