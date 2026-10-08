# 任务二修正版：原始 DICOM 传统去噪

本目录保留原 `task2` 不变，修正了以下问题：

- 直接读取严格配对的 LDCT/FDCT DICOM，不再读取带标题和白边的截图。
- 使用 DICOM 内置肺窗：窗位 `-600 HU`、窗宽 `1500 HU`。
- 两张图用同一 HU 截断范围归一化到 `uint8 0~255`。
- 所有滤波只作用于纯 `512×512` 医学像素。
- 实现真正的自适应中值滤波，而不是固定窗口 `medianBlur`。
- PSNR、SSIM 只比较医学图像矩阵，不包含标题、坐标轴或空白。
- 参数选择规则固定为：SSIM 优先，PSNR 作为同分决胜项。

## 唯一运行命令

```bash
/home/sanxi/.pyenv/versions/3.11.9/bin/python3.11 denoise_dicom.py
```

请从本目录运行：

```bash
cd /home/sanxi/DIP/experiment1/task2_1
/home/sanxi/.pyenv/versions/3.11.9/bin/python3.11 denoise_dicom.py
```

## 目录结构

```text
task2_1/
├── denoise_dicom.py
├── README.md
├── requirements.txt
├── images/                    # 纯 512×512 灰度图
├── figures/                   # 同尺度对比图、局部放大图和指标图
└── results/
    ├── parameter_search.csv   # 全部候选参数
    ├── metrics.csv            # 每种方法的最优指标
    ├── summary.txt            # 文本结论
    └── reproducibility.json   # 输入哈希、环境和参数
```

## 评价边界

当前只使用 L067 的第 1 张严格配对切片。它可以验证代码和比较这张切片上的算法，不能直接代表全部患者，也未把该切片宣称为已由医生确认的肺结节切片。
