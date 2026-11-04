# DIP案例一：完整课堂讲解

用LaTeX Beamer制作，16:9。统一21页，不限定5分钟；概念、公式、实验结果和关键代码按讲解顺序整合。

每页都直接使用普通frame环境，标题写在环境内部：

```tex
\begin{frame}
\frametitle{标题}
% 本页内容
\end{frame}
```

可直接在相应frame内部修改文字、公式与图片。

- `case1_slides.tex`：唯一完整入口，直接编译21页。
- `讲稿_完整讲解.md`：21页逐页讲稿、概念补充和可复制关键代码。
- `assets/`：现有实验图及相同坐标局部图，已随源码提供。
- `asset_manifest.json`：图像来源、SHA256与裁剪坐标。
- `order_values.tex`：从本次保存CSV生成的显示数值。
- `experiments/run_order_comparison.py`：本次实际运行的直接4倍与顺序实验代码。
- `experiments/results/`：原始输出PNG、共同参考、CSV、执行清单和独立核验记录。

## 编译

需要XeLaTeX、Beamer、ctex、fontspec、amsmath与booktabs（常规TeX Live/MiKTeX组件）。使用TeX Gyre Heros西文字体；中文优先Noto Sans CJK SC，缺失时回退TeX Live的Fandol黑体。

在本目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build case1_slides.tex
```

也可运行 `bash build.sh`。没有latexmk时，可在本目录对同一入口连续运行两次 `xelatex`。

讲解依次为：案例与预处理、PSNR/SSIM、五种去噪及数学原理、去噪结果、插值原理与两组实验、顺序对照及数学解释、最终结论、参考来源。第15页是512→2048直接4倍插值；第16–17页是原512→128→512补充恢复实验；第18–19页完整展示两种处理顺序及其理论。第10、17、18页含关键代码。全部内容采用连续页码。

## 数据与图像

原尺寸去噪与补充恢复数值来自已核对的 `task4/results/metrics.csv`；直接4倍和顺序数值来自本次执行的 `experiments/results/metrics.csv`。

原尺寸去噪及补充恢复实验以512×512 FDCT为参考。本次按题目直接512→2048放大，三种插值和两条顺序均使用同一双三次放大的FDCT作为尺寸对齐参考；这是构造参考，不是真实采集的高分辨率真值，不能与原尺寸指标跨页直接比较，参考核也可能影响排序。

新直接放大结果：最近邻33.580 dB/0.93602；双线性38.790 dB/0.97025；双三次39.242 dB/0.96948。双三次PSNR最高，双线性SSIM略高，不能称双三次两项都最优。NLM与双三次顺序对照：A先去噪为39.242 dB/0.96948；B先插值为38.465 dB/0.96402。结论只适用本次单切片和数值参数；两种尺寸下NLM窗口对应的物理范围不同，插值还改变噪声统计。

第4页写明指标全称：PSNR为Peak Signal-to-Noise Ratio（峰值信噪比），SSIM为Structural Similarity Index（结构相似性指数）。

## 关键代码来源

- NLM调用来自 `task2/denoise_dicom.py`；h=4是本次参数搜索选中值，图像块7×7、搜索窗21×21。
- 直接4倍与顺序调用来自本次执行的 `experiments/run_order_comparison.py`；两种顺序的输入、输出尺寸、参考和数值参数相同。
- 第17页的512→128→512恢复代码来自 `task3/main.py`，对应指标在第16页。
- `ldct`为统一肺窗uint8输入。完整可复现脚本在experiments/，课堂页代码只是摘录。

## 复现本次顺序实验

需NumPy、OpenCV、Pillow、scikit-image。在本目录运行：

```bash
python experiments/run_order_comparison.py
```

源码包已包含统一肺窗PNG，脚本会保存无标题2048×2048结果、CSV、SHA256、命令、环境和参数；若原项目的DICOM存在，还会逐像素核对统一肺窗输入和配对几何。本次使用Python3.11.9、OpenCV5.0.0、NumPy2.4.6、scikit-image0.26.0；结果由保存PNG独立重算一致，见independent_verification.json。原task5脚本及旧结果未覆盖。

## 理论范围

相似块的加权平均有助于抑制随机波动并减少结构混合；一般偏差--方差恒等式适用于有限二阶矩的估计。具体的加权误差公式使用固定权重、独立零均值同方差噪声的简化模型。真实NLM权重取决于带噪图像，CT噪声也可能相关。

双三次使用4×4邻域的三次核，可描述比线性更复杂的局部变化；边缘处仍可能过冲或振铃，不能生成真实缺失细节。第14页给出分段核和负权重过冲例子。

将浮点插值写为U，独立输入噪声的协方差经插值变为σ_n²UUᵀ；输出噪声一般相关。NLM权重又随图像变化，因此与插值通常不可交换。第19页解释这一机制，并限定模型与当前数值参数，不能从它推出普遍最优顺序。

原512局部图坐标为 `(80, 150, 240, 310)`；本次2048图对应 `(320, 600, 960, 1240)`。仅作组织边缘/纹理比较，没有病灶标注。展示裁剪不改变结果像素、不调节对比度；去噪和插值由可复现脚本执行，所有原始结果已保存。

在原始experiment1项目中可运行 `python prepare_assets.py` 更新旧512图片；运行本次实验脚本更新新2048局部图。编译时无需重跑，源码包已包含全部图像。

## 教材

Gonzalez与Woods，《数字图像处理》/Digital Image Processing，第4版：第2.4节采样与插值，第3.4–3.5节空间滤波，第5.3节自适应中值，第6.10节小波，第4.5节重采样。NLM、软阈值和SSIM同时列有原始文献作为补充，见第21页及讲稿。
