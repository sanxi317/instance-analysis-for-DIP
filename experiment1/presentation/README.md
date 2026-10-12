# DIP案例一：5分钟课堂汇报

用LaTeX Beamer制作，16:9。主讲10页，备用6页。

每页都直接使用普通frame环境，标题写在环境内部：

```tex
\begin{frame}
\frametitle{标题}
% 本页内容
\end{frame}
```

可直接在相应frame内部修改文字、公式与图片。

- `case1_slides.tex`：默认编译10页主讲。
- `case1_with_backup.tex`：编译主讲与备用页，共16页。
- `讲稿_5分钟.md`：逐页讲稿、300秒时间分配和概念补充。
- `assets/`：现有实验图及相同坐标局部图，已随源码提供。
- `asset_manifest.json`：图像来源、SHA256与裁剪坐标。

## 编译

需要XeLaTeX、Beamer、ctex、fontspec、amsmath与booktabs（常规TeX Live/MiKTeX组件）。使用TeX Gyre Heros西文字体；中文优先Noto Sans CJK SC，缺失时回退TeX Live的Fandol黑体。

在本目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build case1_slides.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build case1_with_backup.tex
```

也可运行 `bash build.sh`。没有latexmk时，可在本目录对同一入口连续运行两次 `xelatex`。

主讲时在第10页“最终结论”结束。备用页解释均值/高斯、自适应中值、小波软阈值、双线性插值、主要来源和NLM偏差--方差模型，不计入5分钟。第5页与第9页含关键代码，代码采用等宽文本排版，仍保持普通frame环境。

## 数据与图像

主讲数值使用已核对的 `task4/results/metrics.csv`。

去噪与插值恢复页统一使用512×512 FDCT参考；插值恢复实际流程为512→128→512。理论页解释NLM和双三次在本例取得较好效果的可能机制，不作普遍最优性证明，也不保留未经本次确认的顺序实验结论。

第6页写明指标全称：PSNR为Peak Signal-to-Noise Ratio（峰值信噪比），SSIM为Structural Similarity Index（结构相似性指数）。

## 关键代码来源

- NLM调用来自 `task2/denoise_dicom.py`；h=4是本次参数搜索选中值，图像块7×7、搜索窗21×21。
- 双三次恢复调用来自 `task3/main.py`；展示区域降采样到128×128，以及双三次恢复到512×512两步。仅为投影阅读重新换行，不改变操作。
- `ldct`为统一肺窗uint8输入；`nlm_image`为任务二保存的NLM图像。课堂页摘录不是独立完整程序。

## 理论范围

相似块的加权平均有助于抑制随机波动并减少结构混合；偏差--方差推导只在固定权重、独立零均值同方差噪声的简化模型下成立。真实NLM权重取决于带噪图像，CT噪声也可能相关。

双三次使用4×4邻域的三次核，可描述比线性更复杂的局部变化；边缘处仍可能过冲或振铃，不能生成真实缺失细节。详见讲稿与备用页。

局部图来自原图坐标 `(80, 150, 240, 310)`，仅作组织边缘/纹理比较，没有病灶标注。图像没有调节对比度、重新去噪或重新插值。

在原始experiment1项目中，可运行 `python prepare_assets.py` 重新复制图像和裁剪；编译时无需运行此脚本，源码包已包含全部图像。

## 教材

Gonzalez与Woods，《数字图像处理》/Digital Image Processing，第4版：第2.4节采样与插值，第3.4–3.5节空间滤波，第5.3节自适应中值，第6.10节小波，第4.5节重采样。NLM、软阈值和SSIM同时列有原始文献作为补充，见备用A5及讲稿。
