# 论文图片 Python 代码索引

本目录用于复核和修改论文全部正式图片。所有图均由 Python 生成，未进行手工
图像编辑。`figure_code_manifest.csv` 将每个 LaTeX 图片文件映射到其规范生成
代码；`figure_python_code_snapshot.zip` 是本版本全部实验入口、绘图模块和配置
文件的轻量快照。

## Figure 3 与 Figure 10

本次版面修订使用独立入口：

```bash
cd /Users/dangdangdang/Documents/Entropy/entropy_crime_bike_overleaf_v2_delb_fano
PYTHONPATH=empirical/src /opt/anaconda3/bin/python3 \
  empirical/run_manuscript_figure_layout.py \
  --e02-run empirical/runs/20260718_172231_E02_spatial_temporal_panel \
  --e05-run empirical/runs/20260718_190356_E05_local_spatial_information
```

该入口只读取已验收的 E02/E05 CSV，不重新运行实验。每次成功运行会生成新的
`empirical/runs/<timestamp>_manuscript_figure_layout/`，保存输入哈希、输出哈希、
运行命令和面板像素尺寸。正式图片随后复制到 `figures/e11/`。

## 其他图片

其他图片的生成逻辑保留在 `empirical/src/entropy_crime_bike/` 的相应实验模块
中，运行入口位于 `empirical/run_*.py`。复刻时应使用版本化配置和原实验输出，
不要从论文 PDF 反向提取图片或手工修改图像。
