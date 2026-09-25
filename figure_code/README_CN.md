# 论文图片 Python 代码索引

本目录用于复核和修改论文全部正式图片。所有图均由 Python 生成，未进行手工
图像编辑。`figure_code_manifest.csv` 将每个 LaTeX 图片文件映射到其规范生成
代码；`TYPOGRAPHY.md` 说明本轮按图内标题、坐标轴标题、刻度、图例和标注
分类统一字号的规则。

## 本轮统一字号的生成入口

数据无关的 Figure 1--3 使用独立入口：

```bash
python empirical/run_problem_chain_figure.py --output-dir figures/problem_chain
python empirical/rebuild_submission_diagrams.py --output-root figures
PYTHONPATH=empirical/src python empirical/rebuild_plot_typography.py \
  --runs-root /path/to/accepted/empirical/runs --output-root figures
```

最后一个入口只读取已验收实验的汇总文件，不重新运行实验；直接更新正式图片。

## 其他图片

各图的规范源代码仍保留在 `empirical/src/entropy_crime_bike/` 的相应模块中。
复刻时应使用版本化配置和已验收实验输出，不要从论文 PDF 反向提取图片。
