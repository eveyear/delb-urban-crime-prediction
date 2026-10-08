# 论文图片 Python 代码索引

本目录用于复核和修改论文全部正式图片。所有图均由 Python 生成，未进行手工
图像编辑。`figure_code_manifest.csv` 将每个 LaTeX 图片文件映射到其规范生成
代码；`TYPOGRAPHY.md` 说明本轮按图内标题、坐标轴标题、刻度、图例和标注
分类统一字号的规则。

## 本轮统一字号的生成入口

数据无关的 Figure 1--3 使用统一入口。Figure 1 和 Figure 2 的唯一规范实现为
`empirical/src/entropy_crime_bike/editor_framework_figures.py`；Figure 1 单独生成
命令和 E11 工作流均调用此实现，不再维护另一份旧版图形。

```bash
python empirical/rebuild_submission_diagrams.py --output-root figures
PYTHONPATH=empirical/src python empirical/rebuild_plot_typography.py \
  --runs-root /path/to/accepted/empirical/runs --output-root figures
```

最后一个入口只读取已验收实验的汇总文件，不重新运行实验；直接更新正式图片。

2026 年 10 月 8 日的主文采用根目录 `applsci-4501207.tex`，正文已合并到该文件中；
`sections/` 保留为历史分节源文件，不作为最终主文编译入口。
`final_proof_manifest.json` 记录最终主文及九张 PDF 图片的校验值。正式校样的
PDF 图件直接保留出版方提供的版本；重绘 PDF 的生成日期等元数据可不同。
已在同一渲染设置下核验 Figure 1--2 与校样逐像素一致。

## 其他图片

各图的规范源代码仍保留在 `empirical/src/entropy_crime_bike/` 的相应模块中。
复刻时应使用版本化配置和已验收实验输出，不要从论文 PDF 反向提取图片。
