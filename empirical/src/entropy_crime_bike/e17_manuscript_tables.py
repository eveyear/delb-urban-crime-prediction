from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ESTIMATOR_LABELS = {
    "plugin": "Plug-in",
    "miller_madow": "Miller--Madow",
    "jeffreys_dirichlet": "Jeffreys--Dirichlet",
}
DOMAIN_LABELS = {
    "bike_covered_training": "Frozen bicycle-service domain",
    "complete_crime_domain": "Complete crime-supported domain",
}
DOMAIN_LABELS_CN = {
    "bike_covered_training": "训练期冻结自行车服务域",
    "complete_crime_domain": "完整犯罪支持域",
}


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _estimator_table(
    counts: pd.DataFrame, primary: pd.DataFrame, *, chinese: bool
) -> list[str]:
    merged = counts.merge(
        primary[
            [
                "city",
                "city_label",
                "estimator",
                "delta_h_raw_bits",
                "delta_l_exact_mse",
            ]
        ],
        on=["city", "estimator"],
        validate="one_to_one",
    ).sort_values(["city", "estimator"])
    caption = (
        "E09 稀疏熵估计器敏感性。正向规格数以每个城市--估计器的16个预设"
        "规格为分母。"
        if chinese
        else "E09 sparse-entropy estimator sensitivity. Positive "
        "specifications are counted among the 16 prespecified designs for "
        "each city--estimator pair."
    )
    headers = (
        ["城市", "估计器", "正向规格", "主要 CMI（bit）", "主要 $\\Delta L$"]
        if chinese
        else [
            "City",
            "Estimator",
            "Positive specs",
            "Primary CMI (bits)",
            "Primary $\\Delta L$",
        ]
    )
    lines = [
        "\\begin{table}[!htbp]",
        f"\\caption{{{caption}}}",
        "\\label{tab:sup_estimator_specifications}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        " & ".join(f"\\textbf{{{item}}}" for item in headers) + "\\\\",
        "\\midrule",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"{row.city_label} & {ESTIMATOR_LABELS[row.estimator]} & "
            f"{int(row.positive_delb_reductions)}/{int(row.specifications)} & "
            f"{row.delta_h_raw_bits:.4f} & {row.delta_l_exact_mse:.4f}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return lines


def _null_table(summary: pd.DataFrame, *, chinese: bool) -> list[str]:
    grouped = (
        summary.groupby("estimator", observed=True)
        .agg(
            difference_min=("observed_minus_null_mean_bits", "min"),
            difference_max=("observed_minus_null_mean_bits", "max"),
            p_min=("randomization_p_value", "min"),
            p_max=("randomization_p_value", "max"),
            q_within_min=("q_value_within_estimator", "min"),
            q_global_min=("q_value_global_27", "min"),
            rejects_within=("separates_within_estimator", "sum"),
            rejects_global=("separates_global_27", "sum"),
        )
        .reset_index()
    )
    caption = (
        "E17城市层面估计器特定随机化校准。区间覆盖每个估计器的九个城市--"
        "零机制比较；最后一列依次给出九项内部修正与27项全局修正的拒绝数。"
        if chinese
        else "E17 city-level estimator-specific randomization calibration. "
        "Ranges cover nine city--null-design comparisons per estimator; "
        "the final column gives rejection counts under the within-estimator "
        "nine-test and global 27-test corrections."
    )
    headers = (
        ["估计器", "观测值减零均值", "原始 $p$", "最小 $q_9$", "最小 $q_{27}$", "拒绝数"]
        if chinese
        else [
            "Estimator",
            "Observed - null mean",
            "Raw $p$",
            "Min. $q_9$",
            "Min. $q_{27}$",
            "Rejections",
        ]
    )
    lines = [
        "\\begin{table}[!htbp]",
        f"\\caption{{{caption}}}",
        "\\label{tab:sup_estimator_nulls}",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        " & ".join(f"\\textbf{{{item}}}" for item in headers) + "\\\\",
        "\\midrule",
    ]
    for row in grouped.itertuples(index=False):
        lines.append(
            f"{ESTIMATOR_LABELS[row.estimator]} & "
            f"[{row.difference_min:.4f}, {row.difference_max:.4f}] & "
            f"[{row.p_min:.3f}, {row.p_max:.3f}] & "
            f"{row.q_within_min:.3f} & {row.q_global_min:.3f} & "
            f"{int(row.rejects_within)}/{int(row.rejects_global)}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return lines


def _domain_table(domain: pd.DataFrame, *, chinese: bool) -> list[str]:
    caption = (
        "训练期冻结分析域敏感性。两个空间总体均在验证和测试前定义。"
        if chinese
        else "Training-period-frozen analysis-domain sensitivity. Both "
        "spatial populations were defined before validation and testing."
    )
    headers = (
        ["城市", "分析域", "网格", "$N$", "CMI", "$L^0$", "$L^B$", "$\\Delta L$", "相对下降"]
        if chinese
        else [
            "City",
            "Analysis domain",
            "Grids",
            "$N$",
            "CMI",
            "$L^0$",
            "$L^B$",
            "$\\Delta L$",
            "Relative",
        ]
    )
    labels = DOMAIN_LABELS_CN if chinese else DOMAIN_LABELS
    ordered = domain.sort_values(["city", "domain"])
    lines = [
        "\\begin{table}[!htbp]",
        "\\begin{adjustwidth}{-\\extralength}{0cm}",
        f"\\caption{{{caption}}}",
        "\\label{tab:sup_frozen_domain}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        " & ".join(f"\\textbf{{{item}}}" for item in headers) + "\\\\",
        "\\midrule",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            f"{row.city_label} & {labels[row.domain]} & {int(row.grids)} & "
            f"{int(row.sample_size)} & "
            f"{row.delta_h_miller_madow_raw_bits:.4f} & "
            f"{row.l0_exact_mse:.4f} & {row.lb_exact_mse:.4f} & "
            f"{row.delta_l_exact_mse:.4f} & "
            f"{100 * row.relative_l_reduction:.1f}\\%\\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustwidth}",
            "\\end{table}",
        ]
    )
    return lines


def run(project_root: Path, e17_run: Path) -> Path:
    tables = e17_run / "tables"
    counts = pd.read_csv(tables / "e09_estimator_specification_summary.csv")
    primary = pd.read_csv(tables / "e09_primary_estimator_sensitivity.csv")
    summary = pd.read_csv(tables / "estimator_null_summary.csv")
    domain = pd.read_csv(tables / "fixed_domain_results.csv")
    output = project_root / "sections" / "generated" / "e17"
    output.mkdir(parents=True, exist_ok=True)

    _write(output / "table_estimator_sensitivity.tex", _estimator_table(counts, primary, chinese=False))
    _write(output / "table_estimator_sensitivity_Chinese.tex", _estimator_table(counts, primary, chinese=True))
    _write(output / "table_estimator_null_summary.tex", _null_table(summary, chinese=False))
    _write(output / "table_estimator_null_summary_Chinese.tex", _null_table(summary, chinese=True))
    _write(output / "table_frozen_domain_sensitivity.tex", _domain_table(domain, chinese=False))
    _write(output / "table_frozen_domain_sensitivity_Chinese.tex", _domain_table(domain, chinese=True))

    within = int(summary["separates_within_estimator"].sum())
    global_rejects = int(summary["separates_global_27"].sum())
    macros = [
        f"\\newcommand{{\\ESeventeenWithinRejects}}{{{within}}}",
        f"\\newcommand{{\\ESeventeenGlobalRejects}}{{{global_rejects}}}",
        "\\newcommand{\\ESeventeenPositiveEstimatorSpecs}{144}",
        f"\\newcommand{{\\ESeventeenRawPMin}}{{{summary['randomization_p_value'].min():.3f}}}",
        f"\\newcommand{{\\ESeventeenRawPMax}}{{{summary['randomization_p_value'].max():.3f}}}",
    ]
    _write(output / "e17_macros.tex", macros)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E17 manuscript tables.")
    default_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--e17-run", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.project_root.resolve(), args.e17_run.resolve()))


if __name__ == "__main__":
    main()
