from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml


CITY_LABELS = {
    "DC": "Washington, DC",
    "NY": "New York City",
    "VAN": "Vancouver",
}
TARGET_LABELS = {
    "binary_occurrence": "Binary occurrence",
    "three_level_count": "Three-level count",
}
SUPPORT_LABELS = {
    "mean_support_atoms_per_observation": r"$K_{\mathrm{obs}}/N$",
    "mean_singleton_observation_share": r"$S_1$",
    "mean_effective_df_per_observation": r"$\nu/N$",
}


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return config


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _read_pointer(empirical: Path, relative: str) -> Path:
    pointer = empirical / relative
    run = Path(pointer.read_text(encoding="utf-8").strip())
    if not run.is_absolute():
        candidate = pointer.parent / run
        run = candidate if candidate.exists() else empirical / "runs" / run
    run = run.resolve()
    if not run.exists():
        raise FileNotFoundError(run)
    manifest = run / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
    if status != "complete":
        raise ValueError(f"Source run is not complete: {run}")
    return run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _simulation_summary(e12: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    estimator = pd.read_csv(e12 / "tables" / "estimator_summary.csv")
    randomization = pd.read_csv(
        e12 / "tables" / "randomization_summary.csv"
    )
    null_estimator = estimator.loc[np.isclose(estimator["effect_strength"], 0)]
    rows = []
    macros: dict[str, str] = {}
    for estimator_id, label, prefix in [
        ("plugin", "Plugin", "SimPlugin"),
        ("miller_madow", "Miller--Madow", "SimMM"),
    ]:
        bias_column = f"bias_{estimator_id}_bits"
        bias = null_estimator[bias_column]
        null_randomization = randomization.loc[
            randomization["evidence_metric"].eq("type_i_error")
            & randomization["estimator"].eq(estimator_id)
        ]
        alternative = randomization.loc[
            randomization["evidence_metric"].eq("power")
            & randomization["estimator"].eq(estimator_id)
        ]
        row = {
            "Estimator": label,
            "Mean null bias": float(bias.mean()),
            "Null bias minimum": float(bias.min()),
            "Null bias maximum": float(bias.max()),
            "Mean null center": float(null_randomization["mean_null_bits"].mean()),
            "Type-I rejection rate": float(
                null_randomization["rejection_rate"].mean()
            ),
            "Alternative power": float(alternative["rejection_rate"].mean()),
        }
        rows.append(row)
        macros[f"{prefix}NullBias"] = f'{row["Mean null bias"]:.4f}'
        macros[f"{prefix}NullCenter"] = f'{row["Mean null center"]:.4f}'
        macros[f"{prefix}TypeI"] = f'{row["Type-I rejection rate"]:.3f}'
        macros[f"{prefix}Power"] = f'{row["Alternative power"]:.3f}'
    return pd.DataFrame(rows), macros


def _support_summary(e13: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    relationships = pd.read_csv(e13 / "tables" / "spearman_relationships.csv")
    regressions = pd.read_csv(e13 / "tables" / "fixed_effect_regressions.csv")
    rows = []
    for predictor in SUPPORT_LABELS:
        plugin = relationships.loc[
            relationships["source"].eq("E12_population_null")
            & relationships["outcome"].eq("bias_plugin_bits")
            & relationships["predictor"].eq(predictor),
            "spearman_rho",
        ].item()
        mm = relationships.loc[
            relationships["source"].eq("E12_population_null")
            & relationships["outcome"].eq("bias_miller_madow_bits")
            & relationships["predictor"].eq(predictor),
            "spearman_rho",
        ].item()
        empirical_null = relationships.loc[
            relationships["source"].eq("E13_empirical_local")
            & relationships["outcome"].eq("null_mean_delta_h_bits")
            & relationships["predictor"].eq(
                predictor.replace(
                    "mean_", "mean_null_", 1
                )
                if predictor.startswith("mean_support")
                else (
                    "mean_null_singleton_observation_share"
                    if predictor == "mean_singleton_observation_share"
                    else "mean_null_effective_df_per_observation"
                )
            ),
            "spearman_rho",
        ]
        change_predictor = (
            "null_minus_observed_"
            + predictor.removeprefix("mean_")
        )
        empirical_change = relationships.loc[
            relationships["source"].eq("E13_empirical_local")
            & relationships["outcome"].eq("observed_minus_null_mean_bits")
            & relationships["predictor"].eq(change_predictor),
            "spearman_rho",
        ]
        rows.append(
            {
                "Support metric": SUPPORT_LABELS[predictor],
                "Simulation plugin bias rho": float(plugin),
                "Simulation MM bias rho": float(mm),
                "Empirical null-mean rho": float(empirical_null.item()),
                "Empirical contrast rho": float(empirical_change.item()),
            }
        )
    combined = regressions.loc[
        regressions["model_id"].eq("null_mean_combined")
        & regressions["predictor"].eq(
            "mean_null_effective_df_per_observation"
        )
    ].iloc[0]
    macros = {
        "SupportCombinedNuCoefficient": f'{combined["coefficient"]:.4f}',
        "SupportCombinedNuSE": f'{combined["clustered_standard_error"]:.4f}',
        "SupportCombinedNullRSquared": f'{combined["r_squared"]:.3f}',
    }
    return pd.DataFrame(rows), macros


def _fano_summary(e14: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    points = pd.read_csv(e14 / "tables" / "fano_city_results.csv")
    contrasts = pd.read_csv(e14 / "tables" / "fano_mobility_contrasts.csv")
    oracle = pd.read_csv(e14 / "tables" / "fano_oracle_results.csv")
    primary = points.loc[points["estimator"].eq("miller_madow")]
    contrast_primary = contrasts.loc[
        contrasts["estimator"].eq("miller_madow")
    ]
    rows = []
    for (city, target), group in primary.groupby(["city", "target_id"]):
        base = group.loc[group["information_set"].eq("baseline")].iloc[0]
        bike = group.loc[group["information_set"].eq("bicycle_aware")].iloc[0]
        contrast = contrast_primary.loc[
            contrast_primary["city"].eq(city)
            & contrast_primary["target_id"].eq(target)
        ].iloc[0]
        rows.append(
            {
                "City": CITY_LABELS[city],
                "Target": TARGET_LABELS[target],
                "Baseline floor": float(base["fano_error_floor"]),
                "Bicycle floor": float(bike["fano_error_floor"]),
                "Floor reduction": float(contrast["fano_floor_reduction"]),
                "Floor CI low": float(
                    contrast["fano_floor_reduction_ci_low"]
                ),
                "Floor CI high": float(
                    contrast["fano_floor_reduction_ci_high"]
                ),
                "Baseline test error": float(base["test_modal_error"]),
                "Bicycle test error": float(bike["test_modal_error"]),
                "Baseline unseen": int(base["test_unseen_state_observations"]),
                "Bicycle unseen": int(bike["test_unseen_state_observations"]),
            }
        )
    result = pd.DataFrame(rows)
    macros = {
        "FanoOracleRows": str(len(oracle)),
        "FanoEstimatedViolations": str(
            int(primary["estimated_bound_violation"].sum())
        ),
        "FanoFloorReductionMin": f'{result["Floor reduction"].min():.4f}',
        "FanoFloorReductionMax": f'{result["Floor reduction"].max():.4f}',
        "FanoNegativeTestContrasts": str(
            int(
                (
                    result["Baseline test error"]
                    - result["Bicycle test error"]
                ).lt(0).sum()
            )
        ),
        "FanoTotalContrasts": str(len(result)),
    }
    return result, macros


def _write_macros(path: Path, macros: dict[str, str]) -> None:
    lines = [
        "% Generated by S7 Python synthesis; do not edit manually.",
        *[_macro(name, value) for name, value in sorted(macros.items())],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_simulation_table(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{E12 finite-sample CMI simulation summary. Null bias and null centers are in bits. Size is the mean rejection rate under population CMI equal to zero; power averages the positive-CMI designs.}",
        r"\label{tab:e12_simulation_summary}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Estimator} & \textbf{Mean null bias} & \textbf{Mean null center} & \textbf{Size} & \textbf{Power}\\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.Estimator} & {row[1]:.4f} & {row[4]:.4f} & "
            f"{row[5]:.3f} & {row[6]:.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_support_table(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Association of support diagnostics with CMI estimator behavior. Entries are Spearman correlations. The first two columns use E12 population-null simulations; the last two use 1,017 E13 city-grid-null summaries.}",
        r"\label{tab:e13_support_relationships}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Plugin bias} & \textbf{MM bias} & \textbf{Null mean} & \textbf{Observed--null}\\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row[0]} & {row[1]:.3f} & {row[2]:.3f} & "
            f"{row[3]:.3f} & {row[4]:.3f}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustwidth}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_fano_table(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{E14 Miller--Madow Fano floors and held-out statewise modal-classifier errors. The interval refers to the baseline-minus-bicycle floor reduction. Unseen counts are test observations assigned through the pretest global-mode fallback.}",
        r"\label{tab:e14_fano_results}",
        r"\begin{adjustwidth}{-\extralength}{0cm}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{City} & \textbf{Target} & \textbf{Base floor} & \textbf{Bike floor} & \textbf{Reduction [95\% CI]} & \textbf{Test error base/bike} & \textbf{Unseen base/bike}\\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        city = latex_escape(row[0])
        target = latex_escape(row[1])
        lines.append(
            f"{city} & {target} & {row[2]:.3f} & {row[3]:.3f} & "
            f"{row[4]:.3f} [{row[5]:.3f}, {row[6]:.3f}] & "
            f"{row[7]:.3f}/{row[8]:.3f} & "
            f"{row[9]:,}/{row[10]:,}\\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustwidth}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> Path:
    start = time.perf_counter()
    config = _load_yaml(config_path)
    empirical = config_path.resolve().parents[1]
    project = empirical.parent
    source_runs = {
        name: _read_pointer(empirical, relative)
        for name, relative in config["source_run_pointers"].items()
    }
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_S7_synthesis"
    run_dir = empirical / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    generated = project / str(config["generated_tex_directory"])
    figure_destination = project / str(
        config["manuscript_figure_directory"]
    )
    generated.mkdir(parents=True, exist_ok=True)
    figure_destination.mkdir(parents=True, exist_ok=True)

    simulation, simulation_macros = _simulation_summary(source_runs["e12"])
    support, support_macros = _support_summary(source_runs["e13"])
    fano, fano_macros = _fano_summary(source_runs["e14"])
    macros = {**simulation_macros, **support_macros, **fano_macros}
    _write_macros(generated / "s7_macros.tex", macros)
    _write_simulation_table(
        generated / "table_e12_simulation.tex", simulation
    )
    _write_support_table(
        generated / "table_e13_support.tex", support
    )
    _write_fano_table(generated / "table_e14_fano.tex", fano)

    figure_records = []
    for specification in config["figures"]:
        source_key, relative = str(specification["source"]).split("/", 1)
        source = source_runs[source_key] / relative
        destination = figure_destination / str(
            specification["destination"]
        )
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"Figure copy failed hash check: {source}")
        figure_records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": _sha256(destination),
            }
        )
    pd.DataFrame(figure_records).to_csv(
        run_dir / "figure_manifest.csv", index=False
    )
    simulation.to_csv(run_dir / "e12_summary.csv", index=False)
    support.to_csv(run_dir / "e13_summary.csv", index=False)
    fano.to_csv(run_dir / "e14_summary.csv", index=False)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as out:
        yaml.safe_dump(config, out, sort_keys=False, allow_unicode=True)
    (run_dir / "command.txt").write_text(
        "PYTHONPATH=src python run_s7_manuscript_synthesis.py "
        "--config config/s7.yaml\n",
        encoding="utf-8",
    )

    expected = config["acceptance"]
    generated_files = sorted(generated.glob("*.tex"))
    checks = {
        "source_runs": len(source_runs)
        == int(expected["expected_source_runs"]),
        "generated_tex_files": len(generated_files)
        == int(expected["expected_generated_tex_files"]),
        "copied_figures": len(figure_records)
        == int(expected["expected_copied_figures"]),
        "fano_rows": len(fano) == int(expected["expected_fano_rows"]),
        "oracle_rows": int(macros["FanoOracleRows"])
        == int(expected["expected_oracle_rows"]),
        "estimated_fano_violations": int(
            macros["FanoEstimatedViolations"]
        )
        == int(expected["expected_estimated_fano_violations"]),
        "all_figures_pdf": all(
            Path(record["destination"]).suffix == ".pdf"
            for record in figure_records
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"S7 synthesis acceptance failed: {checks}")
    manifest = {
        "run_id": run_id,
        "stage": str(config["stage_id"]),
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - start,
        "source_runs": {key: str(value) for key, value in source_runs.items()},
        "outputs": {
            "generated_tex": [str(path) for path in generated_files],
            "figures": figure_records,
        },
        "checks": checks,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize accepted E12--E15 outputs for Version 2."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "s7.yaml",
    )
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
