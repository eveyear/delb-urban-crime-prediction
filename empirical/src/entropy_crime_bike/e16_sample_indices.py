from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml


CITY_ORDER = ["DC", "NY", "VAN"]
RESOLUTIONS = [500, 1000, 2000]


def _load_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(master: int, city: str, temporal: str, replicate: int) -> int:
    label = f"{city}|{temporal}|{replicate}"
    return int((master + zlib.crc32(label.encode("utf-8"))) % 2**32)


def _season(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("e16_sample_indices")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in [
        logging.FileHandler(run_dir / "logs" / "e16_s16_3.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _environment(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"timestamp={datetime.now().isoformat()}",
                f"python={sys.version.replace(os.linesep, ' ')}",
                f"executable={sys.executable}",
                f"platform={platform.platform()}",
                f"pandas={pd.__version__}",
                f"numpy={np.__version__}",
                f"pyarrow={pyarrow.__version__}",
                f"matplotlib={plt.matplotlib.__version__}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def balanced_stratified_order(
    blocks: pd.DataFrame, *, seed: int
) -> pd.DataFrame:
    """Create one randomized prefix order with approximately balanced strata.

    Each stratum is shuffled independently. A normalized within-stratum rank
    then interleaves the strata, so every prefix remains close to the full
    year-by-season composition while sample sizes are exact and nested.
    """
    required = {"block_id", "stratum_year", "stratum_season"}
    if not required.issubset(blocks.columns):
        raise ValueError(f"Missing block fields: {required-set(blocks.columns)}")
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in blocks.groupby(
        ["stratum_year", "stratum_season"], sort=True
    ):
        part = part.copy()
        permutation = rng.permutation(len(part))
        part = part.iloc[permutation].reset_index(drop=True)
        part["within_stratum_rank"] = np.arange(1, len(part) + 1)
        part["balance_score"] = (
            (part["within_stratum_rank"] - 0.5) / len(part)
            + rng.uniform(0, 1e-9, len(part))
        )
        parts.append(part)
    ordered = pd.concat(parts, ignore_index=True).sort_values(
        ["balance_score", "stratum_year", "stratum_season", "block_id"]
    )
    ordered["selection_rank"] = np.arange(1, len(ordered) + 1)
    return ordered.reset_index(drop=True)


def make_block_universe(
    period_starts: pd.DatetimeIndex, *, temporal: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(period_starts).unique()))
    if temporal == "day":
        periods_per_block = 7
        prefix = "D7"
        period_duration_days = 1
    elif temporal == "week":
        periods_per_block = 4
        prefix = "W4"
        period_duration_days = 7
    else:
        raise ValueError(temporal)
    complete_count = len(dates) // periods_per_block
    retained_count = complete_count * periods_per_block
    records = []
    period_records = []
    for block_index in range(complete_count):
        members = dates[
            block_index * periods_per_block : (block_index + 1) * periods_per_block
        ]
        expected_step = pd.Timedelta(days=period_duration_days)
        if len(members) > 1 and not (np.diff(members.values) == expected_step.to_timedelta64()).all():
            raise AssertionError("Non-consecutive period in sampling block.")
        block_id = f"{prefix}_{block_index + 1:03d}"
        block_start = members[0]
        block_end = members[-1] + pd.Timedelta(days=period_duration_days - 1)
        midpoint = block_start + (block_end - block_start) / 2
        records.append(
            {
                "temporal_resolution": temporal,
                "block_id": block_id,
                "block_index": block_index + 1,
                "block_start": block_start,
                "block_end": block_end,
                "periods_in_block": periods_per_block,
                "stratum_year": midpoint.year,
                "stratum_season": _season(midpoint.month),
            }
        )
        period_records.extend(
            {
                "temporal_resolution": temporal,
                "period_start": value,
                "block_id": block_id,
                "included_in_primary_universe": True,
                "residual_reason": "",
            }
            for value in members
        )
    for value in dates[retained_count:]:
        period_records.append(
            {
                "temporal_resolution": temporal,
                "period_start": value,
                "block_id": "",
                "included_in_primary_universe": False,
                "residual_reason": "incomplete_terminal_sampling_block",
            }
        )
    return pd.DataFrame(records), pd.DataFrame(period_records)


def sample_prefixes(
    blocks: pd.DataFrame,
    *,
    fractions: list[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = balanced_stratified_order(blocks, seed=seed)
    membership = []
    summaries = []
    total = len(ordered)
    for fraction in fractions:
        count = total if fraction == 1 else max(1, int(round(total * fraction)))
        chosen = ordered.iloc[:count].copy()
        membership.extend(
            {
                "fraction": float(fraction),
                "block_id": row.block_id,
                "selection_rank": int(row.selection_rank),
            }
            for row in chosen.itertuples(index=False)
        )
        summaries.append(
            {
                "fraction": float(fraction),
                "selected_blocks": count,
                "available_blocks": total,
                "realized_block_fraction": count / total,
                "first_period": chosen["block_start"].min(),
                "last_period": chosen["block_end"].max(),
            }
        )
    return pd.DataFrame(membership), pd.DataFrame(summaries)


def expanding_windows(
    periods: pd.DatetimeIndex, fractions: list[float], *, temporal: str
) -> pd.DataFrame:
    values = pd.DatetimeIndex(sorted(pd.to_datetime(periods).unique()))
    duration = 1 if temporal == "day" else 7
    records = []
    for fraction in fractions:
        count = len(values) if fraction == 1 else max(1, int(round(len(values) * fraction)))
        selected = values[:count]
        records.append(
            {
                "temporal_resolution": temporal,
                "fraction": float(fraction),
                "available_periods": len(values),
                "selected_periods": count,
                "realized_period_fraction": count / len(values),
                "window_start": selected[0],
                "window_end_period_start": selected[-1],
                "window_end": selected[-1] + pd.Timedelta(days=duration - 1),
            }
        )
    return pd.DataFrame(records)


def _panel_periods(data_dir: Path, city: str, temporal: str) -> pd.DatetimeIndex:
    files = sorted(
        (
            data_dir
            / "common_core"
            / "resolution=500m"
            / f"temporal={temporal}"
            / f"city={city}"
        ).rglob("*.parquet")
    )
    if not files:
        raise FileNotFoundError((data_dir, city, temporal))
    dates = []
    for path in files:
        dates.append(pd.read_parquet(path, columns=["date"])["date"])
    return pd.DatetimeIndex(pd.concat(dates, ignore_index=True).unique()).sort_values()


def _nested_check(membership: pd.DataFrame, fractions: list[float]) -> bool:
    previous: set[str] = set()
    for fraction in sorted(fractions):
        current = set(
            membership.loc[membership["fraction"].eq(fraction), "block_id"].astype(str)
        )
        if not previous.issubset(current):
            return False
        previous = current
    return True


def _plot_qc(summary: pd.DataFrame, balance: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for temporal, marker in [("day", "o"), ("week", "s")]:
        part = summary.loc[
            summary["temporal_resolution"].eq(temporal)
            & summary["resolution_m"].eq(1000)
            & summary["replicate"].gt(0)
        ]
        grouped = part.groupby("fraction", as_index=False)["realized_block_fraction"].mean()
        axes[0].plot(grouped["fraction"], grouped["realized_block_fraction"], marker=marker, label=temporal)
    axes[0].plot([0, 1], [0, 1], color="0.3", linestyle="--", linewidth=1, label="target")
    axes[0].set(xlabel="Target sample fraction", ylabel="Realized block fraction", title="Block-sample size")
    balance_plot = balance.loc[balance["fraction"].lt(1)].copy()
    axes[1].boxplot(
        [
            balance_plot.loc[balance_plot["temporal_resolution"].eq(value), "maximum_absolute_stratum_share_deviation"]
            for value in ["day", "week"]
        ],
        tick_labels=["day", "week"],
        showfliers=False,
    )
    axes[1].set(ylabel="Maximum absolute share deviation", title="Year–season balance")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "e16_s16_3_sample_index_qc.pdf", bbox_inches="tight")
    fig.savefig(output / "e16_s16_3_sample_index_qc.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    empirical_root = config_path.parent.parent
    config = _load_yaml(config_path)
    s16_2_run = Path(
        (empirical_root / "runs" / "latest_e16_s16_2_run.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    status = json.loads((s16_2_run / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("stage") != "S16.2":
        raise ValueError("S16.3 requires accepted S16.2.")
    data_dir = Path(status["data_directory"])
    timezone = ZoneInfo("Asia/Shanghai")
    run_id = datetime.now(timezone).strftime("%Y%m%d_%H%M%S") + "_E16_S16_3_sample_indices"
    run_dir = empirical_root / "runs" / run_id
    for directory in [run_dir / "logs", run_dir / "tables", run_dir / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir)
    (run_dir / "command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    _environment(run_dir / "environment.txt")
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    fractions = [float(value) for value in config["sample_size_scales"]["fractions"]]
    repetitions = int(config["randomness"]["sample_fraction_repetitions"])
    master_seed = int(config["randomness"]["master_seed"])
    inventory = pd.read_csv(s16_2_run / "tables" / "multiscale_panel_inventory.csv")

    universe_parts = []
    period_map_parts = []
    membership_parts = []
    summary_parts = []
    balance_parts = []
    seed_records = []
    expanding_parts = []

    for city in CITY_ORDER:
        for temporal in ["day", "week"]:
            periods = _panel_periods(data_dir, city, temporal)
            blocks, period_map = make_block_universe(periods, temporal=temporal)
            blocks.insert(0, "city", city)
            period_map.insert(0, "city", city)
            universe_parts.append(blocks)
            period_map_parts.append(period_map)
            full_strata = (
                blocks.groupby(["stratum_year", "stratum_season"]).size() / len(blocks)
            )
            for replicate in range(1, repetitions + 1):
                seed = _seed(master_seed, city, temporal, replicate)
                seed_records.append(
                    {"city": city, "temporal_resolution": temporal, "replicate": replicate, "seed": seed}
                )
                membership, summary = sample_prefixes(
                    blocks, fractions=[value for value in fractions if value < 1], seed=seed
                )
                membership.insert(0, "replicate", replicate)
                membership.insert(0, "temporal_resolution", temporal)
                membership.insert(0, "city", city)
                membership_parts.append(membership)
                for fraction in sorted(value for value in fractions if value < 1):
                    selected_ids = membership.loc[membership["fraction"].eq(fraction), "block_id"]
                    selected = blocks.loc[blocks["block_id"].isin(selected_ids)]
                    selected_strata = selected.groupby(["stratum_year", "stratum_season"]).size() / len(selected)
                    aligned = full_strata.to_frame("full").join(selected_strata.rename("sample"), how="outer").fillna(0)
                    balance_parts.append(
                        {
                            "city": city,
                            "temporal_resolution": temporal,
                            "replicate": replicate,
                            "fraction": fraction,
                            "maximum_absolute_stratum_share_deviation": float((aligned["sample"] - aligned["full"]).abs().max()),
                        }
                    )
                for resolution in RESOLUTIONS:
                    grids = int(
                        inventory.loc[
                            inventory["city"].eq(city)
                            & inventory["resolution_m"].eq(resolution)
                            & inventory["temporal_resolution"].eq(temporal),
                            "grids",
                        ].iloc[0]
                    )
                    local = summary.copy()
                    local.insert(0, "replicate", replicate)
                    local.insert(0, "resolution_m", resolution)
                    local.insert(0, "temporal_resolution", temporal)
                    local.insert(0, "city", city)
                    periods_per_block = int(blocks["periods_in_block"].iloc[0])
                    local["selected_periods"] = local["selected_blocks"] * periods_per_block
                    local["grids"] = grids
                    local["nominal_panel_rows"] = local["selected_periods"] * grids
                    summary_parts.append(local)
            full_membership = blocks[["block_id"]].copy()
            full_membership["fraction"] = 1.0
            full_membership["selection_rank"] = blocks["block_index"]
            full_membership.insert(0, "replicate", 0)
            full_membership.insert(0, "temporal_resolution", temporal)
            full_membership.insert(0, "city", city)
            membership_parts.append(full_membership)
            for resolution in RESOLUTIONS:
                grids = int(
                    inventory.loc[
                        inventory["city"].eq(city)
                        & inventory["resolution_m"].eq(resolution)
                        & inventory["temporal_resolution"].eq(temporal),
                        "grids",
                    ].iloc[0]
                )
                summary_parts.append(
                    pd.DataFrame(
                        [
                            {
                                "city": city,
                                "temporal_resolution": temporal,
                                "resolution_m": resolution,
                                "replicate": 0,
                                "fraction": 1.0,
                                "selected_blocks": len(blocks),
                                "available_blocks": len(blocks),
                                "realized_block_fraction": 1.0,
                                "first_period": blocks["block_start"].min(),
                                "last_period": blocks["block_end"].max(),
                                "selected_periods": int(blocks["periods_in_block"].sum()),
                                "grids": grids,
                                "nominal_panel_rows": int(blocks["periods_in_block"].sum()) * grids,
                            }
                        ]
                    )
                )
            windows = expanding_windows(periods, fractions, temporal=temporal)
            for resolution in RESOLUTIONS:
                grids = int(
                    inventory.loc[
                        inventory["city"].eq(city)
                        & inventory["resolution_m"].eq(resolution)
                        & inventory["temporal_resolution"].eq(temporal),
                        "grids",
                    ].iloc[0]
                )
                local = windows.copy()
                local.insert(0, "resolution_m", resolution)
                local.insert(0, "city", city)
                local["grids"] = grids
                local["nominal_panel_rows"] = local["selected_periods"] * grids
                expanding_parts.append(local)
            logger.info("%s %s: %d periods, %d complete blocks", city, temporal, len(periods), len(blocks))

    universe = pd.concat(universe_parts, ignore_index=True)
    period_map = pd.concat(period_map_parts, ignore_index=True)
    membership = pd.concat(membership_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)
    balance = pd.DataFrame(balance_parts)
    seeds = pd.DataFrame(seed_records)
    expanding = pd.concat(expanding_parts, ignore_index=True)

    tables = run_dir / "tables"
    universe.to_csv(tables / "primary_block_universe.csv", index=False)
    period_map.to_parquet(tables / "period_to_primary_block.parquet", index=False, compression="zstd")
    membership.to_parquet(tables / "primary_block_membership.parquet", index=False, compression="zstd")
    summary.to_csv(tables / "primary_sample_summary.csv", index=False)
    balance.to_csv(tables / "stratum_balance_diagnostics.csv", index=False)
    seeds.to_csv(tables / "sample_seeds.csv", index=False)
    expanding.to_csv(tables / "expanding_window_index.csv", index=False)

    nested_count = 0
    for (city, temporal, replicate), part in membership.loc[membership["replicate"].gt(0)].groupby(["city", "temporal_resolution", "replicate"]):
        nested_count += int(_nested_check(part, [value for value in fractions if value < 1]))
    spatial_sync = (
        summary.loc[summary["replicate"].gt(0)]
        .groupby(["city", "temporal_resolution", "replicate", "fraction"])["selected_periods"]
        .nunique()
        .eq(1)
    )
    expanding_nested = (
        expanding.sort_values("fraction")
        .groupby(["city", "resolution_m", "temporal_resolution"])["selected_periods"]
        .apply(lambda value: value.is_monotonic_increasing)
    )
    expected_specs = len(CITY_ORDER) * len(RESOLUTIONS) * 2 * (repetitions * 4 + 1)
    checks = [
        ("six city-temporal block universes", universe.groupby(["city", "temporal_resolution"]).ngroups, 6),
        ("daily universes contain 156 complete blocks", int(universe.loc[universe["temporal_resolution"].eq("day")].groupby("city")["block_id"].nunique().eq(156).sum()), 3),
        ("weekly universes contain 38 complete blocks", int(universe.loc[universe["temporal_resolution"].eq("week")].groupby("city")["block_id"].nunique().eq(38).sum()), 3),
        ("daily residual contains four periods", int(period_map.loc[period_map["temporal_resolution"].eq("day")].groupby("city")["included_in_primary_universe"].apply(lambda x: (~x).sum()).eq(4).sum()), 3),
        ("weekly residual contains three periods", int(period_map.loc[period_map["temporal_resolution"].eq("week")].groupby("city")["included_in_primary_universe"].apply(lambda x: (~x).sum()).eq(3).sum()), 3),
        ("primary sample specification count", len(summary), expected_specs),
        ("all repeated samples are nested", nested_count, len(CITY_ORDER) * 2 * repetitions),
        ("spatial resolutions use synchronized dates", int(spatial_sync.sum()), len(spatial_sync)),
        ("no duplicate selected block", int((membership.groupby(["city", "temporal_resolution", "replicate", "fraction", "block_id"]).size() == 1).sum()), int(membership.groupby(["city", "temporal_resolution", "replicate", "fraction", "block_id"]).ngroups)),
        ("expanding-window specifications", len(expanding), 90),
        ("expanding windows are nested", int(expanding_nested.sum()), len(expanding_nested)),
    ]
    acceptance = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    acceptance["status"] = np.where(acceptance["observed"].eq(acceptance["expected"]), "PASS", "FAIL")
    acceptance.to_csv(tables / "acceptance_checklist.csv", index=False)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("S16.3 acceptance failed: " + "; ".join(acceptance.loc[acceptance["status"].ne("PASS"), "check"]))

    _plot_qc(summary, balance, run_dir / "figures", int(config["reporting"]["figure_dpi"]))
    figure_manifest = pd.DataFrame(
        [
            {"file": path.name, "sha256": _sha256(path), "generator": "empirical/run_e16_sample_indices.py"}
            for path in sorted((run_dir / "figures").iterdir())
        ]
    )
    figure_manifest.to_csv(tables / "figure_manifest.csv", index=False)
    elapsed = time.monotonic() - started
    run_status = {
        "experiment": "E16",
        "stage": "S16.3",
        "status": "complete",
        "run_id": run_id,
        "source_s16_2_run": str(s16_2_run),
        "primary_sample_specifications": len(summary),
        "expanding_window_specifications": len(expanding),
        "membership_rows": len(membership),
        "acceptance_passed": int(acceptance["status"].eq("PASS").sum()),
        "acceptance_total": len(acceptance),
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now(timezone).isoformat(),
    }
    (run_dir / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    (run_dir / "interpretation.md").write_text(
        "# S16.3 sample-index design\n\n"
        "The primary design thins complete time blocks while retaining every grid in selected periods. Daily blocks contain seven consecutive days; weekly blocks contain four consecutive complete ISO weeks. Fractions are nested within each of 50 deterministic repetitions, and all three spatial resolutions share the same selected periods. The robustness design uses chronological expanding windows. These files are sample indices only: no entropy, CMI, DELB, randomization statistic, or prediction model was estimated.\n",
        encoding="utf-8",
    )
    (empirical_root / "runs" / "latest_e16_s16_3_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    logger.info("S16.3 complete in %.2f seconds", elapsed)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate E16 S16.3 sample-size indices")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()

