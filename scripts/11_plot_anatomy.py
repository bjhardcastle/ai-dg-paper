# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Render discovery-only anatomical-enrichment Figure 4."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np
import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.figure_anatomy  # noqa: E402

FIGURE_STEM = "figure_4_anatomy"
INPUT_KEYS = {
    "anatomical_axis_unit_scores": "tables/anatomical_axis_unit_scores.parquet",
    "anatomy_region_coverage": "tables/anatomy_region_coverage.csv",
    "mouse_anatomical_enrichment": "tables/mouse_anatomical_enrichment.csv",
    "anatomy_enrichment_statistics": "tables/anatomy_enrichment_statistics.csv",
    "anatomy_permutation_null": "tables/anatomy_permutation_null.parquet",
}
ELIGIBLE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A5195",
    "#5F6B6D",
)
INELIGIBLE_COLOR = "#C7C7C7"


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments()
    if arguments.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    results_root = arguments.results_root.resolve()
    figure_root = results_root / "figures" / "main"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_manifest_path = figure_root / f"{FIGURE_STEM}_manifest.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "figure_id": "Figure 4",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/11_plot_anatomy.py",
        "started_at_utc": started_at.isoformat(),
    }
    dg.artifacts.write_json(in_progress, figure_manifest_path)
    try:
        _run_figure(
            arguments,
            started_at=started_at,
            results_root=results_root,
            figure_root=figure_root,
            figure_manifest_path=figure_manifest_path,
        )
    except BaseException as error:
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(
                {
                    **in_progress,
                    "run_status": "failed",
                    "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                figure_manifest_path,
            )
        raise


def _run_figure(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    results_root: pathlib.Path,
    figure_root: pathlib.Path,
    figure_manifest_path: pathlib.Path,
) -> None:
    analysis_manifest_path = results_root / "manifests" / "anatomy_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (analysis_manifest_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure 4 requires scripts/10_analyze_anatomy.py outputs: "
            + ", ".join(str(path) for path in missing)
        )
    manifest_record_at_start = _file_record(analysis_manifest_path, base=results_root)
    input_records_at_start = {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    }
    analysis_manifest = _validated_analysis_manifest(
        analysis_manifest_path,
        inputs,
        results_root=results_root,
    )
    prepared = dg.figure_anatomy.prepare_anatomy_figure_data(
        pl.read_parquet(inputs["anatomical_axis_unit_scores"]),
        pl.read_csv(inputs["anatomy_region_coverage"]),
        pl.read_csv(
            inputs["mouse_anatomical_enrichment"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(inputs["anatomy_enrichment_statistics"]),
        pl.read_parquet(inputs["anatomy_permutation_null"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panel_a_unit_coordinates": dg.artifacts.write_frame(
            prepared.unit_coordinates,
            source_root / "panel_a_unit_coordinates.csv",
        ),
        "panel_a_region_coverage": dg.artifacts.write_frame(
            prepared.region_coverage,
            source_root / "panel_a_region_coverage.csv",
        ),
        "panel_b_mouse_effects": dg.artifacts.write_frame(
            prepared.mouse_effects,
            source_root / "panel_b_mouse_effects.csv",
        ),
        "panel_b_statistics": dg.artifacts.write_frame(
            prepared.statistics,
            source_root / "panel_b_statistics.csv",
        ),
        "panel_c_lead_permutation_null": dg.artifacts.write_frame(
            prepared.lead_permutation_null,
            source_root / "panel_c_lead_permutation_null.csv",
        ),
    }
    source_manifest_path = source_root / "manifest.json"
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 4",
            "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "analysis_manifest": manifest_record_at_start,
            "files": _file_records(source_paths.values(), base=results_root),
            "lead_screen_region": {
                "id": prepared.lead_region_id,
                "acronym": prepared.lead_region_acronym,
                "name": prepared.lead_region_name,
            },
            "aggregation": "region within session; equal sessions within mouse; equal mice",
            "inference": "mouse bootstrap CI and session-preserving anatomy-label permutation",
        },
        source_manifest_path,
    )

    figure = create_figure(prepared)
    output_paths = {
        extension: figure_root / f"{FIGURE_STEM}.{extension}" for extension in ("svg", "pdf", "png")
    }
    try:
        for extension, output_path in output_paths.items():
            _save_figure_atomic(
                figure,
                output_path,
                file_format=extension,
                dpi=arguments.dpi,
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    if _file_record(analysis_manifest_path, base=results_root) != manifest_record_at_start:
        raise RuntimeError("anatomy manifest changed while Figure 4 was generated")
    if {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    } != input_records_at_start:
        raise RuntimeError("an anatomy input changed while Figure 4 was generated")

    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "anatomical_enrichment.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_anatomy.py",
        generator,
    )
    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 4",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "generated_at_utc": completed_at.isoformat(),
            "analysis_id": analysis_manifest["analysis_id"],
            "analysis_tier": analysis_manifest["analysis_tier"],
            "dandiset_version": analysis_manifest.get("dandiset_version"),
            "code_version": analysis_manifest.get("code_version"),
            "confirmation_accessed": False,
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "local_sources": _file_records(local_sources, base=REPOSITORY_ROOT),
            "analysis_manifest": manifest_record_at_start,
            "inputs": list(input_records_at_start.values()),
            "source_data_manifest": _file_record(source_manifest_path, base=results_root),
            "source_data": _file_records(source_paths.values(), base=results_root),
            "outputs": _file_records(output_paths.values(), base=results_root),
            "panels": {
                "A": "CCF coordinate coverage of isolation-qualified discovery units",
                "B": (
                    "all-mouse absolute reward-state-axis loading enrichment with "
                    "mouse-bootstrap confidence intervals"
                ),
                "C": "session-preserving permutation null for the transparent screen lead",
            },
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        figure_manifest_path,
    )
    print(f"Wrote Figure 4 to {figure_root}")


def create_figure(prepared: dg.figure_anatomy.AnatomyFigureData):
    """Create the CCF coverage, mouse forest, and permutation panels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.3,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.4,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(style):
        figure = plt.figure(figsize=(7.15, 4.2), constrained_layout=True)
        grid = figure.add_gridspec(1, 3, width_ratios=(1.05, 1.25, 0.9))
        coordinate_axis = figure.add_subplot(grid[0, 0])
        effect_axis = figure.add_subplot(grid[0, 1])
        null_axis = figure.add_subplot(grid[0, 2])
        colors = _region_colors(prepared.region_coverage)
        _draw_coordinate_coverage(
            coordinate_axis,
            prepared.unit_coordinates,
            prepared.region_coverage,
            colors,
        )
        _draw_mouse_forest(
            effect_axis,
            prepared.mouse_effects,
            prepared.statistics,
            colors,
        )
        _draw_permutation_null(
            null_axis,
            prepared.lead_permutation_null,
            prepared.statistics,
            prepared.lead_region_id,
            colors[prepared.lead_region_id],
        )
        for axis, label in zip(
            (coordinate_axis, effect_axis, null_axis),
            ("A", "B", "C"),
            strict=True,
        ):
            axis.text(
                -0.19,
                1.09,
                label,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
        return figure


def _region_colors(coverage: pl.DataFrame) -> dict[int, str]:
    eligible = coverage.filter(pl.col("coverage_eligible")).sort("parent_region_order")
    return {
        row["parent_region_id"]: ELIGIBLE_COLORS[index % len(ELIGIBLE_COLORS)]
        for index, row in enumerate(eligible.iter_rows(named=True))
    }


def _draw_coordinate_coverage(axis, units, coverage, colors) -> None:
    ineligible = units.filter(~pl.col("coverage_eligible"))
    if ineligible.height:
        axis.scatter(
            ineligible.get_column("anterior_posterior_ccf_coordinate").to_numpy() / 1_000,
            ineligible.get_column("dorsal_ventral_ccf_coordinate").to_numpy() / 1_000,
            s=1.5,
            color=INELIGIBLE_COLOR,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
            label="Below coverage",
        )
    for row in coverage.filter(pl.col("coverage_eligible")).iter_rows(named=True):
        selected = units.filter(pl.col("parent_region_id") == row["parent_region_id"])
        axis.scatter(
            selected.get_column("anterior_posterior_ccf_coordinate").to_numpy() / 1_000,
            selected.get_column("dorsal_ventral_ccf_coordinate").to_numpy() / 1_000,
            s=2.0,
            color=colors[row["parent_region_id"]],
            alpha=0.22,
            linewidths=0,
            rasterized=True,
            label=row["parent_region_acronym"],
        )
    axis.invert_yaxis()
    axis.set_xlabel("Anterior-posterior CCF (mm)")
    axis.set_ylabel("Dorsal-ventral CCF (mm)")
    axis.set_title("Recorded-unit coverage", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, markerscale=3.2, handletextpad=0.2, ncols=2)


def _draw_mouse_forest(axis, mouse_effects, statistics, colors) -> None:
    ordered = statistics.sort(pl.col("estimate"), descending=True)
    y_positions = np.arange(ordered.height, dtype=float)
    all_values: list[float] = [0.0]
    for y, row in zip(y_positions, ordered.iter_rows(named=True), strict=True):
        region_id = row["parent_region_id"]
        mice = mouse_effects.filter(pl.col("parent_region_id") == region_id).sort("subject_id")
        values = mice.get_column("mouse_effect").to_numpy()
        jitter = np.linspace(-0.13, 0.13, values.size) if values.size > 1 else np.array([0.0])
        color = colors[region_id]
        axis.scatter(
            values,
            y + jitter,
            s=9,
            color=color,
            alpha=0.33,
            linewidths=0,
            zorder=1,
        )
        axis.errorbar(
            row["estimate"],
            y,
            xerr=np.array(
                [
                    [row["estimate"] - row["ci_low"]],
                    [row["ci_high"] - row["estimate"]],
                ]
            ),
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=4.8,
            capsize=2.2,
            linewidth=1.3,
            zorder=3,
        )
        all_values.extend([*values.tolist(), row["ci_low"], row["ci_high"]])
    labels = [
        f"{row['parent_region_name']}\n{row['n_mice']} mice, {row['n_units']:,} units"
        for row in ordered.iter_rows(named=True)
    ]
    axis.axvline(0, color="#333333", linewidth=0.8, linestyle=":", zorder=0)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    span = max(max(all_values) - min(all_values), 0.3)
    axis.set_xlim(min(all_values) - 0.10 * span, max(all_values) + 0.10 * span)
    axis.set_xlabel("Absolute loading enrichment\n(within-session z score)")
    axis.set_title("Regional enrichment by mouse", loc="left", fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#E1E1E1", linewidth=0.45)


def _draw_permutation_null(axis, null, statistics, region_id, color) -> None:
    row = statistics.filter(pl.col("parent_region_id") == region_id).row(0, named=True)
    values = null.get_column("null_estimate").to_numpy()
    axis.hist(values, bins=28, color="#B8B8B8", edgecolor="white", linewidth=0.35)
    axis.axvline(row["estimate"], color=color, linewidth=1.8)
    axis.axvline(0, color="#333333", linewidth=0.7, linestyle=":")
    axis.set_xlabel("Permuted mouse-mean\nenrichment")
    axis.set_ylabel("Permutation count")
    axis.set_title(
        f"Screen lead: {row['parent_region_name']}",
        loc="left",
        fontweight="bold",
    )
    axis.text(
        0.98,
        0.96,
        f"observed = {row['estimate']:.2f}\n$P_{{Holm}}$ = {_format_p(row['adjusted_p_value'])}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#333333",
    )
    axis.spines[["top", "right"]].set_visible(False)


def _format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _validated_analysis_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
    *,
    results_root: pathlib.Path,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "analysis_id": "anatomical_axis_enrichment",
        "analysis_tier": "discovery",
        "run_status": "complete",
        "authoritative": True,
        "confirmation_accessed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"anatomy analysis manifest has invalid {key!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("anatomy analysis manifest lacks outputs")
    for key, input_path in inputs.items():
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"anatomy analysis manifest does not bind {key}")
        observed = _file_record(input_path, base=results_root)
        if any(record.get(field) != observed[field] for field in ("path", "sha256", "size_bytes")):
            raise RuntimeError(f"{input_path.name} changed after anatomy analysis")
    return manifest


def _save_figure_atomic(
    figure,
    destination: pathlib.Path,
    *,
    file_format: str,
    dpi: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(name)
    try:
        figure.savefig(temporary, format=file_format, dpi=dpi, bbox_inches="tight")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(base.resolve())),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _file_records(paths, *, base: pathlib.Path) -> list[dict[str, Any]]:
    return [_file_record(pathlib.Path(path), base=base) for path in paths]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
