# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render discovery-only simultaneous-network Figure 5."""

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

import numpy as np
import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.figure_network  # noqa: E402

FIGURE_STEM = "figure_5_network_interaction"
INPUT_KEYS = {
    "mouse_network_block_performance": "tables/mouse_network_block_performance.csv",
    "mouse_network_effects": "tables/mouse_network_effects.csv",
    "network_statistics": "tables/network_statistics.csv",
}
REAL_COLOR = "#0072B2"
SHIFT_COLOR = "#888888"
BLOCK_COLORS = ("#3B7EA1", "#D55E00", "#009E73")


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
    manifest_path = figure_root / f"{FIGURE_STEM}_manifest.json"
    started_at = datetime.datetime.now(datetime.UTC)
    marker = {
        "figure_id": "Figure 5",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/13_plot_network.py",
        "started_at_utc": started_at.isoformat(),
    }
    dg.artifacts.write_json(marker, manifest_path)
    try:
        _run_figure(
            arguments,
            started_at=started_at,
            results_root=results_root,
            figure_root=figure_root,
            figure_manifest_path=manifest_path,
        )
    except BaseException as error:
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(
                {
                    **marker,
                    "run_status": "failed",
                    "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                manifest_path,
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
    analysis_manifest_path = results_root / "manifests" / "network_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (analysis_manifest_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure 5 requires scripts/12_analyze_network.py outputs: "
            + ", ".join(map(str, missing))
        )
    manifest_record = _file_record(analysis_manifest_path, base=results_root)
    input_records = {key: _file_record(path, base=results_root) for key, path in inputs.items()}
    analysis_manifest = _validated_analysis_manifest(analysis_manifest_path, inputs)
    nomination = analysis_manifest["nominated_pair"]
    prepared = dg.figure_network.prepare_network_figure_data(
        pl.read_csv(
            inputs["mouse_network_block_performance"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(
            inputs["mouse_network_effects"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(inputs["network_statistics"]),
        source_region=str(nomination["source_region"]),
        target_region=str(nomination["target_region"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panel_b_mouse_blocks": dg.artifacts.write_frame(
            prepared.mouse_blocks, source_root / "panel_b_mouse_blocks.csv"
        ),
        "panel_b_block_summary": dg.artifacts.write_frame(
            prepared.block_summary, source_root / "panel_b_block_summary.csv"
        ),
        "panel_c_mouse_controls": dg.artifacts.write_frame(
            prepared.mouse_effects, source_root / "panel_c_mouse_controls.csv"
        ),
        "panel_c_statistic": dg.artifacts.write_frame(
            prepared.statistic, source_root / "panel_c_statistic.csv"
        ),
    }
    source_manifest_path = source_root / "manifest.json"
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 5",
            "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "analysis_manifest": manifest_record,
            "source_region": prepared.source_region,
            "target_region": prepared.target_region,
            "files": _file_records(source_paths.values(), base=results_root),
            "aggregation": "equal sessions within mouse, then equal mice",
        },
        source_manifest_path,
    )

    figure = create_figure(prepared)
    output_paths = {
        extension: figure_root / f"{FIGURE_STEM}.{extension}" for extension in ("svg", "pdf", "png")
    }
    try:
        for extension, path in output_paths.items():
            _save_figure_atomic(figure, path, file_format=extension, dpi=arguments.dpi)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    if _file_record(analysis_manifest_path, base=results_root) != manifest_record:
        raise RuntimeError("network analysis manifest changed while Figure 5 was generated")
    if {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    } != input_records:
        raise RuntimeError("a network figure input changed while Figure 5 was generated")

    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_network.py",
        generator,
    )
    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 5",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "analysis_id": analysis_manifest["analysis_id"],
            "analysis_tier": analysis_manifest["analysis_tier"],
            "dandiset_version": analysis_manifest.get("dandiset_version"),
            "code_version": analysis_manifest.get("code_version"),
            "confirmation_accessed": analysis_manifest.get("confirmation_accessed"),
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "local_sources": _file_records(local_sources, base=REPOSITORY_ROOT),
            "analysis_manifest": manifest_record,
            "inputs": list(input_records.values()),
            "source_data_manifest": _file_record(source_manifest_path, base=results_root),
            "source_data": _file_records(source_paths.values(), base=results_root),
            "outputs": _file_records(output_paths.values(), base=results_root),
            "panels": {
                "A": "positive-lag residual reduced-rank prediction and control schematic",
                "B": "nominated-pair source-added held-out target R-squared by reward block",
                "C": "per-mouse reversible real versus within-block trial-shift effects",
            },
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        figure_manifest_path,
    )
    print(f"Wrote Figure 5 to {figure_root}")


def create_figure(prepared: dg.figure_network.NetworkFigureData):
    """Create the model schematic, block effect, and trial-shift control panels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(style):
        figure = plt.figure(figsize=(7.15, 2.85), constrained_layout=True)
        grid = figure.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 1.0))
        schematic_axis = figure.add_subplot(grid[0, 0])
        block_axis = figure.add_subplot(grid[0, 1])
        control_axis = figure.add_subplot(grid[0, 2])
        _draw_schematic(schematic_axis, prepared.source_region, prepared.target_region)
        _draw_blocks(block_axis, prepared.mouse_blocks, prepared.block_summary)
        _draw_control(control_axis, prepared.mouse_effects, prepared.statistic)
        for label, axis in zip("ABC", (schematic_axis, block_axis, control_axis), strict=True):
            axis.text(
                -0.16,
                1.10,
                label,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
        figure.suptitle(
            f"Familywise-null screen lead: {prepared.source_region} → {prepared.target_region}",
            fontsize=9,
            fontweight="bold",
        )
        return figure


def _draw_schematic(axis, source_region: str, target_region: str) -> None:
    from matplotlib import patches

    axis.set_axis_off()
    box = {"boxstyle": "round,pad=0.35", "linewidth": 0.8}
    axis.text(
        0.22,
        0.69,
        f"{source_region}\nearly response\n0–150 ms",
        ha="center",
        va="center",
        bbox={**box, "facecolor": "#DCEAF4", "edgecolor": REAL_COLOR},
        transform=axis.transAxes,
    )
    axis.text(
        0.78,
        0.69,
        f"{target_region}\nlate response\n150–300 ms",
        ha="center",
        va="center",
        bbox={**box, "facecolor": "#E2F2EA", "edgecolor": "#009E73"},
        transform=axis.transAxes,
    )
    axis.add_patch(
        patches.FancyArrowPatch(
            (0.40, 0.69),
            (0.60, 0.69),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.3,
            color=REAL_COLOR,
            transform=axis.transAxes,
        )
    )
    axis.text(
        0.50,
        0.82,
        "positive lag",
        ha="center",
        color=REAL_COLOR,
        fontsize=6.8,
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.40,
        "Base: stimulus + session time + eventual response\n"
        "+ target early history\n\n"
        "Full: base + residual source (ridge RRR)\n"
        r"Interaction = held-out $\Delta R^2$",
        ha="center",
        va="center",
        fontsize=7,
        transform=axis.transAxes,
    )
    axis.text(
        0.50,
        0.10,
        "Control: shift source trials within block",
        ha="center",
        va="center",
        color="#555555",
        fontsize=6.8,
        transform=axis.transAxes,
    )


def _draw_blocks(axis, mouse_blocks: pl.DataFrame, summary: pl.DataFrame) -> None:
    block_labels = ("E1", "NR", "E2")
    block_names = ("engaged_1", "no_reward", "engaged_2")
    for subject_id in sorted(mouse_blocks.get_column("subject_id").unique().to_list()):
        rows = mouse_blocks.filter(pl.col("subject_id") == subject_id)
        lookup = {row["reward_block"]: row for row in rows.iter_rows(named=True)}
        axis.plot(
            range(3),
            [lookup[block]["source_added_r2"] for block in block_names],
            color="#777777",
            alpha=0.28,
            linewidth=0.7,
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
        )
    for index, (block, color) in enumerate(zip(block_names, BLOCK_COLORS, strict=True)):
        row = summary.filter(pl.col("reward_block") == block).row(0, named=True)
        axis.errorbar(
            index,
            row["mean_source_added_r2"],
            yerr=np.array(
                [
                    [row["mean_source_added_r2"] - row["ci_low"]],
                    [row["ci_high"] - row["mean_source_added_r2"]],
                ]
            ),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6,
            capsize=2.5,
            linewidth=1.4,
            zorder=4,
        )
    axis.axhline(0, color="#222222", linewidth=0.7, linestyle=":")
    axis.set_xticks(range(3), block_labels)
    axis.set_ylabel("Source-added held-out $R^2$")
    axis.set_title("Held-out prediction by block", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _draw_control(axis, effects: pl.DataFrame, statistic: pl.DataFrame) -> None:
    for row in effects.iter_rows(named=True):
        axis.plot(
            (0, 1),
            (row["real_reversible_contrast"], row["shift_reversible_contrast"]),
            color="#777777",
            alpha=0.3,
            linewidth=0.7,
        )
        axis.scatter(
            (0, 1),
            (row["real_reversible_contrast"], row["shift_reversible_contrast"]),
            color=(REAL_COLOR, SHIFT_COLOR),
            alpha=0.45,
            s=11,
            linewidths=0,
            zorder=2,
        )
    real_mean = float(effects.get_column("real_reversible_contrast").mean())
    shift_mean = float(effects.get_column("shift_reversible_contrast").mean())
    axis.plot((0, 1), (real_mean, shift_mean), color="#111111", linewidth=1.4, zorder=3)
    axis.scatter(
        (0, 1),
        (real_mean, shift_mean),
        color=(REAL_COLOR, SHIFT_COLOR),
        edgecolor="white",
        linewidth=0.7,
        s=36,
        zorder=4,
    )
    row = statistic.row(0, named=True)
    axis.axhline(0, color="#222222", linewidth=0.7, linestyle=":")
    axis.set_xlim(-0.45, 1.45)
    axis.set_xticks((0, 1), ("Real", "Trial-shift"))
    axis.set_ylabel("Reversible source-added $R^2$ contrast")
    axis.set_title("Trial-shift control", loc="left", fontweight="bold")
    axis.text(
        0.5,
        0.98,
        f"real − shift = {row['estimate']:.3f}\n"
        f"$P_{{Holm}}$ = {_format_adjusted_p(row['adjusted_p_value'])}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="#444444",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _format_adjusted_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _validated_analysis_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("network analysis manifest must be a JSON object")
    required = {
        "analysis_id": "simultaneous_network_interaction",
        "analysis_tier": "discovery",
        "run_status": "complete",
        "authoritative": True,
        "confirmation_accessed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"network analysis manifest {key!r} is not {expected!r}")
    nomination = manifest.get("nominated_pair")
    if not isinstance(nomination, dict) or not {
        "source_region",
        "target_region",
    }.issubset(nomination):
        raise RuntimeError("network analysis manifest lacks one nominated pair")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("network analysis manifest lacks keyed output records")
    for key, input_path in inputs.items():
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"network analysis manifest does not bind {input_path.name}")
        observed = _file_record(input_path, base=input_path.parents[1])
        if (
            record.get("sha256") != observed["sha256"]
            or record.get("size_bytes") != observed["size_bytes"]
        ):
            raise RuntimeError(f"{input_path.name} changed after network analysis")
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
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(name)
    try:
        figure.savefig(temporary, format=file_format, dpi=dpi, bbox_inches="tight")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(base.resolve())),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _file_records(paths, *, base: pathlib.Path) -> list[dict[str, object]]:
    return [_file_record(pathlib.Path(path), base=base) for path in paths]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
