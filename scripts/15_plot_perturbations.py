# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render Figure 6 from held-out discovery perturbation outputs."""

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
import dg.figure_perturbations  # noqa: E402

FIGURE_STEM = "figure_6_perturbations"
INPUT_KEYS = {
    "session_perturbation_blocks": "tables/session_perturbation_blocks.csv",
    "mouse_perturbation_contrasts": "tables/mouse_perturbation_contrasts.csv",
    "mouse_perturbation_interactions": "tables/mouse_perturbation_interactions.csv",
    "perturbation_statistics": "tables/perturbation_statistics.csv",
}
CONDITION_STYLE = {
    ("contrast", "full"): ("#0072B2", "o", "-"),
    ("contrast", "reduced"): ("#56B4E9", "s", "-"),
    ("novelty", "familiar"): ("#666666", "^", "--"),
    ("novelty", "designated_novel"): ("#CC79A7", "D", "--"),
}


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
    in_progress = {
        "figure_id": "Figure 6",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/15_plot_perturbations.py",
        "started_at_utc": started_at.isoformat(),
    }
    dg.artifacts.write_json(in_progress, manifest_path)
    try:
        _run_figure(
            arguments,
            results_root=results_root,
            figure_root=figure_root,
            manifest_path=manifest_path,
            started_at=started_at,
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
                manifest_path,
            )
        raise


def _run_figure(
    arguments: argparse.Namespace,
    *,
    results_root: pathlib.Path,
    figure_root: pathlib.Path,
    manifest_path: pathlib.Path,
    started_at: datetime.datetime,
) -> None:
    analysis_manifest_path = results_root / "manifests" / "perturbation_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (analysis_manifest_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure 6 requires scripts/14_analyze_perturbations.py outputs: "
            + ", ".join(map(str, missing))
        )

    input_records = {key: _file_record(path, base=results_root) for key, path in inputs.items()}
    manifest_record = _file_record(analysis_manifest_path, base=results_root)
    analysis_manifest = _validated_analysis_manifest(analysis_manifest_path, inputs)
    prepared = dg.figure_perturbations.prepare_perturbation_figure_data(
        pl.read_csv(
            inputs["session_perturbation_blocks"], schema_overrides={"subject_id": pl.String}
        ),
        pl.read_csv(
            inputs["mouse_perturbation_contrasts"], schema_overrides={"subject_id": pl.String}
        ),
        pl.read_csv(
            inputs["mouse_perturbation_interactions"], schema_overrides={"subject_id": pl.String}
        ),
        pl.read_csv(inputs["perturbation_statistics"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "panel_ab_mouse_blocks": dg.artifacts.write_frame(
            prepared.mouse_blocks, source_root / "panel_ab_mouse_blocks.csv"
        ),
        "panel_ab_block_summary": dg.artifacts.write_frame(
            prepared.block_summary, source_root / "panel_ab_block_summary.csv"
        ),
        "panel_cd_mouse_interactions": dg.artifacts.write_frame(
            prepared.mouse_interactions, source_root / "panel_cd_mouse_interactions.csv"
        ),
        "panel_cd_statistics": dg.artifacts.write_frame(
            prepared.panel_statistics, source_root / "panel_cd_statistics.csv"
        ),
    }
    source_manifest_path = source_root / "manifest.json"
    generated_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 6",
            "generated_at_utc": generated_at.isoformat(),
            "analysis_manifest": manifest_record,
            "files": _file_records(sources.values(), base=results_root),
            "aggregation": "equal sessions within mouse, then equal mice",
            "interval": "10,000-resample mouse percentile bootstrap",
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

    if _file_record(analysis_manifest_path, base=results_root) != manifest_record:
        raise RuntimeError("perturbation manifest changed while Figure 6 was generated")
    if {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    } != input_records:
        raise RuntimeError("a perturbation input changed while Figure 6 was generated")

    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_perturbations.py",
        generator,
    )
    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 6",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "analysis_id": analysis_manifest["analysis_id"],
            "analysis_tier": analysis_manifest["analysis_tier"],
            "dandiset_version": analysis_manifest["dandiset_version"],
            "code_version": analysis_manifest["code_version"],
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "generated_at_utc": generated_at.isoformat(),
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "local_sources": _file_records(local_sources, base=REPOSITORY_ROOT),
            "analysis_manifest": manifest_record,
            "inputs": list(input_records.values()),
            "source_data_manifest": _file_record(source_manifest_path, base=results_root),
            "source_data": _file_records(sources.values(), base=results_root),
            "outputs": _file_records(output_paths.values(), base=results_root),
            "panels": {
                "A": (
                    "behavioral responses across E1, NR, and E2 for contrast "
                    "and designated-identity probes"
                ),
                "B": "frozen early reward-axis scores across states for the same probes",
                "C": "mouse-level behavioral state-by-perturbation interactions",
                "D": (
                    "mouse-level reversible reward-state contrasts for the two "
                    "held-out neural probe conditions"
                ),
            },
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        manifest_path,
    )
    print(f"Wrote Figure 6 to {figure_root}")


def create_figure(prepared: dg.figure_perturbations.PerturbationFigureData):
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
        "legend.fontsize": 6.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.1), constrained_layout=True)
        _draw_block_lines(
            axes[0, 0],
            prepared.block_summary,
            metric="behavior_response",
            title="Behavior generalizes across probes",
            ylabel="Response probability",
            show_legend=True,
        )
        _draw_block_lines(
            axes[0, 1],
            prepared.block_summary,
            metric="early_reward_axis",
            title="Frozen early axis transfers",
            ylabel="Early reward-axis score (a.u.)",
            show_legend=False,
        )
        _draw_interactions(
            axes[1, 0],
            prepared.mouse_interactions,
            prepared.panel_statistics,
            metric="behavior_response",
            title="Behavior: state x perturbation",
            ylabel="Test - reference\nreversible contrast",
        )
        _draw_heldout_axis_contrasts(
            axes[1, 1],
            prepared.mouse_contrasts,
            prepared.panel_statistics,
            title="Held-out early-axis transfer",
            ylabel="Reversible reward-state\ncontrast",
        )
        for axis, label in zip(axes.flat, ("A", "B", "C", "D"), strict=True):
            axis.text(
                -0.18,
                1.08,
                label,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
        return figure


def _draw_block_lines(
    axis,
    summary: pl.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    selected = summary.filter(pl.col("metric") == metric)
    for family, condition, _, label in dg.figure_perturbations.CONDITIONS:
        rows = selected.filter(
            (pl.col("perturbation_family") == family) & (pl.col("condition") == condition)
        ).sort("block_order")
        if rows.is_empty():
            continue
        color, marker, linestyle = CONDITION_STYLE[(family, condition)]
        x = rows.get_column("block_order").to_numpy()
        estimate = rows.get_column("estimate").to_numpy()
        low = rows.get_column("ci_low").to_numpy()
        high = rows.get_column("ci_high").to_numpy()
        axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        axis.plot(
            x,
            estimate,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.25,
            markersize=3.5,
            label=label,
        )
    axis.set_xticks((1, 2, 3), ("E1", "NR", "E2"))
    axis.set_xlim(0.75, 3.25)
    axis.set_xlabel("Reward state")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    if show_legend:
        axis.legend(frameon=False, ncol=2, loc="best", handlelength=2.2)


def _draw_interactions(
    axis,
    interactions: pl.DataFrame,
    statistics: pl.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":", zorder=0)
    colors = {"contrast": "#56B4E9", "novelty": "#CC79A7"}
    labels = {"contrast": "70% - full", "novelty": "novel - familiar"}
    all_values: list[float] = []
    for x, family in enumerate(("contrast", "novelty")):
        values = interactions.filter(
            (pl.col("metric") == metric) & (pl.col("perturbation_family") == family)
        ).sort("subject_id")
        statistic = statistics.filter(
            (pl.col("metric") == metric) & (pl.col("perturbation_family") == family)
        ).row(0, named=True)
        observations = values.get_column("interaction").to_numpy()
        jitter = np.linspace(-0.10, 0.10, observations.size)
        axis.scatter(
            x + jitter,
            observations,
            s=13,
            color=colors[family],
            alpha=0.38,
            linewidths=0,
        )
        axis.errorbar(
            x,
            statistic["estimate"],
            yerr=np.array(
                [
                    [statistic["estimate"] - statistic["ci_low"]],
                    [statistic["ci_high"] - statistic["estimate"]],
                ]
            ),
            fmt="o",
            color=colors[family],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.8,
            capsize=2.5,
            linewidth=1.4,
            zorder=3,
        )
        all_values.extend((*observations.tolist(), statistic["ci_low"], statistic["ci_high"]))
        axis.text(
            x,
            statistic["ci_high"],
            f"$P_{{Holm}}$ = {_format_p(statistic['adjusted_p_value'])}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#444444",
        )
    span = max(max(all_values) - min(all_values), 0.1)
    axis.set_ylim(min(0.0, min(all_values)) - 0.14 * span, max(0.0, max(all_values)) + 0.30 * span)
    axis.set_xticks((0, 1), (labels["contrast"], labels["novelty"]))
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _draw_heldout_axis_contrasts(
    axis,
    contrasts: pl.DataFrame,
    statistics: pl.DataFrame,
    *,
    title: str,
    ylabel: str,
) -> None:
    axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":", zorder=0)
    specifications = (
        ("contrast", "reduced", "70% contrast", "#56B4E9"),
        ("novelty", "designated_novel", "Designated novel", "#CC79A7"),
    )
    all_values: list[float] = []
    for x, (family, condition, _label, color) in enumerate(specifications):
        values = contrasts.filter(
            (pl.col("metric") == "early_reward_axis")
            & (pl.col("perturbation_family") == family)
            & (pl.col("condition") == condition)
        ).sort("subject_id")
        statistic = statistics.filter(
            (pl.col("panel_estimand") == "heldout_condition")
            & (pl.col("perturbation_family") == family)
        ).row(0, named=True)
        observations = values.get_column("reversible_contrast").to_numpy()
        jitter = np.linspace(-0.10, 0.10, observations.size)
        axis.scatter(
            x + jitter,
            observations,
            s=13,
            color=color,
            alpha=0.38,
            linewidths=0,
        )
        axis.errorbar(
            x,
            statistic["estimate"],
            yerr=np.array(
                [
                    [statistic["estimate"] - statistic["ci_low"]],
                    [statistic["ci_high"] - statistic["estimate"]],
                ]
            ),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=5.8,
            capsize=2.5,
            linewidth=1.4,
            zorder=3,
        )
        all_values.extend((*observations.tolist(), statistic["ci_low"], statistic["ci_high"]))
        axis.text(
            x,
            statistic["ci_high"],
            f"$P_{{Holm}}$ = {_format_p(statistic['adjusted_p_value'])}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#444444",
        )
    span = max(max(all_values) - min(all_values), 0.1)
    axis.set_ylim(min(0.0, min(all_values)) - 0.14 * span, max(all_values) + 0.30 * span)
    axis.set_xticks((0, 1), [row[2] for row in specifications])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _validated_analysis_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("perturbation manifest must be a JSON object")
    if (
        manifest.get("analysis_id") != "heldout_perturbation_axis"
        or manifest.get("run_status") != "complete"
        or manifest.get("analysis_tier") != "discovery"
        or manifest.get("confirmation_accessed") is not False
    ):
        raise RuntimeError("perturbation analysis is not a completed sealed discovery run")
    output_records = manifest.get("outputs")
    if not isinstance(output_records, dict):
        raise RuntimeError("perturbation manifest has no output records")
    for key, input_path in inputs.items():
        record = output_records.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"perturbation manifest does not bind {input_path.name}")
        observed = _file_record(input_path, base=input_path.parents[1])
        if (
            record.get("sha256") != observed["sha256"]
            or record.get("size_bytes") != observed["size_bytes"]
        ):
            raise RuntimeError(f"{input_path.name} changed after perturbation analysis")
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
