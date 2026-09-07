# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render neural-transition Figure 2 from discovery-cohort result tables."""

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
import dg.figure_neural_transitions  # noqa: E402

FIGURE_STEM = "figure_2_neural_transitions"
INPUT_KEYS = {
    "neural_transition_trajectory": "tables/neural_transition_trajectory.csv",
    "mouse_neural_transition_effects": "tables/mouse_neural_transition_effects.csv",
    "neural_transition_statistics": "tables/neural_transition_statistics.csv",
}
WITHDRAWAL_COLOR = "#D55E00"
RESTORATION_COLOR = "#009E73"
PSEUDO_COLOR = "#777777"


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Artifact root (default: repository results/)",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args(arguments)


def main() -> None:
    """Render Figure 2 and leave an explicit failure marker on error."""

    arguments = parse_arguments()
    if arguments.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    results_root = arguments.results_root.resolve()
    figure_root = results_root / "figures" / "main"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_manifest_path = figure_root / f"{FIGURE_STEM}_manifest.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "figure_id": "Figure 2",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/07_plot_neural_transitions.py",
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
        failed = {
            **in_progress,
            "run_status": "failed",
            "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(failed, figure_manifest_path)
        raise


def _run_figure(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    results_root: pathlib.Path,
    figure_root: pathlib.Path,
    figure_manifest_path: pathlib.Path,
) -> None:
    analysis_manifest_path = results_root / "manifests" / "neural_transition_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (analysis_manifest_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure 2 requires scripts/06_analyze_neural_transitions.py outputs: "
            + ", ".join(str(path) for path in missing)
        )

    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_neural_transitions.py",
        generator,
    )
    manifest_record_at_start = _file_record(analysis_manifest_path, base=results_root)
    input_records_at_start = {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    }
    local_source_records = _file_records(local_sources, base=REPOSITORY_ROOT)
    analysis_manifest = _validated_analysis_manifest(analysis_manifest_path, inputs)
    prepared = dg.figure_neural_transitions.prepare_neural_transition_figure_data(
        pl.read_csv(
            inputs["neural_transition_trajectory"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(
            inputs["mouse_neural_transition_effects"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(inputs["neural_transition_statistics"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panel_a_mouse_trajectories": dg.artifacts.write_frame(
            prepared.mouse_trajectories,
            source_root / "panel_a_mouse_trajectories.csv",
        ),
        "panel_a_trajectory_summary": dg.artifacts.write_frame(
            prepared.trajectory_summary,
            source_root / "panel_a_trajectory_summary.csv",
        ),
        "panel_b_mouse_effects": dg.artifacts.write_frame(
            prepared.mouse_effects,
            source_root / "panel_b_mouse_effects.csv",
        ),
        "panel_b_statistics": dg.artifacts.write_frame(
            prepared.statistics,
            source_root / "panel_b_statistics.csv",
        ),
    }
    source_manifest_path = source_root / "manifest.json"
    generated_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 2",
            "generated_at_utc": generated_at.isoformat(),
            "analysis_manifest": manifest_record_at_start,
            "files": _file_records(source_paths.values(), base=results_root),
            "aggregation": "equal sessions within mouse, then equal mice",
            "effect_window_minutes_per_side": 6,
            "trajectory_ci": (
                f"{dg.figure_neural_transitions.BOOTSTRAP_RESAMPLES:,}-resample "
                "mouse percentile bootstrap"
            ),
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
        raise RuntimeError("neural-transition manifest changed while Figure 2 was generated")
    if {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    } != input_records_at_start:
        raise RuntimeError("a neural-transition input changed while Figure 2 was generated")

    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 2",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "generated_at_utc": completed_at.isoformat(),
            "analysis_id": analysis_manifest["analysis_id"],
            "analysis_tier": analysis_manifest.get("analysis_tier", "discovery"),
            "dandiset_version": analysis_manifest.get("dandiset_version"),
            "code_version": analysis_manifest.get("code_version"),
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "local_sources": local_source_records,
            "analysis_manifest": manifest_record_at_start,
            "inputs": list(input_records_at_start.values()),
            "source_data_manifest": _file_record(source_manifest_path, base=results_root),
            "source_data": _file_records(source_paths.values(), base=results_root),
            "outputs": _file_records(output_paths.values(), base=results_root),
            "panels": {
                "A": (
                    "mouse-level reward-state axis trajectories around real and "
                    "within-state pseudo-boundaries at withdrawal and restoration"
                ),
                "B": (
                    "all-mouse direction-corrected real-minus-pseudo transition "
                    "effects over six minutes per side with mouse-bootstrap 95% "
                    "confidence intervals"
                ),
            },
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        figure_manifest_path,
    )
    print(f"Wrote Figure 2 to {figure_root}")


def create_figure(prepared: dg.figure_neural_transitions.NeuralTransitionFigureData):
    """Create the two-facet trajectory panel and controlled-effect panel."""

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
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(style):
        figure = plt.figure(figsize=(7.15, 3.15), constrained_layout=True)
        grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.12))
        first_trajectory_axis = figure.add_subplot(grid[0, 0])
        trajectory_axes = [
            first_trajectory_axis,
            figure.add_subplot(grid[0, 1], sharey=first_trajectory_axis),
        ]
        effect_axis = figure.add_subplot(grid[0, 2])

        for axis, (transition, _, label) in zip(
            trajectory_axes,
            dg.figure_neural_transitions.TRANSITIONS,
            strict=True,
        ):
            _draw_trajectory(axis, prepared.trajectory_summary, transition, label)
        trajectory_axes[0].set_ylabel("Reward-state axis score (a.u.)")
        trajectory_axes[1].legend(frameon=False, loc="best", handlelength=2.0)
        _draw_effects(effect_axis, prepared.mouse_effects, prepared.statistics)

        trajectory_axes[0].text(
            -0.22,
            1.10,
            "A",
            transform=trajectory_axes[0].transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        effect_axis.text(
            -0.22,
            1.10,
            "B",
            transform=effect_axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        return figure


def _draw_trajectory(axis, summary: pl.DataFrame, transition: str, title: str) -> None:
    transition_color = WITHDRAWAL_COLOR if transition == "withdrawal" else RESTORATION_COLOR
    selected = summary.filter(pl.col("transition") == transition)
    for anchor_type, label, color, marker, linestyle in (
        ("real", "Real boundary", transition_color, "o", "-"),
        ("pseudo", "Pseudo-boundary", PSEUDO_COLOR, "s", "--"),
    ):
        rows = selected.filter(pl.col("anchor_type") == anchor_type).sort("bin_center_minutes")
        x = rows.get_column("bin_center_minutes").to_numpy()
        estimate = rows.get_column("mean_axis_score").to_numpy()
        low = rows.get_column("ci_low").to_numpy()
        high = rows.get_column("ci_high").to_numpy()
        axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        axis.plot(
            x,
            estimate,
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.25,
            linestyle=linestyle,
            label=label,
        )
    axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle=":")
    axis.axhline(0.0, color="#BBBBBB", linewidth=0.55, zorder=0)
    axis.set_xlim(-6.0, 6.0)
    axis.set_xticks((-5, -3, -1, 1, 3, 5))
    axis.set_xlabel("Minutes from boundary")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _draw_effects(axis, mouse_effects: pl.DataFrame, statistics: pl.DataFrame) -> None:
    axis.axhline(0.0, color="#222222", linewidth=0.8, linestyle=":", zorder=0)
    labels: list[str] = []
    all_values: list[float] = []
    for x, (transition, _, label) in enumerate(dg.figure_neural_transitions.TRANSITIONS):
        effects = mouse_effects.filter(pl.col("transition") == transition).sort("subject_id")
        statistic = statistics.filter(pl.col("transition") == transition).row(0, named=True)
        values = effects.get_column("real_minus_pseudo").to_numpy()
        jitter = np.linspace(-0.10, 0.10, values.size) if values.size > 1 else np.array([0.0])
        color = WITHDRAWAL_COLOR if transition == "withdrawal" else RESTORATION_COLOR
        axis.scatter(
            x + jitter,
            values,
            s=13,
            color=color,
            alpha=0.34,
            linewidths=0,
            zorder=1,
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
        labels.append(label.replace("Reward ", ""))
        all_values.extend((*values.tolist(), statistic["ci_low"], statistic["ci_high"]))
        axis.text(
            x,
            max(all_values),
            f"$P_{{Holm}}$ = {_format_adjusted_p(statistic['adjusted_p_value'])}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#444444",
        )
    value_min = min(all_values)
    value_max = max(all_values)
    span = max(value_max - value_min, 0.25)
    axis.set_ylim(min(0.0, value_min) - 0.13 * span, max(0.0, value_max) + 0.28 * span)
    axis.set_xlim(-0.45, 1.45)
    axis.set_xticks((0, 1), labels)
    axis.set_ylabel("Direction-corrected axis step\n(real - pseudo)")
    axis.set_title("Six-minute transition effect", loc="left", fontweight="bold")
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
        raise RuntimeError("neural-transition manifest must be a JSON object")
    if (
        manifest.get("analysis_id") != "neural_transition_axis"
        or manifest.get("run_status") != "complete"
    ):
        raise RuntimeError("neural-transition analysis is not a completed discovery run")
    if manifest.get("authoritative") is False:
        raise RuntimeError("neural-transition analysis is marked non-authoritative")
    output_value = manifest.get("outputs")
    if isinstance(output_value, dict):
        output_records = list(output_value.values())
        records_by_key = output_value
    elif isinstance(output_value, list):
        output_records = output_value
        records_by_key = {}
    else:
        raise RuntimeError("neural-transition manifest has no output records")

    for key, input_path in inputs.items():
        candidates = []
        keyed = records_by_key.get(key)
        if isinstance(keyed, dict):
            candidates.append(keyed)
        candidates.extend(
            record
            for record in output_records
            if isinstance(record, dict)
            and pathlib.Path(str(record.get("path", ""))).name == input_path.name
        )
        if not candidates:
            raise RuntimeError(f"neural-transition manifest does not bind {input_path.name}")
        observed = _file_record(input_path, base=input_path.parents[1])
        if not any(
            record.get("sha256") == observed["sha256"]
            and record.get("size_bytes") == observed["size_bytes"]
            for record in candidates
        ):
            raise RuntimeError(f"{input_path.name} changed after neural-transition analysis")
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


def _file_records(
    paths,
    *,
    base: pathlib.Path,
) -> list[dict[str, object]]:
    return [_file_record(pathlib.Path(path), base=base) for path in paths]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
