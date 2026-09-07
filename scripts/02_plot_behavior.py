# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render manuscript Figure 1 from validated behavior result artifacts."""

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
import dg.figure_behavior  # noqa: E402

INPUT_FILENAMES = (
    "mouse_behavior_blocks.csv",
    "mouse_gating.csv",
    "session_block_timing.csv",
    "session_attrition.csv",
    "behavior_statistics.csv",
)
FIGURE_STEM = "figure_1_behavior"
BLOCK_LABELS = ("E1", "late NR", "early E2")
BLOCK_COLORS = ("#0072B2", "#D55E00", "#009E73")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Artifact root (default: repository results/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Raster output resolution (default: 400)",
    )
    return parser.parse_args()


def main() -> None:
    """Publish a fail-closed run marker around Figure 1 generation."""

    arguments = parse_arguments()
    if arguments.dpi < 72:
        raise ValueError("--dpi must be at least 72")

    result_root = arguments.results_root.resolve()
    figure_root = result_root / "figures" / "main"
    figure_root.mkdir(parents=True, exist_ok=True)
    manifest_path = figure_root / f"{FIGURE_STEM}_manifest.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "figure_id": "Figure 1",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/02_plot_behavior.py",
        "started_at_utc": started_at.isoformat(),
    }
    dg.artifacts.write_json(in_progress, manifest_path)
    try:
        _run_figure(
            arguments,
            started_at=started_at,
            result_root=result_root,
            figure_root=figure_root,
            manifest_path=manifest_path,
        )
    except BaseException as error:
        failed = {
            **in_progress,
            "run_status": "failed",
            "authoritative": False,
            "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(failed, manifest_path)
        raise


def _run_figure(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    result_root: pathlib.Path,
    figure_root: pathlib.Path,
    manifest_path: pathlib.Path,
) -> None:
    """Render Figure 1 from one stable, content-addressed input snapshot."""

    table_root = result_root / "tables"
    inputs = {name: table_root / name for name in INPUT_FILENAMES}
    behavior_manifest_path = result_root / "manifests" / "behavior_analysis_run.json"
    missing = [
        str(path) for path in (*inputs.values(), behavior_manifest_path) if not path.is_file()
    ]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Figure 1 requires behavior artifacts from scripts/01_analyze_behavior.py. "
            f"Missing:\n  - {formatted}"
        )
    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_behavior.py",
        generator,
    )
    behavior_manifest_record_at_start = _file_record(
        behavior_manifest_path,
        base=result_root,
    )
    input_records_at_start = _file_records(inputs.values(), base=result_root)
    local_source_records_at_start = _file_records(local_sources, base=REPOSITORY_ROOT)
    behavior_manifest = _validated_behavior_manifest(behavior_manifest_path, inputs)

    prepared = dg.figure_behavior.prepare_behavior_figure_data(
        pl.read_csv(inputs["mouse_behavior_blocks.csv"]),
        pl.read_csv(inputs["mouse_gating.csv"]),
        pl.read_csv(inputs["session_block_timing.csv"]),
        pl.read_csv(inputs["session_attrition.csv"]),
        pl.read_csv(inputs["behavior_statistics.csv"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panel_a_sessions": dg.artifacts.write_frame(
            prepared.session_timing,
            source_root / "panel_a_session_block_timing.csv",
        ),
        "panel_a_summary": dg.artifacts.write_frame(
            prepared.timing_summary,
            source_root / "panel_a_median_block_timing.csv",
        ),
        "panel_b": dg.artifacts.write_frame(
            prepared.trajectories,
            source_root / "panel_b_mouse_block_trajectories.csv",
        ),
        "panel_c": dg.artifacts.write_frame(
            prepared.gating,
            source_root / "panel_c_mouse_reversible_gating.csv",
        ),
        "panel_c_attrition": dg.artifacts.write_frame(
            prepared.attrition,
            source_root / "panel_c_session_attrition.csv",
        ),
        "gating_summary_statistic": dg.artifacts.write_frame(
            prepared.gating_summary_statistic,
            source_root / "gating_summary_statistic.csv",
        ),
    }

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

    _require_unchanged_figure_inputs(
        behavior_manifest_path=behavior_manifest_path,
        behavior_manifest_record=behavior_manifest_record_at_start,
        inputs=inputs,
        input_records=input_records_at_start,
        local_sources=local_sources,
        local_source_records=local_source_records_at_start,
        result_root=result_root,
    )

    summary_statistic = prepared.gating_summary_statistic.row(0, named=True)
    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 1",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "generated_at_utc": completed_at.isoformat(),
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": local_source_records_at_start[-1]["sha256"],
            "local_sources": local_source_records_at_start,
            "dandiset_version": summary_statistic.get("dandiset_version"),
            "code_version": summary_statistic.get("code_version"),
            "analysis_id": dg.figure_behavior.SYMMETRIC_SUMMARY_ANALYSIS_ID,
            "symmetric_summary_result_id": dg.figure_behavior.SYMMETRIC_SUMMARY_RESULT_ID,
            "symmetric_summary_role": "plotted summary only; not a primary inferential test",
            "analysis_tier": summary_statistic.get("analysis_tier"),
            "result_status": summary_statistic.get("status"),
            "status_reason": summary_statistic.get("reason"),
            "session_cohort": dg.figure_behavior.TECHNICALLY_VALID_COHORT,
            "primary_blocks": [block for block, _ in dg.figure_behavior.PRIMARY_BLOCKS],
            "contrast": "0.5 * engaged_1 - no_reward_late + 0.5 * engaged_2_early",
            "figure_scope": "preliminary blockwise behavior summary",
            "transition_resolution": (
                "three block windows; trial-time trajectories around transitions remain pending"
            ),
            "aggregation": (
                "panel B: per-block equal-session means within mouse across technically valid "
                "sessions; panel C: equal-session complete-case contrasts within mouse, then "
                "equal-mouse inference"
            ),
            "n_mice": prepared.gating.height,
            "behavior_analysis_manifest": behavior_manifest_record_at_start,
            "behavior_analysis_run_status": behavior_manifest["run_status"],
            "inputs": input_records_at_start,
            "source_data": _file_records(source_paths.values(), base=result_root),
            "outputs": _file_records(output_paths.values(), base=result_root),
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        manifest_path,
    )
    print(f"Wrote Figure 1 to {figure_root}")


def create_figure(prepared: dg.figure_behavior.BehaviorFigureData):
    """Create the three-panel behavior figure from validated source data."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        figure = plt.figure(
            figsize=(7.5, 2.85),
            layout="constrained",
        )
        grid = figure.add_gridspec(1, 3, width_ratios=(1.70, 2.20, 1.55))
        timing_axis = figure.add_subplot(grid[0, 0])
        trajectory_axis = figure.add_subplot(grid[0, 1])
        gating_grid = grid[0, 2].subgridspec(2, 1, height_ratios=(4.0, 1.35), hspace=0.18)
        gating_axis = figure.add_subplot(gating_grid[0, 0])
        attrition_axis = figure.add_subplot(gating_grid[1, 0])
        _plot_timing(timing_axis, prepared.timing_summary)
        _plot_trajectories(trajectory_axis, prepared.trajectories)
        _plot_gating(gating_axis, prepared.gating, prepared.gating_summary_statistic)
        _plot_attrition_strip(attrition_axis, prepared.attrition)
        for label, axis, x_position in zip(
            ("A", "B", "C"),
            (timing_axis, trajectory_axis, gating_axis),
            (-0.17, -0.17, -0.30),
            strict=True,
        ):
            axis.text(
                x_position,
                1.08,
                label,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
            )
    return figure


def _plot_timing(axis, timing_summary: pl.DataFrame) -> None:
    ordered = timing_summary.sort("block_order")
    starts = ordered.get_column("median_start_seconds").to_numpy() / 60.0
    stops = ordered.get_column("median_stop_seconds").to_numpy() / 60.0
    widths = stops - starts
    durations = ordered.get_column("median_duration_seconds").to_numpy() / 60.0
    block_labels = ("E1", "NR", "E2")
    for start, width, duration, label, color in zip(
        starts,
        widths,
        durations,
        block_labels,
        BLOCK_COLORS,
        strict=True,
    ):
        axis.barh(
            0.0,
            width,
            left=start,
            height=0.42,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.text(
            start + width / 2.0,
            0.0,
            f"{label}\n{duration:.1f} min",
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
            fontsize=6.3,
        )
    n_sessions = ordered.get_column("n_sessions").first()
    n_mice = ordered.get_column("n_mice").first()
    axis.set_title("Task blocks (median)", loc="left")
    axis.set_xlabel("Time from first trial (min)")
    axis.set_yticks(())
    axis.set_ylim(-0.55, 0.55)
    padding = max(0.5, (stops.max() - starts.min()) * 0.025)
    axis.set_xlim(max(0.0, starts.min() - padding), stops.max() + padding)
    axis.text(
        0.02,
        0.08,
        f"{n_sessions} sessions; {n_mice} mice",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#4C5963",
    )
    axis.spines["left"].set_visible(False)
    _finish_axis(axis)


def _plot_trajectories(axis, trajectories: pl.DataFrame) -> None:
    block_positions = np.arange(len(dg.figure_behavior.PRIMARY_BLOCKS), dtype=float)
    subjects = trajectories.get_column("subject_id").unique(maintain_order=True).to_list()
    n_connected_mice = 0
    for subject_id in subjects:
        subject_rows = trajectories.filter(pl.col("subject_id") == subject_id).sort("block_order")
        values = subject_rows.get_column("response_probability").to_numpy()
        denominator_match = subject_rows.get_column("block_denominators_match_gating").first()
        connect = bool(denominator_match) and np.all(np.isfinite(values))
        n_connected_mice += int(connect)
        axis.plot(
            block_positions,
            values,
            color="#8D959C",
            linewidth=0.65,
            linestyle="-" if connect else "none",
            marker="o",
            markersize=1.8,
            markeredgewidth=0,
            alpha=0.42,
            zorder=1,
        )

    block_summary = (
        trajectories.group_by("block_order")
        .agg(
            pl.col("response_probability").mean().alias("go_mean"),
            pl.col("response_probability").count().alias("n_go_mice"),
            pl.col("false_alarm_probability").mean().alias("catch_mean"),
            pl.col("false_alarm_probability").count().alias("n_catch_mice"),
        )
        .sort("block_order")
    )
    go_means = block_summary.get_column("go_mean").to_numpy()
    catch_means = block_summary.get_column("catch_mean").to_numpy()
    axis.plot(
        block_positions,
        go_means,
        color="#20262B",
        linewidth=1.5,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Go response",
        zorder=3,
    )
    axis.scatter(
        block_positions,
        go_means,
        c=BLOCK_COLORS,
        s=15,
        linewidths=0,
        zorder=4,
    )
    axis.plot(
        block_positions,
        catch_means,
        color="#4C5963",
        linewidth=1.0,
        linestyle=(0, (2.2, 1.7)),
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="Catch response",
        zorder=2,
    )
    axis.set_title("Blockwise mouse behavior", loc="left")
    axis.set_ylabel("Response probability")
    axis.set_xticks(block_positions, BLOCK_LABELS)
    axis.set_ylim(-0.03, 1.03)
    axis.set_yticks(np.linspace(0.0, 1.0, 5))
    axis.legend(
        loc="lower left",
        handlelength=2.1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
    )
    axis.text(
        0.98,
        0.98,
        "mouse n (go): "
        + "/".join(str(value) for value in block_summary.get_column("n_go_mice"))
        + "\nmouse n (catch): "
        + "/".join(str(value) for value in block_summary.get_column("n_catch_mice")),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#4C5963",
    )
    if n_connected_mice != len(subjects):
        axis.text(
            0.98,
            0.03,
            f"{n_connected_mice}/{len(subjects)} mice connected",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.3,
            color="#4C5963",
        )
    _finish_axis(axis)


def _plot_gating(
    axis,
    gating: pl.DataFrame,
    gating_summary_statistic: pl.DataFrame,
) -> None:
    ordered = gating.sort("subject_id")
    values = ordered.get_column("reversible_gating_estimate").to_numpy()
    offsets = dg.figure_behavior.deterministic_strip_offsets(len(values), width=0.20)
    statistic = gating_summary_statistic.row(0, named=True)
    estimate = float(statistic["estimate"])
    ci_low = float(statistic["ci_low"])
    ci_high = float(statistic["ci_high"])
    confidence_percentage = 100 * float(statistic["confidence_level"])

    axis.axhline(0.0, color="#9AA1A7", linewidth=0.75, linestyle=(0, (2.5, 2.0)), zorder=0)
    axis.scatter(
        offsets,
        values,
        s=17,
        color="#575F66",
        edgecolors="white",
        linewidths=0.35,
        alpha=0.9,
        zorder=2,
    )
    axis.errorbar(
        0.34,
        estimate,
        yerr=np.asarray([[estimate - ci_low], [ci_high - estimate]]),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=1.5,
        capsize=3,
        capthick=1.2,
        markersize=4.8,
        zorder=3,
    )
    axis.set_title("Gating score", loc="left")
    axis.set_ylabel("½(E1 + early E2) − late NR")
    axis.set_xticks(
        (0.0, 0.34),
        ("Mice", "Mean"),
    )
    axis.set_xlim(-0.19, 0.52)
    axis.set_ylim(-1.03, 1.03)
    axis.set_yticks(np.linspace(-1.0, 1.0, 5))
    axis.text(
        0.98,
        0.98,
        f"n = {len(values)} mice",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#4C5963",
    )
    axis.text(
        0.98,
        0.03,
        f"mean {estimate:.2f}\n{confidence_percentage:.0f}% CI [{ci_low:.2f}, {ci_high:.2f}]",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#4C5963",
    )
    _finish_axis(axis)


def _plot_attrition_strip(axis, attrition: pl.DataFrame) -> None:
    ordered = attrition.sort("stage_order")
    counts = {
        row["attrition_stage"]: (row["n_sessions"], row["n_mice"])
        for row in ordered.iter_rows(named=True)
    }
    all_sessions, all_mice = counts["inventory"]
    technical_sessions, technical_mice = counts["technically_valid"]
    estimable_sessions, estimable_mice = counts["reversible_gating_estimable"]
    threshold_sessions, threshold_mice = counts["threshold_selected"]

    axis.set_axis_off()
    axis.text(
        0.0,
        0.96,
        "Session QC (sessions; mice)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold",
        color="#20262B",
    )
    axis.text(
        0.0,
        0.60,
        f"All {all_sessions} ({all_mice}) / technical {technical_sessions} ({technical_mice})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#30363B",
    )
    axis.text(
        0.0,
        0.22,
        f"Estimable {estimable_sessions} ({estimable_mice}) / "
        f"threshold {threshold_sessions} ({threshold_mice})",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color="#30363B",
    )


def _finish_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=2.5, pad=2.5)


def _require_unchanged_figure_inputs(
    *,
    behavior_manifest_path: pathlib.Path,
    behavior_manifest_record: dict[str, object],
    inputs: dict[str, pathlib.Path],
    input_records: list[dict[str, object]],
    local_sources: tuple[pathlib.Path, ...],
    local_source_records: list[dict[str, object]],
    result_root: pathlib.Path,
) -> None:
    """Fail if any parent, input, or local source changed during rendering."""

    if _file_record(behavior_manifest_path, base=result_root) != behavior_manifest_record:
        raise RuntimeError("behavior-analysis manifest changed while Figure 1 was generated")
    if _file_records(inputs.values(), base=result_root) != input_records:
        raise RuntimeError("a behavior input changed while Figure 1 was generated")
    if _file_records(local_sources, base=REPOSITORY_ROOT) != local_source_records:
        raise RuntimeError("a local Figure 1 source changed while the figure was generated")


def _validated_behavior_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
) -> dict[str, object]:
    """Require a completed behavior run that content-addresses every input."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read behavior analysis manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("behavior analysis manifest must be a JSON object")
    if (
        manifest.get("analysis_id") != dg.figure_behavior.SYMMETRIC_SUMMARY_ANALYSIS_ID
        or manifest.get("analysis_status") != "pass"
        or manifest.get("analysis_tier") != "exploratory"
        or manifest.get("run_status") != "complete"
        or manifest.get("authoritative") is not True
    ):
        raise RuntimeError("behavior analysis is not a completed authoritative exploratory run")
    output_records = manifest.get("outputs")
    if not isinstance(output_records, dict):
        raise RuntimeError("behavior analysis manifest lacks output records")
    checkpoint = manifest.get("behavior_trials_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("output_matches_trusted_checkpoint") is not True
    ):
        raise RuntimeError("behavior analysis is not bound to its trusted trial checkpoint")
    for filename, input_path in inputs.items():
        output_id = pathlib.Path(filename).stem
        expected = output_records.get(output_id)
        if not isinstance(expected, dict):
            raise RuntimeError(f"behavior manifest lacks output {output_id!r}")
        observed = _file_record(input_path, base=input_path.parent.parent)
        if (
            expected.get("sha256") != observed["sha256"]
            or expected.get("size_bytes") != observed["size_bytes"]
        ):
            raise RuntimeError(f"behavior output {output_id!r} changed after its run")
    return manifest


def _save_figure_atomic(
    figure,
    destination: pathlib.Path,
    *,
    file_format: str,
    dpi: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    metadata = {
        "Title": "Reward-dependent reversible visual behavior",
        "Creator": "scripts/02_plot_behavior.py",
    }
    try:
        figure.savefig(
            temporary,
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata=metadata,
        )
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_records(
    paths,
    *,
    base: pathlib.Path,
) -> list[dict[str, object]]:
    return [_file_record(pathlib.Path(path), base=base) for path in paths]


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(base.resolve())
    except ValueError:
        relative = resolved
    return {
        "path": str(relative),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
