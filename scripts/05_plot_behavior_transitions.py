# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render supplementary behavior-transition QC from validated result tables."""

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
import dg.figure_behavior_transitions  # noqa: E402

FIGURE_STEM = "figure_s1_behavior_transitions"
INPUT_KEYS = {
    "behavior_transition_time_trajectory_summary": (
        "tables/behavior_transition_time_trajectory_summary.csv"
    ),
    "mouse_behavior_transition_effects": "tables/mouse_behavior_transition_effects.csv",
    "behavior_transition_statistics": "tables/behavior_transition_statistics.csv",
}
GO_COLOR = "#0072B2"
CATCH_COLOR = "#777777"
WITHDRAWAL_COLOR = "#D55E00"
RESTORATION_COLOR = "#009E73"


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
    """Publish a fail-closed run marker around Figure S1 generation."""

    arguments = parse_arguments()
    if arguments.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    results_root = arguments.results_root.resolve()
    figure_root = results_root / "figures" / "supplementary"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure_manifest_path = figure_root / f"{FIGURE_STEM}_manifest.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "figure_id": "Figure S1",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/05_plot_behavior_transitions.py",
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
    """Render Figure S1 from one stable, content-addressed input snapshot."""

    transition_run_path = results_root / "manifests" / "behavior_transition_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (transition_run_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure S1 requires scripts/04_analyze_behavior_transitions.py outputs: "
            + ", ".join(str(path) for path in missing)
        )
    generator = pathlib.Path(__file__).resolve()
    local_sources = (
        REPOSITORY_ROOT / "src" / "dg" / "artifacts.py",
        REPOSITORY_ROOT / "src" / "dg" / "figure_behavior_transitions.py",
        generator,
    )
    transition_record_at_start = _file_record(transition_run_path, base=results_root)
    input_records_at_start = {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    }
    local_source_records_at_start = _file_records(local_sources, base=REPOSITORY_ROOT)
    transition_manifest = _validated_transition_manifest(transition_run_path, inputs)
    prepared = dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
        pl.read_csv(inputs["behavior_transition_time_trajectory_summary"]),
        pl.read_csv(
            inputs["mouse_behavior_transition_effects"],
            schema_overrides={"subject_id": pl.String},
        ),
        pl.read_csv(inputs["behavior_transition_statistics"]),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panels_a_b_real_time_trajectories": dg.artifacts.write_frame(
            prepared.trajectories,
            source_root / "panels_a_b_real_time_trajectories.csv",
        ),
        "panel_c_mouse_controlled_effects": dg.artifacts.write_frame(
            prepared.mouse_effects,
            source_root / "panel_c_mouse_controlled_effects.csv",
        ),
        "panel_c_controlled_effect_statistics": dg.artifacts.write_frame(
            prepared.statistics,
            source_root / "panel_c_controlled_effect_statistics.csv",
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

    if _file_record(transition_run_path, base=results_root) != transition_record_at_start:
        raise RuntimeError("behavior-transition manifest changed while Figure S1 was generated")
    if {
        key: _file_record(path, base=results_root) for key, path in inputs.items()
    } != input_records_at_start:
        raise RuntimeError("a behavior-transition input changed while Figure S1 was generated")
    if _file_records(local_sources, base=REPOSITORY_ROOT) != local_source_records_at_start:
        raise RuntimeError("a local Figure S1 source changed while the figure was generated")

    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "figure_id": "Figure S1",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "generated_at_utc": completed_at.isoformat(),
            "analysis_scope": (
                "exploratory behavior-only transition QC; distinct from neural Figure 2"
            ),
            "analysis_id": transition_manifest["analysis_id"],
            "analysis_status": transition_manifest["analysis_status"],
            "analysis_tier": transition_manifest["analysis_tier"],
            "dandiset_version": transition_manifest["dandiset_version"],
            "code_version": transition_manifest["code_version"],
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "local_sources": local_source_records_at_start,
            "transition_manifest": transition_record_at_start,
            "inputs": list(input_records_at_start.values()),
            "source_data": _file_records(source_paths.values(), base=results_root),
            "outputs": _file_records(output_paths.values(), base=results_root),
            "panels": {
                "A": "real-boundary response trajectory at reward withdrawal",
                "B": "real-boundary response trajectory at reward restoration",
                "C": (
                    "all-mouse direction-corrected real-minus-midpoint-pseudo effects; "
                    "means and mouse-bootstrap 95% confidence intervals"
                ),
            },
            "inferential_scope": (
                "Holm correction is within the two-transition go family and separately "
                "within the two-transition go-minus-catch family"
            ),
            "software": {
                package: importlib.metadata.version(package)
                for package in ("matplotlib", "numpy", "polars")
            },
        },
        figure_manifest_path,
    )
    print(f"Wrote Figure S1 to {figure_root}")


def create_figure(
    prepared: dg.figure_behavior_transitions.BehaviorTransitionFigureData,
):
    """Create the two trajectories and mouse-level controlled-effect panel."""

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
        figure = plt.figure(figsize=(7.15, 4.9), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
        axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
        effect_axis = figure.add_subplot(grid[1, :])

        for axis, (transition_id, _, transition_label) in zip(
            axes,
            dg.figure_behavior_transitions.TRANSITIONS,
            strict=True,
        ):
            _draw_trajectory(axis, prepared.trajectories, transition_id, transition_label)
        axes[0].set_ylabel("Response probability")
        axes[1].legend(frameon=False, loc="upper left", handlelength=1.8)
        _draw_effects(effect_axis, prepared.mouse_effects, prepared.statistics)

        for label, axis in zip(("A", "B", "C"), (*axes, effect_axis), strict=True):
            axis.text(
                -0.13 if label != "C" else -0.065,
                1.08,
                label,
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
            )
        return figure


def _draw_trajectory(axis, trajectories: pl.DataFrame, transition_id: str, title: str) -> None:
    selected = trajectories.filter(pl.col("transition_id") == transition_id)
    pre_color, post_color = (
        (GO_COLOR, WITHDRAWAL_COLOR)
        if transition_id == "reward_withdrawal"
        else (WITHDRAWAL_COLOR, RESTORATION_COLOR)
    )
    axis.axvspan(-6.0, 0.0, color=pre_color, alpha=0.045, linewidth=0)
    axis.axvspan(0.0, 6.0, color=post_color, alpha=0.045, linewidth=0)
    for condition, label, color, marker in (
        ("go", "Go", GO_COLOR, "o"),
        ("catch", "Catch", CATCH_COLOR, "s"),
    ):
        rows = selected.filter(pl.col("condition") == condition).sort("bin_index")
        x = rows.get_column("relative_bin_center_seconds").to_numpy() / 60.0
        estimate = rows.get_column("response_probability").to_numpy()
        low = rows.get_column("response_probability_ci_low").to_numpy()
        high = rows.get_column("response_probability_ci_high").to_numpy()
        axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        axis.plot(
            x,
            estimate,
            color=color,
            marker=marker,
            markersize=3.4,
            linewidth=1.25,
            label=label,
        )
    axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle="--")
    axis.set_xlim(-6.0, 6.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks((-5, -3, -1, 1, 3, 5))
    axis.set_xlabel("Minutes from audited transition")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)


def _draw_effects(axis, mouse_effects: pl.DataFrame, statistics: pl.DataFrame) -> None:
    rows = statistics.sort("transition_order", "condition_order")
    y_positions = np.arange(rows.height - 1, -1, -1, dtype=float)
    axis.axvline(0.0, color="#222222", linewidth=0.8, linestyle="--", zorder=0)
    labels = []
    for y, row in zip(y_positions, rows.iter_rows(named=True), strict=True):
        selected = mouse_effects.filter(
            (pl.col("transition_id") == row["transition_id"])
            & (pl.col("condition") == row["condition"])
        ).sort("subject_id")
        values = selected.get_column("pseudo_controlled_step").to_numpy()
        offsets = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.array([0.0])
        color = (
            WITHDRAWAL_COLOR if row["transition_id"] == "reward_withdrawal" else RESTORATION_COLOR
        )
        axis.scatter(
            values,
            y + offsets,
            s=11,
            color=color,
            alpha=0.28,
            linewidths=0,
            zorder=1,
        )
        axis.errorbar(
            row["estimate"],
            y,
            xerr=np.array([[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]]),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.3,
            capsize=2.5,
            linewidth=1.4,
            zorder=3,
        )
        labels.append(f"{row['transition_label']} · {row['condition_label']}")
        axis.text(
            1.01,
            y,
            _format_adjusted_p(row["adjusted_p_value"]),
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=6.8,
            color="#444444",
        )
    axis.set_yticks(y_positions, labels)
    axis.set_ylim(-0.65, rows.height - 0.35)
    all_low = min(rows.get_column("ci_low").min(), mouse_effects["pseudo_controlled_step"].min())
    all_high = max(rows.get_column("ci_high").max(), mouse_effects["pseudo_controlled_step"].max())
    margin = max(0.08, (all_high - all_low) * 0.08)
    axis.set_xlim(min(-0.3, all_low - margin), max(0.75, all_high + margin))
    axis.set_xlabel("Direction-corrected real − midpoint-pseudo change in response probability")
    axis.set_title(
        "Transition-specific effects across mice",
        loc="left",
        fontweight="bold",
    )
    axis.text(
        1.01,
        1.04,
        "Holm-adjusted P",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#444444",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.45)


def _format_adjusted_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _validated_transition_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("behavior-transition manifest must be a JSON object")
    if (
        manifest.get("analysis_id") != "behavior_transition_qc"
        or manifest.get("analysis_status") != "pass"
        or manifest.get("analysis_tier") != "exploratory"
        or manifest.get("run_status") != "complete"
        or manifest.get("authoritative") is not True
    ):
        raise RuntimeError(
            "behavior-transition analysis is not a completed authoritative exploratory run"
        )
    output_records = manifest.get("outputs")
    if not isinstance(output_records, dict):
        raise RuntimeError("behavior-transition manifest lacks output records")
    for key, input_path in inputs.items():
        expected = output_records.get(key)
        if not isinstance(expected, dict):
            raise RuntimeError(f"behavior-transition manifest lacks output {key!r}")
        observed = _file_record(input_path, base=input_path.parent.parent)
        if (
            expected.get("sha256") != observed["sha256"]
            or expected.get("size_bytes") != observed["size_bytes"]
        ):
            raise RuntimeError(f"behavior-transition output {key!r} changed after its run")
    return manifest


def _save_figure_atomic(
    figure,
    path: pathlib.Path,
    *,
    file_format: str,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".{file_format}", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = pathlib.Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
