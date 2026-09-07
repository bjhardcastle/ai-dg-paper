# /// script
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0",
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Render functional-population Figure 3 from discovery result tables."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
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
import dg.figure_functional_populations  # noqa: E402

FIGURE_STEM = "figure_3_functional_populations"
INPUT_KEYS = {
    "mouse_psths": "tables/functional_population_mouse_psths.csv",
    "mouse_effects": "tables/functional_population_mouse_effects.csv",
    "statistics": "tables/functional_population_statistics.csv",
    "class_stability": "tables/functional_population_class_stability.csv",
}
BLOCK_COLORS = {
    "engaged_1": "#0072B2",
    "no_reward": "#D55E00",
    "engaged_2": "#009E73",
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
        "figure_id": "Figure 3",
        "figure_stem": FIGURE_STEM,
        "run_status": "in_progress",
        "authoritative": False,
        "generator": "scripts/09_plot_functional_populations.py",
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
    analysis_manifest_path = results_root / "manifests" / "functional_populations_analysis_run.json"
    inputs = {key: results_root / relative for key, relative in INPUT_KEYS.items()}
    missing = [path for path in (analysis_manifest_path, *inputs.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Figure 3 requires scripts/08_analyze_functional_populations.py outputs: "
            + ", ".join(map(str, missing))
        )
    analysis_manifest = _validate_analysis_manifest(analysis_manifest_path)
    records_at_start = {
        "analysis_manifest": _file_record(analysis_manifest_path, base=results_root),
        **{key: _file_record(path, base=results_root) for key, path in inputs.items()},
    }
    prepared = dg.figure_functional_populations.prepare_functional_population_figure_data(
        pl.read_csv(inputs["mouse_psths"], schema_overrides={"subject_id": pl.String}),
        pl.read_csv(inputs["mouse_effects"], schema_overrides={"subject_id": pl.String}),
        pl.read_csv(inputs["statistics"]),
        pl.read_csv(inputs["class_stability"], schema_overrides={"subject_id": pl.String}),
    )

    source_root = figure_root / f"{FIGURE_STEM}_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "panel_a_mouse_psths": dg.artifacts.write_frame(
            prepared.mouse_psths, source_root / "panel_a_mouse_psths.csv"
        ),
        "panel_a_psth_summary": dg.artifacts.write_frame(
            prepared.psth_summary, source_root / "panel_a_psth_summary.csv"
        ),
        "panels_b_c_mouse_effects": dg.artifacts.write_frame(
            prepared.mouse_effects, source_root / "panels_b_c_mouse_effects.csv"
        ),
        "panels_b_c_statistics": dg.artifacts.write_frame(
            prepared.statistics, source_root / "panels_b_c_statistics.csv"
        ),
        "class_stability": dg.artifacts.write_frame(
            prepared.class_stability, source_root / "class_stability.csv"
        ),
    }
    source_manifest_path = source_root / "manifest.json"
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 3",
            "analysis_manifest": records_at_start["analysis_manifest"],
            "files": [_file_record(path, base=results_root) for path in source_paths.values()],
            "aggregation": "units within session, sessions within mouse, equal mice",
            "psth_interval": (
                f"{dg.figure_functional_populations.BOOTSTRAP_RESAMPLES:,}-resample "
                "mouse percentile bootstrap"
            ),
            "confirmation_accessed": False,
        },
        source_manifest_path,
    )

    figure = create_figure(prepared)
    outputs = {
        extension: figure_root / f"{FIGURE_STEM}.{extension}" for extension in ("svg", "pdf", "png")
    }
    try:
        for extension, path in outputs.items():
            _save_figure_atomic(
                figure,
                path,
                file_format=extension,
                dpi=arguments.dpi,
            )
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)

    current_records = {
        "analysis_manifest": _file_record(analysis_manifest_path, base=results_root),
        **{key: _file_record(path, base=results_root) for key, path in inputs.items()},
    }
    if current_records != records_at_start:
        raise RuntimeError("Figure 3 analysis inputs changed during rendering")
    completed_at = datetime.datetime.now(datetime.UTC)
    generator = pathlib.Path(__file__).resolve()
    dg.artifacts.write_json(
        {
            "figure_id": "Figure 3",
            "figure_stem": FIGURE_STEM,
            "run_status": "complete",
            "authoritative": True,
            "preliminary_vertical_slice": True,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "analysis_id": analysis_manifest["analysis_id"],
            "analysis_tier": analysis_manifest["analysis_tier"],
            "dandiset_version": analysis_manifest["dandiset_version"],
            "confirmation_accessed": False,
            "generator": str(generator.relative_to(REPOSITORY_ROOT)),
            "generator_sha256": _sha256(generator),
            "inputs": list(records_at_start.values()),
            "source_data_manifest": _file_record(source_manifest_path, base=results_root),
            "source_data": [
                _file_record(path, base=results_root) for path in source_paths.values()
            ],
            "outputs": [_file_record(path, base=results_root) for path in outputs.values()],
            "caveat": "discovery-only isolation-QC slice; D04 stability is pending",
        },
        manifest_path,
    )


def create_figure(data: dg.figure_functional_populations.FunctionalPopulationFigureData):
    """Create the final three-panel Figure 3."""

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(12.2, 4.1), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=(1.35, 1.35, 1.0, 1.0),
        left=0.065,
        right=0.985,
        bottom=0.20,
        top=0.82,
        wspace=0.48,
    )
    axes_a = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    axis_b = figure.add_subplot(grid[0, 2])
    axis_c = figure.add_subplot(grid[0, 3])

    for axis, (group, _, label) in zip(
        axes_a, dg.figure_functional_populations.RESPONSE_GROUPS, strict=True
    ):
        group_data = data.psth_summary.filter(pl.col("response_group") == group)
        for block, _, block_label in dg.figure_functional_populations.BLOCKS:
            rows = group_data.filter(pl.col("reward_block") == block).sort("bin_center_seconds")
            x = rows.get_column("bin_center_seconds").to_numpy() * 1000
            mean = rows.get_column("mean_sign_aligned_rate_hz").to_numpy()
            low = rows.get_column("ci_low").to_numpy()
            high = rows.get_column("ci_high").to_numpy()
            axis.plot(x, mean, color=BLOCK_COLORS[block], lw=1.6, label=block_label)
            axis.fill_between(x, low, high, color=BLOCK_COLORS[block], alpha=0.13, linewidth=0)
        axis.axvline(0, color="#444444", lw=0.7, ls="--")
        axis.axhline(0, color="#AAAAAA", lw=0.5)
        axis.axvspan(25, 150, color="#0072B2", alpha=0.055, linewidth=0)
        axis.axvspan(150, 600, color="#CC79A7", alpha=0.045, linewidth=0)
        axis.set_title(label, loc="left", fontweight="bold")
        axis.set_xlabel("Time from image change (ms)")
        axis.spines[["top", "right"]].set_visible(False)
    axes_a[0].set_ylabel("Sign-aligned evoked rate (spikes s$^{-1}$)")
    axes_a[1].legend(frameon=False, ncol=3, loc="upper right", handlelength=1.3)

    effects = data.mouse_effects.sort("subject_id")
    adjusted_effects = effects.filter(
        pl.col("early_adjusted").is_not_null() & pl.col("late_adjusted").is_not_null()
    )
    _paired_mouse_plot(
        axis_b,
        adjusted_effects.get_column("early_adjusted").to_numpy(),
        adjusted_effects.get_column("late_adjusted").to_numpy(),
        labels=("Early", "Late"),
        colors=("#0072B2", "#CC79A7"),
    )
    axis_b.set_title("Timing selectivity after adjustment", loc="left", fontweight="bold")
    axis_b.set_ylabel("Reversible state contrast\n(sign-aligned Anscombe rate)")
    adjusted_difference = _statistic(data.statistics, "late_minus_early_model_adjusted")
    _statistic_note(axis_b, adjusted_difference, prefix="Late − early")

    _paired_mouse_plot(
        axis_c,
        adjusted_effects.get_column("late_total").to_numpy(),
        adjusted_effects.get_column("late_adjusted").to_numpy(),
        labels=("Total", "Adjusted"),
        colors=("#666666", "#009E73"),
    )
    axis_c.set_title("Measured-movement control", loc="left", fontweight="bold")
    axis_c.set_ylabel("Late reversible state contrast\n(sign-aligned Anscombe rate)")
    adjusted_late = _statistic(data.statistics, "late_state_model_adjusted")
    _statistic_note(axis_c, adjusted_late, prefix="Adjusted late")

    for axis, label in zip((axes_a[0], axis_b, axis_c), ("A", "B", "C"), strict=True):
        axis.text(
            -0.20,
            1.14,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    stable_fraction = float(data.class_stability.get_column("stable_assignment_fraction").mean())
    n_mice = effects.height
    n_units = int(effects.get_column("n_units").sum())
    figure.suptitle(
        "Reward-state-associated modulation is stronger in late sensory-to-action activity",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.895,
        (
            f"Discovery cohort · {n_mice} mice · {n_units:,} isolation-QC units · "
            f"cross-half stable response labels {stable_fraction:.0%} · D04 pending"
        ),
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )
    return figure


def _paired_mouse_plot(axis, first: np.ndarray, second: np.ndarray, *, labels, colors) -> None:
    for left, right in zip(first, second, strict=True):
        axis.plot([0, 1], [left, right], color="#B8B8B8", lw=0.7, alpha=0.75, zorder=1)
    jitter = np.linspace(-0.045, 0.045, first.size)
    axis.scatter(jitter, first, s=17, color=colors[0], edgecolor="white", linewidth=0.35, zorder=2)
    axis.scatter(
        1 + jitter, second, s=17, color=colors[1], edgecolor="white", linewidth=0.35, zorder=2
    )
    axis.scatter(
        [0, 1],
        [first.mean(), second.mean()],
        marker="_",
        s=180,
        linewidth=2.1,
        color="#111111",
        zorder=3,
    )
    axis.axhline(0, color="#999999", lw=0.6, ls="--")
    axis.set_xlim(-0.35, 1.35)
    axis.set_xticks([0, 1], labels)
    axis.spines[["top", "right"]].set_visible(False)


def _statistic_note(axis, row: dict[str, object], *, prefix: str) -> None:
    if row["estimate"] is None:
        axis.text(
            0.02,
            0.98,
            f"{prefix}: not estimable",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )
        return
    p_value = float(row["adjusted_p_value"])
    interval = f"[{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]"
    axis.text(
        0.02,
        0.98,
        f"{prefix}: {float(row['estimate']):.2f}\n95% CI {interval}\nHolm $p$={p_value:.3g}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=5,
    )


def _statistic(frame: pl.DataFrame, result_id: str) -> dict[str, object]:
    return frame.filter(pl.col("result_id") == result_id).row(0, named=True)


def _validate_analysis_manifest(path: pathlib.Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    if manifest.get("run_status") != "complete":
        raise RuntimeError("functional-population analysis manifest is not complete")
    if manifest.get("analysis_id") != "functional_populations":
        raise RuntimeError("functional-population analysis manifest has wrong identity")
    if manifest.get("analysis_tier") != "discovery":
        raise RuntimeError("Figure 3 accepts discovery analysis only")
    if manifest.get("confirmation_accessed") is not False:
        raise RuntimeError("Figure 3 analysis did not keep confirmation sealed")
    return manifest


def _save_figure_atomic(figure, path: pathlib.Path, *, file_format: str, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=f".{file_format}",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(base)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
