"""Validated source-data preparation for simultaneous-network Figure 5."""

from __future__ import annotations

import dataclasses
import math

import polars as pl

import dg.network_interactions
import dg.statistics

ANALYSIS_ID = "simultaneous_network_interaction"
BOOTSTRAP_SEED = 1051
BOOTSTRAP_RESAMPLES = 10_000


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkFigureData:
    """Publication-facing tables for the single nominated discovery pair."""

    source_region: str
    target_region: str
    mouse_blocks: pl.DataFrame
    block_summary: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistic: pl.DataFrame


def prepare_network_figure_data(
    mouse_blocks: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    statistics: pl.DataFrame,
    *,
    source_region: str,
    target_region: str,
) -> NetworkFigureData:
    """Validate and bind all displayed values to one nominated pair."""

    if not source_region.strip() or not target_region.strip():
        raise ValueError("source_region and target_region must be non-empty")
    if source_region == target_region:
        raise ValueError("Figure 5 requires two distinct simultaneously recorded regions")
    for name, frame in (
        ("mouse_blocks", mouse_blocks),
        ("mouse_effects", mouse_effects),
        ("statistics", statistics),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    blocks = _prepare_mouse_blocks(mouse_blocks, source_region, target_region)
    effects = _prepare_mouse_effects(mouse_effects, blocks, source_region, target_region)
    statistic = _prepare_statistic(statistics, effects, source_region, target_region)
    summary = _summarize_blocks(blocks)
    return NetworkFigureData(
        source_region=source_region,
        target_region=target_region,
        mouse_blocks=blocks,
        block_summary=summary,
        mouse_effects=effects,
        statistic=statistic,
    )


def _prepare_mouse_blocks(
    frame: pl.DataFrame,
    source_region: str,
    target_region: str,
) -> pl.DataFrame:
    required = {
        "subject_id",
        "source_region",
        "target_region",
        "reward_block",
        "n_sessions",
        "n_trials",
        "source_added_r2",
        "shift_source_added_r2",
    }
    _require_columns(frame, required, frame_name="mouse_blocks")
    selected = (
        frame.filter(
            (pl.col("source_region") == source_region) & (pl.col("target_region") == target_region)
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("source_region").cast(pl.String),
            pl.col("target_region").cast(pl.String),
            pl.col("reward_block").cast(pl.String),
            pl.col("n_sessions").cast(pl.Int64),
            pl.col("n_trials").cast(pl.Int64),
            pl.col("source_added_r2").cast(pl.Float64),
            pl.col("shift_source_added_r2").cast(pl.Float64),
        )
        .sort("subject_id", "reward_block")
    )
    if selected.is_empty():
        raise ValueError("mouse_blocks has no rows for the nominated pair")
    if selected.select("subject_id", "reward_block").is_duplicated().any():
        raise ValueError("mouse_blocks contains duplicate nominated pair/mouse/block rows")
    if set(selected.get_column("reward_block")) != set(dg.network_interactions.REWARD_BLOCKS):
        raise ValueError("mouse_blocks does not contain all three reward blocks")
    block_counts = selected.group_by("subject_id").agg(
        pl.col("reward_block").n_unique().alias("n_blocks")
    )
    if block_counts.filter(pl.col("n_blocks") != 3).height:
        raise ValueError("every plotted mouse must contribute all three reward blocks")
    if selected.filter((pl.col("n_sessions") <= 0) | (pl.col("n_trials") <= 0)).height:
        raise ValueError("mouse block rows require positive session and trial counts")
    for column in ("source_added_r2", "shift_source_added_r2"):
        _validate_finite(selected, column, frame_name="mouse_blocks")
    return selected


def _prepare_mouse_effects(
    frame: pl.DataFrame,
    blocks: pl.DataFrame,
    source_region: str,
    target_region: str,
) -> pl.DataFrame:
    required = {
        "subject_id",
        "source_region",
        "target_region",
        "n_sessions",
        "n_trials",
        "real_reversible_contrast",
        "shift_reversible_contrast",
        "controlled_reversible_contrast",
    }
    _require_columns(frame, required, frame_name="mouse_effects")
    selected = (
        frame.filter(
            (pl.col("source_region") == source_region) & (pl.col("target_region") == target_region)
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("source_region").cast(pl.String),
            pl.col("target_region").cast(pl.String),
            pl.col("n_sessions").cast(pl.Int64),
            pl.col("n_trials").cast(pl.Int64),
            pl.col("real_reversible_contrast").cast(pl.Float64),
            pl.col("shift_reversible_contrast").cast(pl.Float64),
            pl.col("controlled_reversible_contrast").cast(pl.Float64),
        )
        .sort("subject_id")
    )
    if selected.is_empty() or selected.get_column("subject_id").n_unique() != selected.height:
        raise ValueError("mouse_effects must contain one nominated-pair row per mouse")
    if set(selected.get_column("subject_id")) != set(blocks.get_column("subject_id")):
        raise ValueError("mouse effects and mouse block rows cover different mice")
    for column in (
        "real_reversible_contrast",
        "shift_reversible_contrast",
        "controlled_reversible_contrast",
    ):
        _validate_finite(selected, column, frame_name="mouse_effects")
    disagreement = selected.filter(
        (
            pl.col("controlled_reversible_contrast")
            - (pl.col("real_reversible_contrast") - pl.col("shift_reversible_contrast"))
        )
        .abs()
        .gt(1e-10)
    )
    if disagreement.height:
        raise ValueError("controlled mouse effects do not reconstruct from real and shift values")
    for effect in selected.iter_rows(named=True):
        mouse_rows = blocks.filter(pl.col("subject_id") == effect["subject_id"])
        lookup = {row["reward_block"]: row for row in mouse_rows.iter_rows(named=True)}
        real = (
            0.5 * lookup["engaged_1"]["source_added_r2"]
            - lookup["no_reward"]["source_added_r2"]
            + 0.5 * lookup["engaged_2"]["source_added_r2"]
        )
        shifted = (
            0.5 * lookup["engaged_1"]["shift_source_added_r2"]
            - lookup["no_reward"]["shift_source_added_r2"]
            + 0.5 * lookup["engaged_2"]["shift_source_added_r2"]
        )
        if not math.isclose(real, effect["real_reversible_contrast"], abs_tol=1e-10):
            raise ValueError("real reversible mouse effect disagrees with block values")
        if not math.isclose(shifted, effect["shift_reversible_contrast"], abs_tol=1e-10):
            raise ValueError("shift reversible mouse effect disagrees with block values")
    return selected


def _prepare_statistic(
    frame: pl.DataFrame,
    effects: pl.DataFrame,
    source_region: str,
    target_region: str,
) -> pl.DataFrame:
    result_id = f"{source_region}_to_{target_region}_controlled_reversible_source_added_r2"
    required = {
        "analysis_id",
        "result_id",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "adjusted_p_value",
        "n_mice",
        "status",
    }
    _require_columns(frame, required, frame_name="statistics")
    selected = frame.filter(pl.col("result_id") == result_id)
    if selected.height != 1:
        raise ValueError("statistics must contain exactly one nominated-pair result")
    row = selected.row(0, named=True)
    if row["analysis_id"] != ANALYSIS_ID:
        raise ValueError("nominated-pair statistic has an unexpected analysis_id")
    for column in ("estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"):
        value = row[column]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"nominated-pair statistic {column} must be finite")
    if not row["ci_low"] <= row["estimate"] <= row["ci_high"]:
        raise ValueError("nominated-pair confidence interval does not contain its estimate")
    if not 0 <= row["p_value"] <= 1 or not 0 <= row["adjusted_p_value"] <= 1:
        raise ValueError("nominated-pair P values must be in [0,1]")
    observed = float(effects.get_column("controlled_reversible_contrast").mean())
    if not math.isclose(float(row["estimate"]), observed, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("nominated-pair statistic disagrees with the plotted mouse mean")
    if row["n_mice"] != effects.height:
        raise ValueError("nominated-pair statistic n_mice disagrees with plotted mice")
    if row["status"] not in ("pass", "fragile", "null"):
        raise ValueError("nominated-pair statistic is not reportable")
    return selected


def _summarize_blocks(blocks: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for index, block in enumerate(dg.network_interactions.REWARD_BLOCKS):
        selected = blocks.filter(pl.col("reward_block") == block)
        interval = dg.statistics.bootstrap_mouse_mean(
            selected.select(
                pl.col("subject_id").alias("mouse_id"),
                pl.col("source_added_r2").alias("value"),
            ),
            seed=BOOTSTRAP_SEED + index,
            value_column="value",
            n_resamples=BOOTSTRAP_RESAMPLES,
        )
        rows.append(
            {
                "reward_block": block,
                "block_order": index,
                "mean_source_added_r2": interval.estimate,
                "ci_low": interval.ci_low,
                "ci_high": interval.ci_high,
                "n_mice": interval.n_mice,
                "n_sessions": int(selected.get_column("n_sessions").sum()),
                "n_trials": int(selected.get_column("n_trials").sum()),
                "aggregation": "equal sessions within mouse, then equal mice",
                "ci_method": "mouse percentile bootstrap",
                "bootstrap_seed": interval.seed,
                "bootstrap_resamples": interval.n_resamples,
            }
        )
    return pl.DataFrame(rows)


def _require_columns(frame: pl.DataFrame, columns: set[str], *, frame_name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} missing columns: {sorted(missing)}")


def _validate_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    if frame.filter(pl.col(column).is_null() | ~pl.col(column).is_finite()).height:
        raise ValueError(f"{frame_name}.{column} must be finite and non-null")
