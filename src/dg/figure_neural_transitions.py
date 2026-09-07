"""Validated source-data preparation for neural-transition Figure 2.

The neural analysis writes one trajectory row per mouse and one controlled
transition effect per mouse.  This module keeps biological replication at the
mouse level, derives only descriptive trajectory intervals, and binds the two
primary effect summaries to the mouse-level values shown in the figure.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import polars as pl

import dg.statistics

TRANSITIONS = (
    ("withdrawal", 1, "Reward withdrawal"),
    ("restoration", 2, "Reward restoration"),
)
ANCHORS = (
    ("real", 1, "Real boundary"),
    ("pseudo", 2, "Within-state pseudo-boundary"),
)
EXPECTED_BIN_CENTERS_MINUTES = (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0)
PRIMARY_STATISTIC_IDS = {
    "withdrawal": "reward_withdrawal_early_axis_real_minus_pseudo_mouse_mean",
    "restoration": "reward_restoration_early_axis_real_minus_pseudo_mouse_mean",
}
ANALYSIS_ID = "neural_transition_axis"
BOOTSTRAP_SEED = 1051
BOOTSTRAP_RESAMPLES = 10_000


@dataclasses.dataclass(frozen=True, slots=True)
class NeuralTransitionFigureData:
    """Publication-facing tables used by neural-transition Figure 2."""

    mouse_trajectories: pl.DataFrame
    trajectory_summary: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistics: pl.DataFrame


def prepare_neural_transition_figure_data(
    trajectories: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    statistics: pl.DataFrame,
) -> NeuralTransitionFigureData:
    """Validate inputs and summarize trajectories with equal mouse weighting."""

    for name, frame in (
        ("trajectories", trajectories),
        ("mouse_effects", mouse_effects),
        ("statistics", statistics),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    prepared_trajectories = _prepare_trajectories(trajectories)
    prepared_mouse_effects = _prepare_mouse_effects(mouse_effects)
    prepared_statistics = _prepare_statistics(statistics, prepared_mouse_effects)
    trajectory_summary = _summarize_trajectories(prepared_trajectories)
    return NeuralTransitionFigureData(
        mouse_trajectories=prepared_trajectories,
        trajectory_summary=trajectory_summary,
        mouse_effects=prepared_mouse_effects,
        statistics=prepared_statistics,
    )


def _prepare_trajectories(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        frame,
        (
            "subject_id",
            "transition",
            "anchor_type",
            "bin_center_minutes",
            "mean_axis_score",
            "n_sessions",
            "n_trials",
        ),
        frame_name="trajectories",
    )
    transition_lookup = _transition_lookup()
    anchor_lookup = _anchor_lookup()
    selected = (
        frame.select(
            pl.col("subject_id").cast(pl.String),
            pl.col("transition").cast(pl.String),
            pl.col("anchor_type").cast(pl.String),
            pl.col("bin_center_minutes").cast(pl.Float64),
            pl.col("mean_axis_score").cast(pl.Float64),
            pl.col("n_sessions").cast(pl.Int64),
            pl.col("n_trials").cast(pl.Int64),
        )
        .join(transition_lookup, on="transition", how="left", validate="m:1")
        .join(anchor_lookup, on="anchor_type", how="left", validate="m:1")
        .sort("transition_order", "anchor_order", "subject_id", "bin_center_minutes")
    )
    if selected.is_empty():
        raise ValueError("trajectories has no rows")
    if selected.filter(
        pl.col("transition_order").is_null() | pl.col("anchor_order").is_null()
    ).height:
        raise ValueError("trajectories contains an unknown transition or anchor type")
    _validate_unique(
        selected,
        ("subject_id", "transition", "anchor_type", "bin_center_minutes"),
        frame_name="trajectories",
    )
    observed_bins = set(selected.get_column("bin_center_minutes").to_list())
    expected_bins = set(EXPECTED_BIN_CENTERS_MINUTES)
    if not observed_bins.issubset(expected_bins):
        raise ValueError("trajectories contains an unexpected two-minute bin center")
    for transition, _, _ in TRANSITIONS:
        for anchor_type, _, _ in ANCHORS:
            rows = selected.filter(
                (pl.col("transition") == transition) & (pl.col("anchor_type") == anchor_type)
            )
            if set(rows.get_column("bin_center_minutes")) != expected_bins:
                raise ValueError(
                    f"{transition}/{anchor_type} lacks the complete -6 to +6 minute bin set"
                )
    if selected.filter((pl.col("n_sessions") < 0) | (pl.col("n_trials") < 0)).height:
        raise ValueError("trajectory denominators must be non-negative")
    invalid_available = selected.filter(
        pl.col("mean_axis_score").is_not_null()
        & ((pl.col("n_sessions") <= 0) | (pl.col("n_trials") <= 0))
    )
    if invalid_available.height:
        raise ValueError("available trajectory scores require positive denominators")
    _validate_optional_finite(selected, "mean_axis_score", frame_name="trajectories")
    return selected


def _prepare_mouse_effects(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        frame,
        (
            "subject_id",
            "transition",
            "real_effect",
            "pseudo_effect",
            "real_minus_pseudo",
            "n_sessions",
        ),
        frame_name="mouse_effects",
    )
    selected = (
        frame.select(
            pl.col("subject_id").cast(pl.String),
            pl.col("transition").cast(pl.String),
            pl.col("real_effect").cast(pl.Float64),
            pl.col("pseudo_effect").cast(pl.Float64),
            pl.col("real_minus_pseudo").cast(pl.Float64),
            pl.col("n_sessions").cast(pl.Int64),
        )
        .join(_transition_lookup(), on="transition", how="left", validate="m:1")
        .sort("transition_order", "subject_id")
    )
    if selected.is_empty():
        raise ValueError("mouse_effects has no rows")
    if selected.filter(pl.col("transition_order").is_null()).height:
        raise ValueError("mouse_effects contains an unknown transition")
    _validate_unique(
        selected,
        ("subject_id", "transition"),
        frame_name="mouse_effects",
    )
    if selected.filter(pl.col("n_sessions") <= 0).height:
        raise ValueError("mouse effects require at least one contributing session")
    for column in ("real_effect", "pseudo_effect", "real_minus_pseudo"):
        _validate_finite(selected, column, frame_name="mouse_effects")
    disagreement = selected.filter(
        (pl.col("real_minus_pseudo") - (pl.col("real_effect") - pl.col("pseudo_effect")))
        .abs()
        .gt(1e-10)
    )
    if disagreement.height:
        raise ValueError("real_minus_pseudo does not reconstruct from the component effects")
    observed = set(selected.get_column("transition"))
    expected = {row[0] for row in TRANSITIONS}
    if observed != expected:
        raise ValueError("mouse_effects must contain both planned transitions")
    return selected


def _prepare_statistics(frame: pl.DataFrame, mouse_effects: pl.DataFrame) -> pl.DataFrame:
    if "adjusted_p_value" not in frame.columns and "p_adjusted" in frame.columns:
        frame = frame.rename({"p_adjusted": "adjusted_p_value"})
    _require_columns(
        frame,
        ("analysis_id", "estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"),
        frame_name="statistics",
    )
    statistic_identity = (
        pl.col("result_id") if "result_id" in frame.columns else pl.col("analysis_id")
    )
    wanted_ids = list(PRIMARY_STATISTIC_IDS.values())
    selected = frame.with_columns(statistic_identity.cast(pl.String).alias("statistic_id")).filter(
        pl.col("statistic_id").is_in(wanted_ids)
    )
    if selected.height != len(wanted_ids):
        raise ValueError(
            "statistics must contain exactly one row for each primary transition effect"
        )
    _validate_unique(selected, ("statistic_id",), frame_name="statistics")
    if (
        "result_id" in frame.columns
        and selected.filter(pl.col("analysis_id") != ANALYSIS_ID).height
    ):
        raise ValueError("primary statistics have an unexpected analysis_id")

    transition_from_id = {value: key for key, value in PRIMARY_STATISTIC_IDS.items()}
    expressions = [
        pl.col("analysis_id").cast(pl.String),
        pl.col("statistic_id"),
        pl.col("statistic_id").replace_strict(transition_from_id).alias("transition"),
        pl.col("estimate").cast(pl.Float64),
        pl.col("ci_low").cast(pl.Float64),
        pl.col("ci_high").cast(pl.Float64),
        pl.col("p_value").cast(pl.Float64),
        pl.col("adjusted_p_value").cast(pl.Float64),
    ]
    for name, dtype in (
        ("confidence_level", pl.Float64),
        ("adjustment_method", pl.String),
        ("multiplicity_family", pl.String),
        ("n_mice", pl.Int64),
        ("n_sessions", pl.Int64),
        ("status", pl.String),
    ):
        expressions.append(
            pl.col(name).cast(dtype)
            if name in selected.columns
            else pl.lit(None, dtype=dtype).alias(name)
        )
    selected = (
        selected.select(*expressions)
        .join(_transition_lookup(), on="transition", validate="1:1")
        .sort("transition_order")
    )
    for column in ("estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"):
        _validate_finite(selected, column, frame_name="statistics")
    if selected.filter(
        (pl.col("ci_low") > pl.col("estimate"))
        | (pl.col("ci_high") < pl.col("estimate"))
        | (pl.col("p_value") < 0.0)
        | (pl.col("p_value") > 1.0)
        | (pl.col("adjusted_p_value") < 0.0)
        | (pl.col("adjusted_p_value") > 1.0)
    ).height:
        raise ValueError("primary statistics contain an invalid interval or P value")
    if selected.filter(
        pl.col("status").is_not_null() & ~pl.col("status").is_in(("pass", "null"))
    ).height:
        raise ValueError("a primary transition statistic is not reportable")
    if selected.filter(
        pl.col("adjustment_method").is_not_null() & (pl.col("adjustment_method") != "holm")
    ).height:
        raise ValueError("primary transition P values must use the planned Holm adjustment")

    for row in selected.iter_rows(named=True):
        effects = mouse_effects.filter(pl.col("transition") == row["transition"])
        observed_mean = float(effects.get_column("real_minus_pseudo").mean())
        if not math.isclose(row["estimate"], observed_mean, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("a primary estimate disagrees with the plotted mouse mean")
        if row["n_mice"] is not None and row["n_mice"] != effects.height:
            raise ValueError("a primary n_mice disagrees with the plotted mouse rows")
    return selected


def _summarize_trajectories(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    group_index = 0
    for transition, transition_order, transition_label in TRANSITIONS:
        for anchor_type, anchor_order, anchor_label in ANCHORS:
            for bin_center in EXPECTED_BIN_CENTERS_MINUTES:
                selected = frame.filter(
                    (pl.col("transition") == transition)
                    & (pl.col("anchor_type") == anchor_type)
                    & (pl.col("bin_center_minutes") == bin_center)
                    & pl.col("mean_axis_score").is_not_null()
                )
                values = selected.get_column("mean_axis_score").to_numpy()
                if values.size == 0:
                    raise ValueError(
                        f"no mouse contributes to {transition}/{anchor_type}/{bin_center:g} min"
                    )
                interval = dg.statistics.bootstrap_mouse_mean(
                    selected.select(
                        pl.col("subject_id").alias("mouse_id"),
                        pl.col("mean_axis_score").alias("value"),
                    ),
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    seed=BOOTSTRAP_SEED + group_index,
                    value_column="value",
                )
                group_index += 1
                rows.append(
                    {
                        "transition": transition,
                        "transition_order": transition_order,
                        "transition_label": transition_label,
                        "anchor_type": anchor_type,
                        "anchor_order": anchor_order,
                        "anchor_label": anchor_label,
                        "bin_center_minutes": bin_center,
                        "mean_axis_score": interval.estimate,
                        "ci_low": interval.ci_low,
                        "ci_high": interval.ci_high,
                        "n_mice": interval.n_mice,
                        "n_sessions": int(selected.get_column("n_sessions").sum()),
                        "n_trials": int(selected.get_column("n_trials").sum()),
                        "aggregation": "equal sessions within mouse, then equal mice",
                        "ci_method": "mouse percentile bootstrap",
                        "confidence_level": interval.confidence_level,
                        "bootstrap_resamples": interval.n_resamples,
                        "bootstrap_seed": interval.seed,
                    }
                )
    return pl.DataFrame(rows).sort("transition_order", "anchor_order", "bin_center_minutes")


def _transition_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transition": [row[0] for row in TRANSITIONS],
            "transition_order": [row[1] for row in TRANSITIONS],
            "transition_label": [row[2] for row in TRANSITIONS],
        }
    )


def _anchor_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "anchor_type": [row[0] for row in ANCHORS],
            "anchor_order": [row[1] for row in ANCHORS],
            "anchor_label": [row[2] for row in ANCHORS],
        }
    )


def _require_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def _validate_unique(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    if frame.select(*columns).n_unique() != frame.height:
        raise ValueError(f"{frame_name} contains duplicate rows for {columns}")


def _validate_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    values = frame.get_column(column)
    if values.null_count() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{frame_name}.{column} must contain finite values")


def _validate_optional_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    values = frame.get_column(column).drop_nulls()
    if values.len() and not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{frame_name}.{column} must contain only finite values or nulls")
