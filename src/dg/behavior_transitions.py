"""Transition-resolved behavioral analyses for Dynamic Gating.

The functions in this module operate only on the audited behavioral trial
table and session-QC metadata.  They retain a complete session-by-bin grid,
represent empty or uncovered bins explicitly, average sessions equally within
mouse, and never read neural activity.

Two reward-state transitions are treated as distinct processes:

``reward_withdrawal``
    The E1 -> no-reward boundary, signed so response suppression is positive.
    Its source-state pseudo-boundary negative control is the temporal midpoint
    of E1.

``reward_restoration``
    The no-reward -> E2 boundary, signed so response recovery is positive.
    Its source-state pseudo-boundary negative control is the temporal midpoint
    of no reward.

Each control is matched within session on window geometry and source state.
Separate controls avoid assuming identical extinction and reacquisition
kinetics, but a single midpoint cannot rule out nonlinear within-block drift.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import polars as pl

import dg.quality
import dg.statistics

SOURCE_COLUMN = "_nwb_path"
MOUSE_COLUMN = "subject_id"

TRANSITION_IDS = ("reward_withdrawal", "reward_restoration")
ANCHOR_TYPES = ("real", "pseudo")
CONDITIONS = ("go", "catch")
EFFECT_CONDITIONS = (*CONDITIONS, "go_minus_catch")


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionConfig:
    """Declared exploratory resolution and effect-window implementation.

    Time bins are half-open intervals relative to each boundary. Trial bins
    use the ordinal position among all completed, non-auto-rewarded go/catch
    events on each side of the boundary, then summarize go and catch trials
    separately. The effect window is independent of the display resolution.
    """

    time_window_seconds: float = 360.0
    time_bin_width_seconds: float = 120.0
    trial_window_count: int = 24
    trial_bin_width: int = 8
    effect_window_seconds: float = 180.0
    minimum_go_trials_per_effect_window: int = 5
    minimum_catch_trials_per_effect_window: int = 3

    def __post_init__(self) -> None:
        for name in (
            "time_window_seconds",
            "time_bin_width_seconds",
            "effect_window_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "trial_window_count",
            "trial_bin_width",
            "minimum_go_trials_per_effect_window",
            "minimum_catch_trials_per_effect_window",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        time_bins = self.time_window_seconds / self.time_bin_width_seconds
        if not time_bins.is_integer():
            raise ValueError("time_window_seconds must be divisible by time_bin_width_seconds")
        if self.trial_window_count % self.trial_bin_width:
            raise ValueError("trial_window_count must be divisible by trial_bin_width")
        if self.effect_window_seconds > self.time_window_seconds:
            raise ValueError("effect_window_seconds cannot exceed time_window_seconds")


DEFAULT_TRANSITION_CONFIG = TransitionConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionTables:
    """Complete transition tables returned by :func:`build_transition_tables`."""

    boundaries: pl.DataFrame
    session_time_bins: pl.DataFrame
    mouse_time_bins: pl.DataFrame
    session_trial_bins: pl.DataFrame
    mouse_trial_bins: pl.DataFrame
    session_effect_windows: pl.DataFrame
    effect_window_count_summary: pl.DataFrame
    session_effects: pl.DataFrame
    mouse_effects: pl.DataFrame
    effect_statistic_sample_sizes: pl.DataFrame
    qc_criterion_summary: pl.DataFrame
    qc_metric_distributions: pl.DataFrame


def build_transition_tables(
    trials: pl.DataFrame | pl.LazyFrame,
    session_qc: pl.DataFrame | pl.LazyFrame,
    *,
    config: TransitionConfig = DEFAULT_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> TransitionTables:
    """Build transition trajectories, effects, and reviewer-facing D03 QC.

    All inventoried session rows in ``session_qc`` are retained. Sessions that
    are not technically valid, bins outside observed block coverage, and bins
    without a condition trial remain in the output with null estimates and an
    explicit status. Mouse trajectories are arithmetic means of session-level
    estimates, so sessions—not trials—receive equal weight within a mouse.
    """

    trial_data = _prepare_trials(
        trials,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    qc_data = _prepare_session_qc(
        session_qc,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    _validate_trial_session_alignment(
        trial_data,
        qc_data,
        source_column=source_column,
        mouse_column=mouse_column,
    )

    boundaries = build_transition_boundaries(
        trial_data,
        qc_data,
        config=config,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    eligible = _eligible_event_trials(
        trial_data,
        qc_data,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    session_time_bins = build_session_time_bins(
        eligible,
        boundaries,
        config=config,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    mouse_time_bins = aggregate_mouse_bins(
        session_time_bins,
        bin_kind="time",
        source_column=source_column,
        mouse_column=mouse_column,
    )
    session_trial_bins = build_session_trial_bins(
        eligible,
        boundaries,
        config=config,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    mouse_trial_bins = aggregate_mouse_bins(
        session_trial_bins,
        bin_kind="trial",
        source_column=source_column,
        mouse_column=mouse_column,
    )
    session_effect_windows = build_session_effect_windows(
        eligible,
        boundaries,
        config=config,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    effect_window_count_summary = summarize_effect_window_counts(
        session_effect_windows,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    session_effects = build_session_transition_effects(
        session_effect_windows,
        boundaries,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    mouse_effects = aggregate_mouse_effects(
        session_effects,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    effect_statistic_sample_sizes = build_effect_statistic_sample_sizes(
        mouse_effects,
        mouse_column=mouse_column,
    )
    criterion_summary, metric_distributions = build_d03_qc_tables(
        qc_data,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    return TransitionTables(
        boundaries=boundaries,
        session_time_bins=session_time_bins,
        mouse_time_bins=mouse_time_bins,
        session_trial_bins=session_trial_bins,
        mouse_trial_bins=mouse_trial_bins,
        session_effect_windows=session_effect_windows,
        effect_window_count_summary=effect_window_count_summary,
        session_effects=session_effects,
        mouse_effects=mouse_effects,
        effect_statistic_sample_sizes=effect_statistic_sample_sizes,
        qc_criterion_summary=criterion_summary,
        qc_metric_distributions=metric_distributions,
    )


def build_transition_boundaries(
    trials: pl.DataFrame,
    session_qc: pl.DataFrame,
    *,
    config: TransitionConfig = DEFAULT_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return audited real boundaries and midpoint negative controls by session.

    The real withdrawal boundary is the first no-reward trial start, matching
    :func:`dg.quality.label_reward_blocks`. The real restoration boundary is
    the final no-reward trial stop. Pseudo-boundaries are temporal midpoints of
    the corresponding source state (E1 and no reward, respectively).
    """

    _require_columns(
        trials,
        (source_column, "reward_block", "start_time", "stop_time"),
        frame_name="trials",
    )
    _require_columns(
        session_qc,
        (source_column, mouse_column, "is_technically_valid", "is_good_session"),
        frame_name="session_qc",
    )
    block_bounds = trials.group_by(source_column).agg(
        *(
            expression
            for block in ("engaged_1", "no_reward", "engaged_2")
            for expression in (
                pl.col("start_time")
                .filter(pl.col("reward_block") == block)
                .min()
                .alias(f"{block}_start_time"),
                pl.col("stop_time")
                .filter(pl.col("reward_block") == block)
                .max()
                .alias(f"{block}_stop_time"),
            )
        )
    )
    metadata_columns = [
        source_column,
        mouse_column,
        "is_technically_valid",
        "is_good_session",
    ]
    metadata_columns.extend(
        column
        for column in ("ecephys_session_id", "recording_day", "session_number")
        if column in session_qc.columns
    )
    sessions = session_qc.select(metadata_columns).join(
        block_bounds,
        on=source_column,
        how="left",
        validate="1:1",
    )
    common = [
        source_column,
        mouse_column,
        *(
            column
            for column in ("ecephys_session_id", "recording_day", "session_number")
            if column in sessions.columns
        ),
        "is_technically_valid",
        "is_good_session",
        "engaged_1_start_time",
        "engaged_1_stop_time",
        "no_reward_start_time",
        "no_reward_stop_time",
        "engaged_2_start_time",
        "engaged_2_stop_time",
    ]
    withdrawal = sessions.select(
        *common,
        pl.lit("reward_withdrawal").alias("transition_id"),
        pl.lit(1, dtype=pl.Int8).alias("transition_order"),
        pl.lit("engaged_1").alias("pre_block"),
        pl.lit("no_reward").alias("post_block"),
        pl.lit("engaged_1").alias("pseudo_block"),
        pl.lit(-1.0).alias("effect_direction_multiplier"),
        pl.col("no_reward_start_time").alias("real_boundary_time"),
        ((pl.col("engaged_1_start_time") + pl.col("engaged_1_stop_time")) / 2).alias(
            "pseudo_boundary_time"
        ),
        pl.col("engaged_1_start_time").alias("real_pre_coverage_start"),
        pl.col("no_reward_start_time").alias("real_pre_coverage_stop"),
        pl.col("no_reward_start_time").alias("real_post_coverage_start"),
        pl.col("no_reward_stop_time").alias("real_post_coverage_stop"),
        pl.col("engaged_1_start_time").alias("pseudo_coverage_start"),
        pl.col("engaged_1_stop_time").alias("pseudo_coverage_stop"),
    )
    restoration = sessions.select(
        *common,
        pl.lit("reward_restoration").alias("transition_id"),
        pl.lit(2, dtype=pl.Int8).alias("transition_order"),
        pl.lit("no_reward").alias("pre_block"),
        pl.lit("engaged_2").alias("post_block"),
        pl.lit("no_reward").alias("pseudo_block"),
        pl.lit(1.0).alias("effect_direction_multiplier"),
        pl.col("no_reward_stop_time").alias("real_boundary_time"),
        ((pl.col("no_reward_start_time") + pl.col("no_reward_stop_time")) / 2).alias(
            "pseudo_boundary_time"
        ),
        pl.col("no_reward_start_time").alias("real_pre_coverage_start"),
        pl.col("no_reward_stop_time").alias("real_pre_coverage_stop"),
        pl.col("no_reward_stop_time").alias("real_post_coverage_start"),
        pl.col("engaged_2_stop_time").alias("real_post_coverage_stop"),
        pl.col("no_reward_start_time").alias("pseudo_coverage_start"),
        pl.col("no_reward_stop_time").alias("pseudo_coverage_stop"),
    )
    boundaries = pl.concat([withdrawal, restoration], how="vertical")
    bound_columns = (
        "engaged_1_start_time",
        "engaged_1_stop_time",
        "no_reward_start_time",
        "no_reward_stop_time",
        "engaged_2_start_time",
        "engaged_2_stop_time",
        "real_boundary_time",
        "pseudo_boundary_time",
    )
    finite_bounds = pl.all_horizontal(
        *(pl.col(column).is_not_null() & pl.col(column).is_finite() for column in bound_columns)
    )
    ordered_bounds = (
        (pl.col("engaged_1_start_time") < pl.col("engaged_1_stop_time"))
        & (pl.col("engaged_1_stop_time") <= pl.col("no_reward_start_time"))
        & (pl.col("no_reward_start_time") < pl.col("no_reward_stop_time"))
        & (pl.col("no_reward_stop_time") <= pl.col("engaged_2_start_time"))
        & (pl.col("engaged_2_start_time") < pl.col("engaged_2_stop_time"))
    )
    real_window_complete = (
        (pl.col("real_boundary_time") - config.time_window_seconds)
        >= pl.col("real_pre_coverage_start")
    ) & (
        (pl.col("real_boundary_time") + config.time_window_seconds)
        <= pl.col("real_post_coverage_stop")
    )
    pseudo_window_complete = (
        (pl.col("pseudo_boundary_time") - config.time_window_seconds)
        >= pl.col("pseudo_coverage_start")
    ) & (
        (pl.col("pseudo_boundary_time") + config.time_window_seconds)
        <= pl.col("pseudo_coverage_stop")
    )
    return (
        boundaries.with_columns(
            (finite_bounds & ordered_bounds).fill_null(False).alias("boundary_values_valid"),
            (-pl.col("effect_direction_multiplier")).alias("latency_direction_multiplier"),
            real_window_complete.fill_null(False).alias("real_time_window_complete"),
            pseudo_window_complete.fill_null(False).alias("pseudo_time_window_complete"),
            pl.lit(config.time_window_seconds).alias("time_window_seconds"),
            pl.lit(config.time_bin_width_seconds).alias("time_bin_width_seconds"),
            pl.lit(config.trial_window_count, dtype=pl.Int32).alias("trial_window_count"),
            pl.lit(config.trial_bin_width, dtype=pl.Int32).alias("trial_bin_width"),
            pl.lit(config.effect_window_seconds).alias("effect_window_seconds"),
            pl.lit(config.minimum_go_trials_per_effect_window, dtype=pl.Int32).alias(
                "minimum_go_trials_per_effect_window"
            ),
            pl.lit(config.minimum_catch_trials_per_effect_window, dtype=pl.Int32).alias(
                "minimum_catch_trials_per_effect_window"
            ),
            pl.lit("temporal_midpoint_of_source_reward_state").alias("pseudo_boundary_definition"),
        )
        .with_columns(
            (
                pl.col("is_technically_valid").fill_null(False) & pl.col("boundary_values_valid")
            ).alias("included_in_transition_analysis"),
            pl.when(~pl.col("is_technically_valid").fill_null(False))
            .then(pl.lit("session_not_technically_valid"))
            .when(~pl.col("boundary_values_valid"))
            .then(pl.lit("invalid_or_incomplete_reward_block_boundaries"))
            .when(~pl.col("real_time_window_complete") | ~pl.col("pseudo_time_window_complete"))
            .then(pl.lit("included_with_incomplete_outer_time_bins"))
            .otherwise(pl.lit("included"))
            .alias("boundary_status"),
        )
        .sort(mouse_column, source_column, "transition_order")
    )


def build_session_time_bins(
    eligible_trials: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    config: TransitionConfig = DEFAULT_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return a complete session × transition × anchor × time-bin grid."""

    anchors = _anchor_table(boundaries, source_column=source_column, mouse_column=mouse_column)
    n_bins_each_side = int(config.time_window_seconds / config.time_bin_width_seconds)
    bins = (
        pl.DataFrame(
            {
                "bin_index": list(range(-n_bins_each_side, n_bins_each_side)),
            },
            schema={"bin_index": pl.Int32},
        )
        .with_columns(
            (pl.col("bin_index") * config.time_bin_width_seconds).alias(
                "relative_bin_start_seconds"
            ),
            ((pl.col("bin_index") + 1) * config.time_bin_width_seconds).alias(
                "relative_bin_stop_seconds"
            ),
        )
        .with_columns(
            (
                (pl.col("relative_bin_start_seconds") + pl.col("relative_bin_stop_seconds")) / 2
            ).alias("relative_bin_center_seconds")
        )
    )
    conditions = pl.DataFrame({"condition": CONDITIONS})
    grid = (
        anchors.join(bins, how="cross")
        .join(conditions, how="cross")
        .with_columns(
            (pl.col("anchor_time") + pl.col("relative_bin_start_seconds")).alias(
                "absolute_bin_start_time"
            ),
            (pl.col("anchor_time") + pl.col("relative_bin_stop_seconds")).alias(
                "absolute_bin_stop_time"
            ),
        )
    )
    grid = grid.with_columns(
        pl.when(pl.col("bin_index") < 0)
        .then(pl.col("anchor_pre_block"))
        .otherwise(pl.col("anchor_post_block"))
        .alias("analysis_block"),
        (
            (pl.col("absolute_bin_start_time") >= pl.col("anchor_coverage_start"))
            & (pl.col("absolute_bin_stop_time") <= pl.col("anchor_coverage_stop"))
        ).alias("bin_fully_covered"),
    )

    joined = eligible_trials.join(
        anchors.select(
            source_column,
            "transition_id",
            "anchor_type",
            "anchor_time",
            "anchor_pre_block",
            "anchor_post_block",
            "included_in_transition_analysis",
        ),
        on=source_column,
        how="inner",
        validate="m:m",
    ).with_columns((pl.col("change_time") - pl.col("anchor_time")).alias("relative_time"))
    allowed_block = (
        pl.when(pl.col("relative_time") < 0)
        .then(pl.col("anchor_pre_block"))
        .otherwise(pl.col("anchor_post_block"))
    )
    contributions = (
        joined.filter(
            pl.col("included_in_transition_analysis")
            & (pl.col("reward_block") == allowed_block)
            & (pl.col("relative_time") >= -config.time_window_seconds)
            & (pl.col("relative_time") < config.time_window_seconds)
        )
        .with_columns(
            (pl.col("relative_time") / config.time_bin_width_seconds)
            .floor()
            .cast(pl.Int32)
            .alias("bin_index")
        )
        .group_by(source_column, "transition_id", "anchor_type", "bin_index", "condition")
        .agg(*_trial_summary_expressions())
    )
    result = grid.join(
        contributions,
        on=(source_column, "transition_id", "anchor_type", "bin_index", "condition"),
        how="left",
        validate="1:1",
    )
    return _finish_session_bins(
        result,
        bin_kind="time",
        source_column=source_column,
        mouse_column=mouse_column,
    )


def build_session_trial_bins(
    eligible_trials: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    config: TransitionConfig = DEFAULT_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return fixed event-trial-ordinal bins around real and pseudo boundaries.

    Ordinals are assigned across all eligible event trials before stratifying
    go and catch, preventing the two conditions from acquiring incomparable
    trial axes. Negative ordinals count backward from a boundary and positive
    ordinals count forward; ordinal zero does not exist.
    """

    anchors = _anchor_table(boundaries, source_column=source_column, mouse_column=mouse_column)
    n_bins_each_side = config.trial_window_count // config.trial_bin_width
    bin_indices = tuple(range(-n_bins_each_side, 0)) + tuple(range(1, n_bins_each_side + 1))
    bins = pl.DataFrame({"bin_index": bin_indices}, schema={"bin_index": pl.Int32})
    bins = bins.with_columns(
        pl.when(pl.col("bin_index") < 0)
        .then(pl.col("bin_index") * config.trial_bin_width)
        .otherwise((pl.col("bin_index") - 1) * config.trial_bin_width + 1)
        .cast(pl.Int32)
        .alias("relative_trial_start"),
        pl.when(pl.col("bin_index") < 0)
        .then((pl.col("bin_index") + 1) * config.trial_bin_width - 1)
        .otherwise(pl.col("bin_index") * config.trial_bin_width)
        .cast(pl.Int32)
        .alias("relative_trial_stop"),
    ).with_columns(
        ((pl.col("relative_trial_start") + pl.col("relative_trial_stop")) / 2).alias(
            "relative_trial_center"
        )
    )
    conditions = pl.DataFrame({"condition": CONDITIONS})
    grid = (
        anchors.join(bins, how="cross")
        .join(conditions, how="cross")
        .with_columns(
            pl.when(pl.col("bin_index") < 0)
            .then(pl.col("anchor_pre_block"))
            .otherwise(pl.col("anchor_post_block"))
            .alias("analysis_block")
        )
    )

    joined = eligible_trials.join(
        anchors.select(
            source_column,
            "transition_id",
            "anchor_type",
            "anchor_time",
            "anchor_pre_block",
            "anchor_post_block",
            "included_in_transition_analysis",
        ),
        on=source_column,
        how="inner",
        validate="m:m",
    ).with_columns(
        pl.when(pl.col("change_time") < pl.col("anchor_time"))
        .then(pl.lit("pre"))
        .otherwise(pl.lit("post"))
        .alias("boundary_side")
    )
    allowed_block = (
        pl.when(pl.col("boundary_side") == "pre")
        .then(pl.col("anchor_pre_block"))
        .otherwise(pl.col("anchor_post_block"))
    )
    joined = joined.filter(
        pl.col("included_in_transition_analysis") & (pl.col("reward_block") == allowed_block)
    ).sort(
        source_column,
        "transition_id",
        "anchor_type",
        "boundary_side",
        "change_time",
        "_table_index",
    )
    rank_keys = (source_column, "transition_id", "anchor_type", "boundary_side")
    joined = joined.with_columns(
        pl.when(pl.col("boundary_side") == "pre")
        .then(
            -pl.col("change_time")
            .rank(method="ordinal", descending=True)
            .over(*rank_keys)
            .cast(pl.Int32)
        )
        .otherwise(pl.col("change_time").rank(method="ordinal").over(*rank_keys).cast(pl.Int32))
        .alias("relative_trial_index")
    ).filter(pl.col("relative_trial_index").abs() <= config.trial_window_count)
    joined = joined.with_columns(
        pl.when(pl.col("relative_trial_index") < 0)
        .then(-((pl.col("relative_trial_index").abs() - 1) // config.trial_bin_width + 1))
        .otherwise((pl.col("relative_trial_index") - 1) // config.trial_bin_width + 1)
        .cast(pl.Int32)
        .alias("bin_index")
    )
    coverage = joined.group_by(source_column, "transition_id", "anchor_type", "bin_index").agg(
        pl.len().cast(pl.Int64).alias("n_event_trials_in_bin")
    )
    contributions = joined.group_by(
        source_column, "transition_id", "anchor_type", "bin_index", "condition"
    ).agg(*_trial_summary_expressions())
    result = (
        grid.join(
            coverage,
            on=(source_column, "transition_id", "anchor_type", "bin_index"),
            how="left",
            validate="m:1",
        )
        .join(
            contributions,
            on=(source_column, "transition_id", "anchor_type", "bin_index", "condition"),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("n_event_trials_in_bin").fill_null(0),
            (pl.col("n_event_trials_in_bin").fill_null(0) == config.trial_bin_width).alias(
                "bin_fully_covered"
            ),
        )
    )
    return _finish_session_bins(
        result,
        bin_kind="trial",
        source_column=source_column,
        mouse_column=mouse_column,
    )


def aggregate_mouse_bins(
    session_bins: pl.DataFrame,
    *,
    bin_kind: str,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Average session-bin estimates equally within mouse, preserving nulls."""

    if bin_kind not in {"time", "trial"}:
        raise ValueError("bin_kind must be 'time' or 'trial'")
    common = [
        "transition_id",
        "transition_order",
        "anchor_type",
        "condition",
        "bin_index",
        "analysis_block",
    ]
    bin_columns = (
        [
            "relative_bin_start_seconds",
            "relative_bin_stop_seconds",
            "relative_bin_center_seconds",
        ]
        if bin_kind == "time"
        else ["relative_trial_start", "relative_trial_stop", "relative_trial_center"]
    )
    group_columns = [mouse_column, *common, *bin_columns]
    result = (
        session_bins.group_by(*group_columns)
        .agg(
            pl.col(source_column).n_unique().cast(pl.Int64).alias("n_sessions_inventory"),
            pl.col(source_column)
            .filter(pl.col("included_in_transition_analysis"))
            .n_unique()
            .cast(pl.Int64)
            .alias("n_sessions_in_analysis"),
            pl.col(source_column)
            .filter(pl.col("response_probability").is_not_null())
            .n_unique()
            .cast(pl.Int64)
            .alias("n_sessions_response_contributing"),
            pl.col("response_probability").mean(),
            pl.col("n_trials")
            .filter(pl.col("response_probability").is_not_null())
            .sum()
            .cast(pl.Int64)
            .alias("n_trials_contributing"),
            pl.col("n_responses")
            .filter(pl.col("response_probability").is_not_null())
            .sum()
            .cast(pl.Int64)
            .alias("n_responses_contributing"),
            pl.col(source_column)
            .filter(pl.col("median_response_latency_seconds").is_not_null())
            .n_unique()
            .cast(pl.Int64)
            .alias("n_sessions_latency_contributing"),
            pl.col("median_response_latency_seconds")
            .mean()
            .alias("mean_session_median_response_latency_seconds"),
            pl.col("n_latency_responses")
            .filter(pl.col("median_response_latency_seconds").is_not_null())
            .sum()
            .cast(pl.Int64)
            .alias("n_latency_responses_contributing"),
        )
        .with_columns(
            (pl.col("n_sessions_in_analysis") - pl.col("n_sessions_response_contributing")).alias(
                "n_sessions_response_missing"
            ),
            (pl.col("n_sessions_in_analysis") - pl.col("n_sessions_latency_contributing")).alias(
                "n_sessions_latency_missing"
            ),
            pl.when(pl.col("n_sessions_in_analysis") == 0)
            .then(pl.lit("no_included_sessions"))
            .when(pl.col("n_sessions_response_contributing") == 0)
            .then(pl.lit("no_condition_trials"))
            .otherwise(pl.lit("observed"))
            .alias("mouse_bin_status"),
            pl.lit("equal_session_mean").alias("within_mouse_aggregation"),
        )
        .sort(mouse_column, "transition_order", "anchor_type", "condition", "bin_index")
    )
    return result


def build_session_effect_windows(
    eligible_trials: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    config: TransitionConfig = DEFAULT_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Summarize fixed ±effect windows for real and pseudo-boundaries."""

    anchors = _anchor_table(boundaries, source_column=source_column, mouse_column=mouse_column)
    phases = pl.DataFrame({"phase": ("pre", "post")})
    conditions = pl.DataFrame({"condition": CONDITIONS})
    grid = anchors.join(phases, how="cross").join(conditions, how="cross")
    grid = grid.with_columns(
        pl.when(pl.col("phase") == "pre")
        .then(pl.col("anchor_pre_block"))
        .otherwise(pl.col("anchor_post_block"))
        .alias("analysis_block"),
        pl.when(pl.col("phase") == "pre")
        .then(pl.col("anchor_time") - config.effect_window_seconds)
        .otherwise(pl.col("anchor_time"))
        .alias("window_start_time"),
        pl.when(pl.col("phase") == "pre")
        .then(pl.col("anchor_time"))
        .otherwise(pl.col("anchor_time") + config.effect_window_seconds)
        .alias("window_stop_time"),
    ).with_columns(
        (
            (pl.col("window_start_time") >= pl.col("anchor_coverage_start"))
            & (pl.col("window_stop_time") <= pl.col("anchor_coverage_stop"))
        ).alias("window_fully_covered")
    )

    joined = eligible_trials.join(
        anchors.select(
            source_column,
            "transition_id",
            "anchor_type",
            "anchor_time",
            "anchor_pre_block",
            "anchor_post_block",
            "included_in_transition_analysis",
        ),
        on=source_column,
        how="inner",
        validate="m:m",
    ).with_columns(
        (pl.col("change_time") - pl.col("anchor_time")).alias("relative_time"),
        pl.when(pl.col("change_time") < pl.col("anchor_time"))
        .then(pl.lit("pre"))
        .otherwise(pl.lit("post"))
        .alias("phase"),
    )
    allowed_block = (
        pl.when(pl.col("phase") == "pre")
        .then(pl.col("anchor_pre_block"))
        .otherwise(pl.col("anchor_post_block"))
    )
    contributions = (
        joined.filter(
            pl.col("included_in_transition_analysis")
            & (pl.col("reward_block") == allowed_block)
            & (pl.col("relative_time") >= -config.effect_window_seconds)
            & (pl.col("relative_time") < config.effect_window_seconds)
        )
        .group_by(source_column, "transition_id", "anchor_type", "phase", "condition")
        .agg(*_trial_summary_expressions())
    )
    result = grid.join(
        contributions,
        on=(source_column, "transition_id", "anchor_type", "phase", "condition"),
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("n_trials").fill_null(0),
        pl.col("n_responses").fill_null(0),
        pl.col("n_latency_responses").fill_null(0),
        pl.when(pl.col("condition") == "go")
        .then(pl.lit(config.minimum_go_trials_per_effect_window))
        .otherwise(pl.lit(config.minimum_catch_trials_per_effect_window))
        .cast(pl.Int32)
        .alias("minimum_trials_required"),
    )
    estimate_allowed = (
        pl.col("included_in_transition_analysis")
        & pl.col("window_fully_covered")
        & (pl.col("n_trials") >= pl.col("minimum_trials_required"))
    )
    return (
        result.with_columns(
            pl.when(estimate_allowed)
            .then(pl.col("n_responses") / pl.col("n_trials"))
            .otherwise(None)
            .alias("response_probability"),
            pl.when(estimate_allowed & (pl.col("n_latency_responses") > 0))
            .then(pl.col("_median_response_latency_seconds"))
            .otherwise(None)
            .alias("median_response_latency_seconds"),
            pl.when(~pl.col("included_in_transition_analysis"))
            .then(pl.lit("excluded_session"))
            .when(~pl.col("window_fully_covered"))
            .then(pl.lit("outside_observed_block_coverage"))
            .when(pl.col("n_trials") == 0)
            .then(pl.lit("no_condition_trials"))
            .when(pl.col("n_trials") < pl.col("minimum_trials_required"))
            .then(pl.lit("insufficient_condition_trials"))
            .otherwise(pl.lit("observed"))
            .alias("window_status"),
        )
        .drop("_median_response_latency_seconds")
        .sort(
            mouse_column,
            source_column,
            "transition_order",
            "anchor_type",
            "condition",
            "phase",
        )
    )


def build_session_transition_effects(
    effect_windows: pl.DataFrame,
    boundaries: pl.DataFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return real, pseudo, and pseudo-controlled effects for each session."""

    metadata = boundaries.select(
        source_column,
        mouse_column,
        "transition_id",
        "transition_order",
        "effect_direction_multiplier",
        "latency_direction_multiplier",
        "included_in_transition_analysis",
    ).unique(maintain_order=True)
    base = metadata.join(pl.DataFrame({"condition": CONDITIONS}), how="cross")
    value_columns = (
        "response_probability",
        "median_response_latency_seconds",
        "n_trials",
        "n_latency_responses",
    )
    effects = base
    for anchor_type in ANCHOR_TYPES:
        for phase in ("pre", "post"):
            prefix = f"{anchor_type}_{phase}"
            window = effect_windows.filter(
                (pl.col("anchor_type") == anchor_type) & (pl.col("phase") == phase)
            ).select(
                source_column,
                "transition_id",
                "condition",
                *(pl.col(column).alias(f"{prefix}_{column}") for column in value_columns),
            )
            effects = effects.join(
                window,
                on=(source_column, "transition_id", "condition"),
                how="left",
                validate="1:1",
            )

    direction = pl.col("effect_direction_multiplier")
    latency_direction = pl.col("latency_direction_multiplier")
    effects = effects.with_columns(
        (
            direction
            * (pl.col("real_post_response_probability") - pl.col("real_pre_response_probability"))
        ).alias("real_step"),
        (
            direction
            * (
                pl.col("pseudo_post_response_probability")
                - pl.col("pseudo_pre_response_probability")
            )
        ).alias("pseudo_step"),
        (
            latency_direction
            * (
                pl.col("real_post_median_response_latency_seconds")
                - pl.col("real_pre_median_response_latency_seconds")
            )
        ).alias("real_latency_step_seconds"),
        (
            latency_direction
            * (
                pl.col("pseudo_post_median_response_latency_seconds")
                - pl.col("pseudo_pre_median_response_latency_seconds")
            )
        ).alias("pseudo_latency_step_seconds"),
        (pl.col("real_pre_n_trials") + pl.col("real_post_n_trials")).alias("n_real_window_trials"),
        (pl.col("pseudo_pre_n_trials") + pl.col("pseudo_post_n_trials")).alias(
            "n_pseudo_window_trials"
        ),
        (pl.col("real_pre_n_latency_responses") + pl.col("real_post_n_latency_responses")).alias(
            "n_real_window_latency_responses"
        ),
        (
            pl.col("pseudo_pre_n_latency_responses") + pl.col("pseudo_post_n_latency_responses")
        ).alias("n_pseudo_window_latency_responses"),
    ).with_columns(
        (pl.col("real_step") - pl.col("pseudo_step")).alias("pseudo_controlled_step"),
        (pl.col("real_latency_step_seconds") - pl.col("pseudo_latency_step_seconds")).alias(
            "pseudo_controlled_latency_step_seconds"
        ),
        (pl.col("n_real_window_trials") + pl.col("n_pseudo_window_trials")).alias(
            "n_controlled_window_trials"
        ),
        (
            pl.col("n_real_window_latency_responses") + pl.col("n_pseudo_window_latency_responses")
        ).alias("n_controlled_window_latency_responses"),
    )
    effects = effects.with_columns(
        pl.col("real_step").is_not_null().alias("real_step_estimable"),
        pl.col("pseudo_step").is_not_null().alias("pseudo_step_estimable"),
        pl.col("pseudo_controlled_step").is_not_null().alias("controlled_step_estimable"),
        pl.col("real_latency_step_seconds").is_not_null().alias("real_latency_estimable"),
        pl.col("pseudo_latency_step_seconds").is_not_null().alias("pseudo_latency_estimable"),
        pl.col("pseudo_controlled_latency_step_seconds")
        .is_not_null()
        .alias("controlled_latency_estimable"),
    )
    specificity = _build_specificity_effects(
        effects,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    return pl.concat([effects, specificity], how="diagonal_relaxed").sort(
        mouse_column,
        source_column,
        "transition_order",
        "condition",
    )


def summarize_effect_window_counts(
    effect_windows: pl.DataFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Describe per-session trial support for every exploratory effect window."""

    required = (
        source_column,
        mouse_column,
        "transition_id",
        "transition_order",
        "anchor_type",
        "phase",
        "condition",
        "n_trials",
        "n_responses",
        "n_latency_responses",
        "minimum_trials_required",
        "included_in_transition_analysis",
        "window_fully_covered",
        "response_probability",
        "median_response_latency_seconds",
    )
    _require_columns(effect_windows, required, frame_name="effect_windows")
    group_columns = (
        "transition_id",
        "transition_order",
        "anchor_type",
        "phase",
        "condition",
        "minimum_trials_required",
    )
    return (
        effect_windows.group_by(*group_columns)
        .agg(
            pl.col(source_column).n_unique().cast(pl.Int64).alias("n_sessions_inventory"),
            pl.col(mouse_column).n_unique().cast(pl.Int64).alias("n_mice_inventory"),
            pl.col("included_in_transition_analysis")
            .sum()
            .cast(pl.Int64)
            .alias("n_sessions_in_analysis"),
            (
                pl.col("included_in_transition_analysis")
                & pl.col("window_fully_covered")
                & (pl.col("n_trials") >= pl.col("minimum_trials_required"))
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_sessions_meeting_minimum"),
            pl.col("response_probability")
            .is_not_null()
            .sum()
            .cast(pl.Int64)
            .alias("n_sessions_response_estimable"),
            pl.col("median_response_latency_seconds")
            .is_not_null()
            .sum()
            .cast(pl.Int64)
            .alias("n_sessions_latency_estimable"),
            pl.col("n_trials").min().cast(pl.Int64).alias("minimum_trials"),
            pl.col("n_trials").quantile(0.05, interpolation="linear").alias("p05_trials"),
            pl.col("n_trials").quantile(0.25, interpolation="linear").alias("q25_trials"),
            pl.col("n_trials").median().alias("median_trials"),
            pl.col("n_trials").quantile(0.75, interpolation="linear").alias("q75_trials"),
            pl.col("n_trials").quantile(0.95, interpolation="linear").alias("p95_trials"),
            pl.col("n_trials").max().cast(pl.Int64).alias("maximum_trials"),
            pl.col("n_responses").sum().cast(pl.Int64).alias("n_responses_total"),
            pl.col("n_latency_responses")
            .sum()
            .cast(pl.Int64)
            .alias("n_finite_response_latencies_total"),
        )
        .sort("transition_order", "anchor_type", "condition", "phase")
    )


def aggregate_mouse_effects(
    session_effects: pl.DataFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Average session transition effects equally within each mouse."""

    metric_specs = (
        ("real_step", "n_real_window_trials"),
        ("pseudo_step", "n_pseudo_window_trials"),
        ("pseudo_controlled_step", "n_controlled_window_trials"),
        ("real_latency_step_seconds", "n_real_window_latency_responses"),
        ("pseudo_latency_step_seconds", "n_pseudo_window_latency_responses"),
        (
            "pseudo_controlled_latency_step_seconds",
            "n_controlled_window_latency_responses",
        ),
    )
    aggregations: list[pl.Expr] = [
        pl.col(source_column).n_unique().cast(pl.Int64).alias("n_sessions_inventory"),
        pl.col(source_column)
        .filter(pl.col("included_in_transition_analysis"))
        .n_unique()
        .cast(pl.Int64)
        .alias("n_sessions_in_analysis"),
    ]
    for metric, count_column in metric_specs:
        aggregations.extend(
            (
                pl.col(metric).mean().alias(metric),
                pl.col(source_column)
                .filter(pl.col(metric).is_not_null())
                .n_unique()
                .cast(pl.Int64)
                .alias(f"n_sessions_{metric}"),
                pl.col(count_column)
                .filter(pl.col(metric).is_not_null())
                .sum()
                .cast(pl.Int64)
                .alias(count_column),
            )
        )
    return (
        session_effects.group_by(
            mouse_column,
            "transition_id",
            "transition_order",
            "condition",
        )
        .agg(*aggregations)
        .with_columns(pl.lit("equal_session_mean").alias("within_mouse_aggregation"))
        .sort(mouse_column, "transition_order", "condition")
    )


def summarize_mouse_trajectory(
    mouse_bins: pl.DataFrame,
    *,
    bin_kind: str,
    seed: int,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Bootstrap equal-mouse trajectory means for response and latency.

    Response intervals use mouse-level equal-session response probabilities.
    Latency intervals use each mouse's equal-session mean of session-median
    latencies and therefore condition on at least one canonical response.
    Counts report the exact number of contributing mice, sessions, trials, and
    response-latency observations separately for every bin.
    """

    if bin_kind not in {"time", "trial"}:
        raise ValueError("bin_kind must be 'time' or 'trial'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    bin_columns = (
        (
            "relative_bin_start_seconds",
            "relative_bin_stop_seconds",
            "relative_bin_center_seconds",
        )
        if bin_kind == "time"
        else ("relative_trial_start", "relative_trial_stop", "relative_trial_center")
    )
    group_columns = (
        "transition_id",
        "transition_order",
        "anchor_type",
        "condition",
        "bin_index",
        "analysis_block",
        *bin_columns,
    )
    required = (
        mouse_column,
        *group_columns,
        "response_probability",
        "mean_session_median_response_latency_seconds",
        "n_sessions_response_contributing",
        "n_trials_contributing",
        "n_responses_contributing",
        "n_sessions_latency_contributing",
        "n_latency_responses_contributing",
    )
    _require_columns(mouse_bins, required, frame_name="mouse_bins")
    rows: list[dict[str, Any]] = []
    ordered = mouse_bins.sort(*group_columns, mouse_column)
    for group_index, partition in enumerate(
        ordered.partition_by(group_columns, maintain_order=True)
    ):
        identity = partition.row(0, named=True)
        response = partition.filter(pl.col("response_probability").is_not_null()).select(
            mouse_column, "response_probability"
        )
        latency = partition.filter(
            pl.col("mean_session_median_response_latency_seconds").is_not_null()
        ).select(mouse_column, "mean_session_median_response_latency_seconds")
        response_summary = _bootstrap_or_point_summary(
            response,
            mouse_column=mouse_column,
            value_column="response_probability",
            seed=seed + 2 * group_index,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
        latency_summary = _bootstrap_or_point_summary(
            latency,
            mouse_column=mouse_column,
            value_column="mean_session_median_response_latency_seconds",
            seed=seed + 2 * group_index + 1,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                **{column: identity[column] for column in group_columns},
                "n_mice_inventory": partition.get_column(mouse_column).n_unique(),
                "n_mice_response_contributing": response.height,
                "n_sessions_response_contributing": int(
                    partition.get_column("n_sessions_response_contributing").sum()
                ),
                "n_trials_contributing": int(partition.get_column("n_trials_contributing").sum()),
                "n_responses_contributing": int(
                    partition.get_column("n_responses_contributing").sum()
                ),
                "response_probability": response_summary["estimate"],
                "response_probability_ci_low": response_summary["ci_low"],
                "response_probability_ci_high": response_summary["ci_high"],
                "response_bootstrap_seed": response_summary["seed"],
                "n_mice_latency_contributing": latency.height,
                "n_sessions_latency_contributing": int(
                    partition.get_column("n_sessions_latency_contributing").sum()
                ),
                "n_latency_responses_contributing": int(
                    partition.get_column("n_latency_responses_contributing").sum()
                ),
                "mean_session_median_response_latency_seconds": latency_summary["estimate"],
                "response_latency_ci_low": latency_summary["ci_low"],
                "response_latency_ci_high": latency_summary["ci_high"],
                "latency_bootstrap_seed": latency_summary["seed"],
                "confidence_level": confidence_level,
                "bootstrap_resamples": n_resamples,
                "ci_method": "mouse_percentile_bootstrap",
                "aggregation": "equal-session mean within mouse; equal-mouse mean",
                "latency_estimand": (
                    "mean across mice of equal-session means of session-median response latency"
                ),
            }
        )
    return pl.DataFrame(rows).sort(*group_columns)


def build_transition_statistics(
    mouse_effects: pl.DataFrame,
    *,
    dandiset_version: str,
    code_version: str,
    seed: int,
    n_bootstrap_resamples: int = 10_000,
    n_sign_flip_resamples: int = 100_000,
    confidence_level: float = 0.95,
    minimum_go_trials_per_effect_window: int = 5,
    minimum_catch_trials_per_effect_window: int = 3,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return canonical exploratory statistics for transition effects.

    Real and pseudo steps, latency effects, and catch effects are descriptive.
    The only inferential tests are two-sided mouse sign-flip tests for the
    pseudo-controlled go effect and pseudo-controlled go-minus-catch
    specificity, with Holm adjustment across the two transitions within each
    declared family.
    """

    if not dandiset_version.strip() or not code_version.strip():
        raise ValueError("dandiset_version and code_version must be non-empty")
    metric_specs = (
        {
            "metric": "real_step",
            "count": "n_real_window_trials",
            "scale": "probability difference",
            "kind": "response",
            "label": "real transition step",
        },
        {
            "metric": "pseudo_step",
            "count": "n_pseudo_window_trials",
            "scale": "probability difference",
            "kind": "response",
            "label": "within-source-state pseudo step",
        },
        {
            "metric": "pseudo_controlled_step",
            "count": "n_controlled_window_trials",
            "scale": "difference in probability differences",
            "kind": "response",
            "label": "real minus pseudo transition step",
        },
        {
            "metric": "real_latency_step_seconds",
            "count": "n_real_window_trials",
            "scale": "seconds",
            "kind": "latency",
            "label": "real transition response-latency step",
        },
        {
            "metric": "pseudo_latency_step_seconds",
            "count": "n_pseudo_window_trials",
            "scale": "seconds",
            "kind": "latency",
            "label": "within-source-state pseudo response-latency step",
        },
        {
            "metric": "pseudo_controlled_latency_step_seconds",
            "count": "n_controlled_window_trials",
            "scale": "seconds",
            "kind": "latency",
            "label": "real minus pseudo response-latency step",
        },
    )
    required = [mouse_column, "transition_id", "condition"]
    for spec in metric_specs:
        required.extend(
            (
                spec["metric"],
                f"n_sessions_{spec['metric']}",
                spec["count"],
            )
        )
    _require_columns(mouse_effects, tuple(required), frame_name="mouse_effects")

    rows: list[dict[str, Any]] = []
    row_index = 0
    for transition_id in TRANSITION_IDS:
        for condition in EFFECT_CONDITIONS:
            for spec in metric_specs:
                if condition == "go_minus_catch" and spec["kind"] == "latency":
                    continue
                row_seed = seed + row_index
                row_index += 1
                subset = mouse_effects.filter(
                    (pl.col("transition_id") == transition_id)
                    & (pl.col("condition") == condition)
                    & pl.col(spec["metric"]).is_not_null()
                )
                inferential = spec["metric"] == "pseudo_controlled_step" and condition in {
                    "go",
                    "go_minus_catch",
                }
                row = _transition_statistic_row(
                    subset,
                    transition_id=transition_id,
                    condition=condition,
                    metric=spec["metric"],
                    count_column=spec["count"],
                    scale=spec["scale"],
                    metric_kind=spec["kind"],
                    metric_label=spec["label"],
                    inferential=inferential,
                    dandiset_version=dandiset_version,
                    code_version=code_version,
                    seed=row_seed,
                    n_bootstrap_resamples=n_bootstrap_resamples,
                    n_sign_flip_resamples=n_sign_flip_resamples,
                    confidence_level=confidence_level,
                    minimum_trials_description=(
                        f">={minimum_go_trials_per_effect_window} go and "
                        f">={minimum_catch_trials_per_effect_window} catch trials"
                        if condition == "go_minus_catch"
                        else (
                            f">={minimum_go_trials_per_effect_window} go trials"
                            if condition == "go"
                            else f">={minimum_catch_trials_per_effect_window} catch trials"
                        )
                    ),
                    mouse_column=mouse_column,
                )
                rows.append(row)
    normalized = dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    return dg.statistics.add_holm_adjustment(normalized)


def build_effect_statistic_sample_sizes(
    mouse_effects: pl.DataFrame,
    *,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Report eligible-trial and responder counts for every effect statistic.

    The canonical statistics schema has one ``n_trials`` field. This companion
    table prevents responder-only latency observations from being conflated
    with total eligible trials by reporting both counts explicitly.
    """

    metric_specs = (
        ("real_step", "n_real_window_trials", "n_real_window_latency_responses", "response"),
        (
            "pseudo_step",
            "n_pseudo_window_trials",
            "n_pseudo_window_latency_responses",
            "response",
        ),
        (
            "pseudo_controlled_step",
            "n_controlled_window_trials",
            "n_controlled_window_latency_responses",
            "response",
        ),
        (
            "real_latency_step_seconds",
            "n_real_window_trials",
            "n_real_window_latency_responses",
            "responder_only_latency",
        ),
        (
            "pseudo_latency_step_seconds",
            "n_pseudo_window_trials",
            "n_pseudo_window_latency_responses",
            "responder_only_latency",
        ),
        (
            "pseudo_controlled_latency_step_seconds",
            "n_controlled_window_trials",
            "n_controlled_window_latency_responses",
            "responder_only_latency",
        ),
    )
    rows: list[dict[str, Any]] = []
    for transition_id in TRANSITION_IDS:
        for condition in EFFECT_CONDITIONS:
            for metric, trial_count, response_count, estimand in metric_specs:
                if condition == "go_minus_catch" and estimand == "responder_only_latency":
                    continue
                subset = mouse_effects.filter(
                    (pl.col("transition_id") == transition_id)
                    & (pl.col("condition") == condition)
                    & pl.col(metric).is_not_null()
                )
                rows.append(
                    {
                        "result_id": f"{transition_id}_{condition}_{metric}_mouse_mean",
                        "transition_id": transition_id,
                        "condition": condition,
                        "metric": metric,
                        "estimand": estimand,
                        "n_mice": subset.height,
                        "n_sessions": (
                            int(subset.get_column(f"n_sessions_{metric}").sum())
                            if subset.height
                            else 0
                        ),
                        "n_eligible_trials": (
                            int(subset.get_column(trial_count).sum()) if subset.height else 0
                        ),
                        "n_finite_canonical_response_latencies": (
                            int(subset.get_column(response_count).sum()) if subset.height else 0
                        ),
                        "inference_role": (
                            "inferential"
                            if metric == "pseudo_controlled_step"
                            and condition in {"go", "go_minus_catch"}
                            else "descriptive"
                        ),
                    }
                )
    return pl.DataFrame(rows)


def _bootstrap_or_point_summary(
    values: pl.DataFrame,
    *,
    mouse_column: str,
    value_column: str,
    seed: int,
    n_resamples: int,
    confidence_level: float,
) -> dict[str, float | int | None]:
    if values.is_empty():
        return {"estimate": None, "ci_low": None, "ci_high": None, "seed": None}
    if values.height == 1:
        return {
            "estimate": float(values.get_column(value_column).item()),
            "ci_low": None,
            "ci_high": None,
            "seed": None,
        }
    result = dg.statistics.bootstrap_mouse_mean(
        values,
        seed=seed,
        mouse_column=mouse_column,
        value_column=value_column,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
    return {
        "estimate": result.estimate,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "seed": result.seed,
    }


def _transition_statistic_row(
    subset: pl.DataFrame,
    *,
    transition_id: str,
    condition: str,
    metric: str,
    count_column: str,
    scale: str,
    metric_kind: str,
    metric_label: str,
    inferential: bool,
    dandiset_version: str,
    code_version: str,
    seed: int,
    n_bootstrap_resamples: int,
    n_sign_flip_resamples: int,
    confidence_level: float,
    minimum_trials_description: str,
    mouse_column: str,
) -> dict[str, Any]:
    summary = _bootstrap_or_point_summary(
        subset.select(mouse_column, metric),
        mouse_column=mouse_column,
        value_column=metric,
        seed=seed,
        n_resamples=n_bootstrap_resamples,
        confidence_level=confidence_level,
    )
    sign_flip = None
    if inferential and subset.height >= 2:
        sign_flip = dg.statistics.two_sided_sign_flip_test(
            subset.select(mouse_column, metric),
            seed=seed,
            mouse_column=mouse_column,
            value_column=metric,
            n_resamples=n_sign_flip_resamples,
        )
    transition_label = (
        "E1-to-no-reward withdrawal"
        if transition_id == "reward_withdrawal"
        else "no-reward-to-E2 restoration"
    )
    if metric_kind == "latency":
        condition_description = (
            f"{condition} responder-only latency; positive means slowing after withdrawal "
            "or speeding after restoration"
        )
        inclusion_definition = (
            "technically valid sessions with at least one canonical response in every required "
            f"3-minute window and {minimum_trials_description} per window; "
            "responder-only conditional latency summary"
        )
        missingness_stratum = "responders_in_all_required_effect_windows"
        hypothesis = (
            f"Describe {transition_label} {condition_description} for {metric_label}; "
            "no unconditional or inferential latency claim."
        )
    else:
        condition_description = condition.replace("_", "-")
        inclusion_definition = (
            "technically valid sessions with "
            f"{minimum_trials_description} in every required 3-minute effect window"
        )
        missingness_stratum = "complete_required_effect_windows"
        hypothesis = f"{transition_label} {metric_label} for {condition_description} responses."
    if inferential:
        hypothesis = (
            f"{transition_label} pseudo-controlled {condition_description} response gating "
            "differs from zero."
        )
    status = "pass" if subset.height >= 2 else "not_estimable"
    reason = None if status == "pass" else "fewer than two mice had a complete effect estimate"
    sessions_column = f"n_sessions_{metric}"
    n_sessions = int(subset.get_column(sessions_column).sum()) if subset.height else 0
    n_observations = int(subset.get_column(count_column).sum()) if subset.height else 0
    model_formula = _effect_formula(metric)
    if metric_kind == "latency":
        model_formula += "; conditional on canonical response in each window"
    family = None
    if inferential:
        family = (
            "behavior_transition_primary_go"
            if condition == "go"
            else "behavior_transition_primary_specificity"
        )
    return {
        "analysis_id": "behavior_transition_qc",
        "result_id": f"{transition_id}_{condition}_{metric}_mouse_mean",
        "contrast_id": f"{transition_id}_{condition}_{metric}",
        "hypothesis": hypothesis,
        "dandiset_version": dandiset_version,
        "code_version": code_version,
        "seed": seed if subset.height >= 2 else None,
        "analysis_tier": "exploratory",
        "inclusion_definition": inclusion_definition,
        "missingness_stratum": missingness_stratum,
        "estimate": summary["estimate"],
        "scale": scale,
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "confidence_level": confidence_level if subset.height >= 2 else None,
        "ci_method": (
            f"mouse percentile bootstrap ({n_bootstrap_resamples} resamples)"
            if subset.height >= 2
            else None
        ),
        "test_statistic": sign_flip.statistic if sign_flip is not None else None,
        "test_method": sign_flip.method if sign_flip is not None else None,
        "p_value": sign_flip.p_value if sign_flip is not None else None,
        "adjusted_p_value": None,
        "adjustment_method": None,
        "multiplicity_family": family if sign_flip is not None else None,
        "sidedness": "two-sided" if sign_flip is not None else "not_applicable",
        "n_mice": subset.height,
        "n_sessions": n_sessions,
        "n_trials": n_observations,
        "aggregation": "equal-session mean within mouse; equal-mouse inference",
        "bootstrap_id": (
            f"seed={seed};n_resamples={n_bootstrap_resamples}" if subset.height >= 2 else None
        ),
        "model_formula": model_formula,
        "cv_grouping": None,
        "status": status,
        "reason": reason,
    }


def _effect_formula(metric: str) -> str:
    if metric == "real_step":
        return "transition-specific signed(post - pre) at real boundary"
    if metric == "pseudo_step":
        return "transition-specific signed(post - pre) at source-state temporal midpoint"
    if metric == "pseudo_controlled_step":
        return "real signed(post - pre) - matched source-state pseudo signed(post - pre)"
    if metric == "real_latency_step_seconds":
        return "latency-specific signed(post - pre) at real boundary"
    if metric == "pseudo_latency_step_seconds":
        return "latency-specific signed(post - pre) at source-state temporal midpoint"
    if metric == "pseudo_controlled_latency_step_seconds":
        return "real latency signed(post - pre) - matched source-state pseudo signed(post - pre)"
    raise ValueError(f"unknown transition metric: {metric!r}")


def build_d03_qc_tables(
    session_qc: pl.DataFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build criterion pass counts and raw-metric distributions for D03."""

    threshold_specs = _d03_threshold_specs()
    required = [source_column, mouse_column, "is_technically_valid", "is_good_session"]
    required.extend(spec["metric_column"] for spec in threshold_specs)
    required.extend(spec["threshold_column"] for spec in threshold_specs)
    required.extend(spec["pass_column"] for spec in _d03_criterion_specs())
    _require_columns(session_qc, tuple(dict.fromkeys(required)), frame_name="session_qc")

    distributions: list[dict[str, Any]] = []
    for spec in threshold_specs:
        metric = session_qc.get_column(spec["metric_column"]).cast(pl.Float64, strict=True)
        threshold_values = (
            session_qc.get_column(spec["threshold_column"]).drop_nulls().unique().to_list()
        )
        if len(threshold_values) != 1:
            raise ValueError(f"{spec['threshold_column']} must contain one non-null value")
        threshold = float(threshold_values[0])
        finite = metric.filter(metric.is_not_null() & metric.is_finite())
        meeting = _threshold_pass_series(metric, threshold=threshold, operator=spec["operator"])
        evaluable_mouse_count = (
            session_qc.filter(
                pl.col(spec["metric_column"]).is_not_null()
                & pl.col(spec["metric_column"]).is_finite()
            )
            .get_column(mouse_column)
            .n_unique()
        )
        distributions.append(
            {
                "metric_id": spec["metric_id"],
                "criterion_group": spec["criterion_group"],
                "metric_column": spec["metric_column"],
                "metric_label": spec["metric_label"],
                "threshold_operator": spec["operator"],
                "threshold_value": threshold,
                "n_sessions_total": session_qc.height,
                "n_sessions_evaluable": finite.len(),
                "n_sessions_missing": session_qc.height - finite.len(),
                "n_sessions_meeting_threshold": int(meeting.sum()),
                "n_mice_total": session_qc.get_column(mouse_column).n_unique(),
                "n_mice_evaluable": evaluable_mouse_count,
                "minimum": _series_stat(finite, "min"),
                "p05": _series_quantile(finite, 0.05),
                "q25": _series_quantile(finite, 0.25),
                "median": _series_stat(finite, "median"),
                "q75": _series_quantile(finite, 0.75),
                "p95": _series_quantile(finite, 0.95),
                "maximum": _series_stat(finite, "max"),
            }
        )

    criterion_rows: list[dict[str, Any]] = []
    for order, spec in enumerate(_d03_criterion_specs(), start=1):
        flag = session_qc.get_column(spec["pass_column"])
        evaluable = flag.is_not_null()
        passed = flag.fill_null(False)
        session_pass = session_qc.with_columns(passed.alias("_criterion_pass"))
        mouse_summary = session_pass.group_by(mouse_column).agg(
            pl.col("_criterion_pass").any().alias("any_session_pass"),
            pl.col("_criterion_pass").all().alias("all_sessions_pass"),
        )
        criterion_rows.append(
            {
                "criterion_order": order,
                "criterion_id": spec["criterion_id"],
                "criterion_label": spec["criterion_label"],
                "pass_column": spec["pass_column"],
                "approved_definition": spec["approved_definition"],
                "n_sessions_total": session_qc.height,
                "n_sessions_evaluable": int(evaluable.sum()),
                "n_sessions_pass": int(passed.sum()),
                "n_sessions_fail": int((evaluable & ~passed).sum()),
                "session_pass_fraction": (
                    float(passed.sum()) / int(evaluable.sum()) if int(evaluable.sum()) else None
                ),
                "n_mice_total": session_qc.get_column(mouse_column).n_unique(),
                "n_mice_with_any_session_pass": int(
                    mouse_summary.get_column("any_session_pass").sum()
                ),
                "n_mice_with_all_sessions_pass": int(
                    mouse_summary.get_column("all_sessions_pass").sum()
                ),
            }
        )
    criterion_pass_columns = tuple(spec["pass_column"] for spec in _d03_criterion_specs())
    overall = session_qc.with_columns(
        pl.all_horizontal(
            *(pl.col(column).fill_null(False) for column in criterion_pass_columns)
        ).alias("_criterion_pass")
    )
    overall_mice = overall.group_by(mouse_column).agg(
        pl.col("_criterion_pass").any().alias("any_session_pass"),
        pl.col("_criterion_pass").all().alias("all_sessions_pass"),
    )
    criterion_rows.append(
        {
            "criterion_order": len(criterion_rows) + 1,
            "criterion_id": "all_approved_d03_thresholds",
            "criterion_label": "All approved D03 behavior thresholds",
            "pass_column": "conjunction_of_eight_d03_pass_flags",
            "approved_definition": "all eight component D03 pass flags are true",
            "n_sessions_total": session_qc.height,
            "n_sessions_evaluable": session_qc.height,
            "n_sessions_pass": int(overall.get_column("_criterion_pass").sum()),
            "n_sessions_fail": int((~overall.get_column("_criterion_pass")).sum()),
            "session_pass_fraction": float(overall.get_column("_criterion_pass").mean()),
            "n_mice_total": session_qc.get_column(mouse_column).n_unique(),
            "n_mice_with_any_session_pass": int(overall_mice.get_column("any_session_pass").sum()),
            "n_mice_with_all_sessions_pass": int(
                overall_mice.get_column("all_sessions_pass").sum()
            ),
        }
    )
    return pl.DataFrame(criterion_rows), pl.DataFrame(distributions)


def _prepare_trials(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    required = (
        source_column,
        mouse_column,
        "_table_index",
        "start_time",
        "stop_time",
        "change_time",
        "reward_block",
        "go",
        "catch",
        "aborted",
        "auto_rewarded",
        "response_in_window",
        "response_latency_from_licks",
    )
    _require_columns(trials, required, frame_name="trials")
    return _collect_projection(trials, required)


def _prepare_session_qc(
    session_qc: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    frame = session_qc.collect() if isinstance(session_qc, pl.LazyFrame) else session_qc
    _require_columns(
        frame,
        (source_column, mouse_column, "is_technically_valid", "is_good_session"),
        frame_name="session_qc",
    )
    _validate_unique_non_null(frame, source_column, frame_name="session_qc")
    if frame.get_column(mouse_column).null_count():
        raise ValueError(f"session_qc.{mouse_column} must not contain nulls")
    return frame


def _validate_trial_session_alignment(
    trials: pl.DataFrame,
    session_qc: pl.DataFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> None:
    trial_sessions = trials.select(source_column, mouse_column).unique()
    _validate_unique_non_null(trial_sessions, source_column, frame_name="trial session mapping")
    comparison = session_qc.select(source_column, mouse_column).join(
        trial_sessions,
        on=source_column,
        how="full",
        coalesce=True,
        suffix="_trial",
        validate="1:1",
    )
    mismatch = comparison.filter(
        pl.col(mouse_column).is_null()
        | pl.col(f"{mouse_column}_trial").is_null()
        | (pl.col(mouse_column) != pl.col(f"{mouse_column}_trial"))
    )
    if mismatch.height:
        raise ValueError("trials and session_qc have different session/mouse coverage")


def _eligible_event_trials(
    trials: pl.DataFrame,
    session_qc: pl.DataFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    candidate = (
        (pl.col("go").fill_null(False) | pl.col("catch").fill_null(False))
        & ~pl.col("aborted").fill_null(False)
        & ~pl.col("auto_rewarded").fill_null(False)
    )
    invalid_labels = trials.filter(
        candidate
        & (
            pl.col("go").is_null()
            | pl.col("catch").is_null()
            | (pl.col("go") == pl.col("catch"))
            | pl.col("response_in_window").is_null()
            | pl.col("change_time").is_null()
            | ~pl.col("change_time").is_finite()
        )
    )
    technical_sources = (
        session_qc.filter(pl.col("is_technically_valid")).get_column(source_column).to_list()
    )
    if invalid_labels.filter(pl.col(source_column).is_in(technical_sources)).height:
        raise ValueError("technically valid sessions contain invalid eligible event trials")

    eligible = trials.filter(
        candidate
        & pl.col("go").is_not_null()
        & pl.col("catch").is_not_null()
        & (pl.col("go") != pl.col("catch"))
        & pl.col("response_in_window").is_not_null()
        & pl.col("change_time").is_not_null()
        & pl.col("change_time").is_finite()
    ).with_columns(
        pl.when(pl.col("go")).then(pl.lit("go")).otherwise(pl.lit("catch")).alias("condition"),
        pl.when(
            pl.col("response_in_window")
            & pl.col("response_latency_from_licks").is_not_null()
            & pl.col("response_latency_from_licks").is_finite()
            & (
                pl.col("response_latency_from_licks")
                > dg.quality.DEFAULT_RESPONSE_WINDOW_START_SECONDS
            )
            & (
                pl.col("response_latency_from_licks")
                <= dg.quality.DEFAULT_RESPONSE_WINDOW_STOP_SECONDS
            )
        )
        .then(pl.col("response_latency_from_licks"))
        .otherwise(None)
        .alias("response_latency_seconds"),
    )
    invalid_response_latency = eligible.filter(
        pl.col("response_in_window") & pl.col("response_latency_seconds").is_null()
    )
    if invalid_response_latency.filter(pl.col(source_column).is_in(technical_sources)).height:
        raise ValueError("technically valid responding trials contain invalid response latency")
    return eligible.select(
        source_column,
        mouse_column,
        "_table_index",
        "change_time",
        "reward_block",
        "condition",
        "response_in_window",
        "response_latency_seconds",
    )


def _anchor_table(
    boundaries: pl.DataFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    metadata = [
        source_column,
        mouse_column,
        *(
            column
            for column in ("ecephys_session_id", "recording_day", "session_number")
            if column in boundaries.columns
        ),
        "transition_id",
        "transition_order",
        "pre_block",
        "post_block",
        "pseudo_block",
        "included_in_transition_analysis",
        "effect_direction_multiplier",
        "latency_direction_multiplier",
    ]
    real = boundaries.select(
        *metadata,
        pl.lit("real").alias("anchor_type"),
        pl.col("real_boundary_time").alias("anchor_time"),
        pl.col("pre_block").alias("anchor_pre_block"),
        pl.col("post_block").alias("anchor_post_block"),
        pl.col("real_pre_coverage_start").alias("anchor_coverage_start"),
        pl.col("real_post_coverage_stop").alias("anchor_coverage_stop"),
    )
    pseudo = boundaries.select(
        *metadata,
        pl.lit("pseudo").alias("anchor_type"),
        pl.col("pseudo_boundary_time").alias("anchor_time"),
        pl.col("pseudo_block").alias("anchor_pre_block"),
        pl.col("pseudo_block").alias("anchor_post_block"),
        pl.col("pseudo_coverage_start").alias("anchor_coverage_start"),
        pl.col("pseudo_coverage_stop").alias("anchor_coverage_stop"),
    )
    return pl.concat([real, pseudo], how="vertical").sort(
        mouse_column,
        source_column,
        "transition_order",
        "anchor_type",
    )


def _trial_summary_expressions() -> tuple[pl.Expr, ...]:
    finite_latency_response = (
        pl.col("response_in_window")
        & pl.col("response_latency_seconds").is_not_null()
        & pl.col("response_latency_seconds").is_finite()
    )
    latency = pl.col("response_latency_seconds").filter(finite_latency_response)
    return (
        pl.len().cast(pl.Int64).alias("n_trials"),
        pl.col("response_in_window").sum().cast(pl.Int64).alias("n_responses"),
        latency.count().cast(pl.Int64).alias("n_latency_responses"),
        latency.median().alias("_median_response_latency_seconds"),
    )


def _finish_session_bins(
    table: pl.DataFrame,
    *,
    bin_kind: str,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    table = table.with_columns(
        pl.col("n_trials").fill_null(0),
        pl.col("n_responses").fill_null(0),
        pl.col("n_latency_responses").fill_null(0),
    )
    estimate_allowed = (
        pl.col("included_in_transition_analysis")
        & pl.col("bin_fully_covered")
        & (pl.col("n_trials") > 0)
    )
    result = table.with_columns(
        pl.when(estimate_allowed)
        .then(pl.col("n_responses") / pl.col("n_trials"))
        .otherwise(None)
        .alias("response_probability"),
        pl.when(estimate_allowed & (pl.col("n_latency_responses") > 0))
        .then(pl.col("_median_response_latency_seconds"))
        .otherwise(None)
        .alias("median_response_latency_seconds"),
        pl.when(~pl.col("included_in_transition_analysis"))
        .then(pl.lit("excluded_session"))
        .when(~pl.col("bin_fully_covered"))
        .then(
            pl.lit(
                "outside_observed_block_coverage"
                if bin_kind == "time"
                else "incomplete_event_trial_bin"
            )
        )
        .when(pl.col("n_trials") == 0)
        .then(pl.lit("no_condition_trials"))
        .otherwise(pl.lit("observed"))
        .alias("bin_status"),
    ).drop("_median_response_latency_seconds")
    return result.sort(
        mouse_column,
        source_column,
        "transition_order",
        "anchor_type",
        "condition",
        "bin_index",
    )


def _build_specificity_effects(
    effects: pl.DataFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> pl.DataFrame:
    key_columns = (
        source_column,
        mouse_column,
        "transition_id",
        "transition_order",
        "effect_direction_multiplier",
        "latency_direction_multiplier",
        "included_in_transition_analysis",
    )
    metric_columns = (
        "real_step",
        "pseudo_step",
        "pseudo_controlled_step",
    )
    count_columns = (
        "n_real_window_trials",
        "n_pseudo_window_trials",
        "n_controlled_window_trials",
        "n_real_window_latency_responses",
        "n_pseudo_window_latency_responses",
        "n_controlled_window_latency_responses",
    )
    go = effects.filter(pl.col("condition") == "go").select(
        *key_columns,
        *(pl.col(column).alias(f"go_{column}") for column in (*metric_columns, *count_columns)),
    )
    catch = effects.filter(pl.col("condition") == "catch").select(
        source_column,
        "transition_id",
        *(pl.col(column).alias(f"catch_{column}") for column in (*metric_columns, *count_columns)),
    )
    result = (
        go.join(
            catch,
            on=(source_column, "transition_id"),
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.lit("go_minus_catch").alias("condition"),
            *(
                (pl.col(f"go_{column}") - pl.col(f"catch_{column}")).alias(column)
                for column in metric_columns
            ),
            *(
                (pl.col(f"go_{column}") + pl.col(f"catch_{column}")).alias(column)
                for column in count_columns
            ),
            pl.lit(None, dtype=pl.Float64).alias("real_latency_step_seconds"),
            pl.lit(None, dtype=pl.Float64).alias("pseudo_latency_step_seconds"),
            pl.lit(None, dtype=pl.Float64).alias("pseudo_controlled_latency_step_seconds"),
        )
        .with_columns(
            pl.col("real_step").is_not_null().alias("real_step_estimable"),
            pl.col("pseudo_step").is_not_null().alias("pseudo_step_estimable"),
            pl.col("pseudo_controlled_step").is_not_null().alias("controlled_step_estimable"),
            pl.lit(False).alias("real_latency_estimable"),
            pl.lit(False).alias("pseudo_latency_estimable"),
            pl.lit(False).alias("controlled_latency_estimable"),
        )
    )
    return result.select(*(column for column in effects.columns if column in result.columns))


def _d03_criterion_specs() -> tuple[dict[str, str], ...]:
    return (
        {
            "criterion_id": "minimum_go_trials_all_blocks",
            "criterion_label": "At least 10 go trials in each analysis block",
            "pass_column": "minimum_go_trials_pass",
            "approved_definition": "E1, late NR, and early E2 each have >=10 eligible go trials",
        },
        {
            "criterion_id": "minimum_catch_trials_engaged_blocks",
            "criterion_label": "At least 5 catch trials in each engaged block",
            "pass_column": "minimum_engaged_catch_trials_pass",
            "approved_definition": "E1 and early E2 each have >=5 eligible catch trials",
        },
        {
            "criterion_id": "engaged_1_response",
            "criterion_label": "E1 go response probability >=0.5",
            "pass_column": "engaged_1_response_pass",
            "approved_definition": "E1 canonical lick response probability >=0.5",
        },
        {
            "criterion_id": "engaged_2_response",
            "criterion_label": "Early-E2 go response probability >=0.5",
            "pass_column": "engaged_2_response_pass",
            "approved_definition": "Early E2 canonical lick response probability >=0.5",
        },
        {
            "criterion_id": "engaged_1_dprime",
            "criterion_label": "E1 d-prime >=1.0",
            "pass_column": "engaged_1_dprime_pass",
            "approved_definition": "E1 loglinear d-prime >=1.0",
        },
        {
            "criterion_id": "engaged_2_dprime",
            "criterion_label": "Early-E2 d-prime >=1.0",
            "pass_column": "engaged_2_dprime_pass",
            "approved_definition": "Early E2 loglinear d-prime >=1.0",
        },
        {
            "criterion_id": "late_no_reward_response",
            "criterion_label": "Late-NR go response probability <=0.2",
            "pass_column": "no_reward_response_pass",
            "approved_definition": "Final 600 s of NR canonical lick response probability <=0.2",
        },
        {
            "criterion_id": "reward_suppression_drop",
            "criterion_label": "Reward suppression drop >=0.3",
            "pass_column": "reward_suppression_drop_pass",
            "approved_definition": "min(E1, early E2) response probability - late NR >=0.3",
        },
    )


def _d03_threshold_specs() -> tuple[dict[str, str], ...]:
    count_threshold_go = "minimum_go_trials_threshold"
    count_threshold_catch = "minimum_engaged_catch_trials_threshold"
    response_threshold = "minimum_engaged_response_rate_threshold"
    dprime_threshold = "minimum_engaged_dprime_threshold"
    return (
        _metric_spec(
            "engaged_1_go_trials",
            "minimum_go_trials_all_blocks",
            "n_engaged_1_go_trials",
            "E1 eligible go trials",
            ">=",
            count_threshold_go,
        ),
        _metric_spec(
            "late_no_reward_go_trials",
            "minimum_go_trials_all_blocks",
            "n_no_reward_late_go_trials",
            "Late-NR eligible go trials",
            ">=",
            count_threshold_go,
        ),
        _metric_spec(
            "early_engaged_2_go_trials",
            "minimum_go_trials_all_blocks",
            "n_engaged_2_early_go_trials",
            "Early-E2 eligible go trials",
            ">=",
            count_threshold_go,
        ),
        _metric_spec(
            "engaged_1_catch_trials",
            "minimum_catch_trials_engaged_blocks",
            "n_engaged_1_catch_trials",
            "E1 eligible catch trials",
            ">=",
            count_threshold_catch,
        ),
        _metric_spec(
            "early_engaged_2_catch_trials",
            "minimum_catch_trials_engaged_blocks",
            "n_engaged_2_early_catch_trials",
            "Early-E2 eligible catch trials",
            ">=",
            count_threshold_catch,
        ),
        _metric_spec(
            "engaged_1_response_probability",
            "engaged_1_response",
            "engaged_1_response_probability",
            "E1 go response probability",
            ">=",
            response_threshold,
        ),
        _metric_spec(
            "early_engaged_2_response_probability",
            "engaged_2_response",
            "engaged_2_early_response_probability",
            "Early-E2 go response probability",
            ">=",
            response_threshold,
        ),
        _metric_spec(
            "engaged_1_dprime",
            "engaged_1_dprime",
            "engaged_1_dprime",
            "E1 loglinear d-prime",
            ">=",
            dprime_threshold,
        ),
        _metric_spec(
            "early_engaged_2_dprime",
            "engaged_2_dprime",
            "engaged_2_early_dprime",
            "Early-E2 loglinear d-prime",
            ">=",
            dprime_threshold,
        ),
        _metric_spec(
            "late_no_reward_response_probability",
            "late_no_reward_response",
            "no_reward_late_response_probability",
            "Late-NR go response probability",
            "<=",
            "maximum_no_reward_response_rate_threshold",
        ),
        _metric_spec(
            "reward_suppression_drop",
            "reward_suppression_drop",
            "reward_suppression_drop",
            "Minimum engaged response minus late-NR response",
            ">=",
            "minimum_reward_suppression_drop_threshold",
        ),
    )


def _metric_spec(
    metric_id: str,
    criterion_group: str,
    metric_column: str,
    metric_label: str,
    operator: str,
    threshold_column: str,
) -> dict[str, str]:
    return {
        "metric_id": metric_id,
        "criterion_group": criterion_group,
        "metric_column": metric_column,
        "metric_label": metric_label,
        "operator": operator,
        "threshold_column": threshold_column,
    }


def _threshold_pass_series(
    metric: pl.Series,
    *,
    threshold: float,
    operator: str,
) -> pl.Series:
    valid = metric.is_not_null() & metric.is_finite()
    if operator == ">=":
        return valid & (metric >= threshold)
    if operator == "<=":
        return valid & (metric <= threshold)
    raise ValueError(f"unsupported threshold operator: {operator!r}")


def _series_stat(series: pl.Series, method: str) -> float | None:
    if series.is_empty():
        return None
    value = getattr(series, method)()
    return float(value) if value is not None else None


def _series_quantile(series: pl.Series, probability: float) -> float | None:
    if series.is_empty():
        return None
    value = series.quantile(probability, interpolation="linear")
    return float(value) if value is not None else None


def _collect_projection(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    selected = frame.select(*columns)
    return selected.collect() if isinstance(selected, pl.LazyFrame) else selected


def _require_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    names = (
        set(frame.collect_schema().names())
        if isinstance(frame, pl.LazyFrame)
        else set(frame.columns)
    )
    missing = set(columns).difference(names)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def _validate_unique_non_null(
    frame: pl.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> None:
    if frame.get_column(column).null_count():
        raise ValueError(f"{frame_name}.{column} must not contain nulls")
    if frame.get_column(column).n_unique() != frame.height:
        raise ValueError(f"{frame_name}.{column} must be unique")
