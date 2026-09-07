"""Behavioral-session and unit-quality filters for Dynamic Gating analyses."""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Any, TypeVar

import numpy as np
import polars as pl

SOURCE_COLUMN = "_nwb_path"
REWARD_BLOCK_COLUMN = "reward_block"
COARSE_UNIT_STABILITY_COLUMN = "coarse_engaged_rate_consistent"

FrameT = TypeVar("FrameT", pl.DataFrame, pl.LazyFrame)


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclasses.dataclass(frozen=True, slots=True)
class UnitQualityThresholds:
    """Isolation thresholds applied directly to the NWB units table."""

    maximum_isi_violations: float = 0.5
    maximum_amplitude_cutoff: float = 0.1

    def __post_init__(self) -> None:
        _validate_positive("maximum_isi_violations", self.maximum_isi_violations)
        _validate_positive("maximum_amplitude_cutoff", self.maximum_amplitude_cutoff)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionQualityThresholds:
    """Operational definition of a behaviorally gated session.

    Defaults require at least ten evaluable go trials in every block, at least
    five catch trials and d-prime of one in each engaged block, at least 50%
    hit rate in each engaged block, and at most 20% during no reward. These
    thresholds are explicit so they can be reviewed before confirmation.
    """

    minimum_go_trials_per_block: int = 10
    minimum_catch_trials_per_engaged_block: int = 5
    minimum_engaged_response_rate: float = 0.5
    minimum_engaged_dprime: float = 1.0
    maximum_no_reward_response_rate: float = 0.2
    minimum_reward_suppression_drop: float = 0.3

    def __post_init__(self) -> None:
        if self.minimum_go_trials_per_block < 1:
            raise ValueError("minimum_go_trials_per_block must be positive")
        if self.minimum_catch_trials_per_engaged_block < 1:
            raise ValueError("minimum_catch_trials_per_engaged_block must be positive")
        _validate_probability("minimum_engaged_response_rate", self.minimum_engaged_response_rate)
        _validate_nonnegative_finite("minimum_engaged_dprime", self.minimum_engaged_dprime)
        _validate_probability(
            "maximum_no_reward_response_rate", self.maximum_no_reward_response_rate
        )
        _validate_probability(
            "minimum_reward_suppression_drop", self.minimum_reward_suppression_drop
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CoarseUnitStabilityThresholds:
    """Thresholds for a coarse whole-block firing-rate consistency screen."""

    minimum_spikes_per_block: int = 20
    minimum_engaged_rate_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.minimum_spikes_per_block < 1:
            raise ValueError("minimum_spikes_per_block must be positive")
        if not 0 < self.minimum_engaged_rate_ratio <= 1:
            raise ValueError("minimum_engaged_rate_ratio must be in (0, 1]")


DEFAULT_UNIT_QUALITY_THRESHOLDS = UnitQualityThresholds()
DEFAULT_SESSION_QUALITY_THRESHOLDS = SessionQualityThresholds()
DEFAULT_COARSE_UNIT_STABILITY_THRESHOLDS = CoarseUnitStabilityThresholds()


def well_isolated_unit_expr(
    thresholds: UnitQualityThresholds = DEFAULT_UNIT_QUALITY_THRESHOLDS,
) -> pl.Expr:
    """Build the strict isolation-quality predicate specified for this project."""

    isi = pl.col("isi_violations")
    amplitude = pl.col("amplitude_cutoff")
    return (
        isi.is_not_null()
        & isi.is_finite()
        & (isi >= 0)
        & (isi < thresholds.maximum_isi_violations)
        & amplitude.is_not_null()
        & amplitude.is_finite()
        & (amplitude >= 0)
        & (amplitude < thresholds.maximum_amplitude_cutoff)
    )


def filter_well_isolated_units(
    units: FrameT,
    *,
    thresholds: UnitQualityThresholds = DEFAULT_UNIT_QUALITY_THRESHOLDS,
) -> FrameT:
    """Keep units below both strict isolation-metric thresholds."""

    return units.filter(well_isolated_unit_expr(thresholds))


def add_well_isolated_unit_flag(
    units: FrameT,
    *,
    thresholds: UnitQualityThresholds = DEFAULT_UNIT_QUALITY_THRESHOLDS,
) -> FrameT:
    """Add auditable isolation pass and exclusion-reason columns."""

    isi = pl.col("isi_violations")
    amplitude = pl.col("amplitude_cutoff")
    return units.with_columns(
        well_isolated_unit_expr(thresholds).alias("well_isolated"),
        pl.concat_str(
            pl.when(isi.is_null() | ~isi.is_finite() | (isi < 0))
            .then(pl.lit("invalid_isi_violations;"))
            .otherwise(pl.lit("")),
            pl.when(
                isi.is_not_null() & isi.is_finite() & (isi >= thresholds.maximum_isi_violations)
            )
            .then(pl.lit("isi_violations_above_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(amplitude.is_null() | ~amplitude.is_finite() | (amplitude < 0))
            .then(pl.lit("invalid_amplitude_cutoff;"))
            .otherwise(pl.lit("")),
            pl.when(
                amplitude.is_not_null()
                & amplitude.is_finite()
                & (amplitude >= thresholds.maximum_amplitude_cutoff)
            )
            .then(pl.lit("amplitude_cutoff_above_threshold;"))
            .otherwise(pl.lit("")),
        )
        .str.strip_chars_end(";")
        .alias("unit_quality_exclusion_reasons"),
    )


def label_reward_blocks(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
) -> pl.LazyFrame:
    """Label trials as ``engaged_1``, ``no_reward``, or ``engaged_2``.

    Block boundaries are derived independently for each source NWB. Trials that
    overlap a boundary, or sessions without a no-reward epoch, are labeled
    ``transition`` and are not used by the quality summaries.
    """

    frame = trials.lazy() if isinstance(trials, pl.DataFrame) else trials
    no_reward = pl.col("no_reward_epoch").fill_null(False)
    return (
        frame.with_columns(
            pl.col("start_time")
            .filter(no_reward)
            .min()
            .over(source_column)
            .alias("_no_reward_start"),
            pl.col("stop_time")
            .filter(no_reward)
            .max()
            .over(source_column)
            .alias("_no_reward_stop"),
        )
        .with_columns(
            pl.when(no_reward)
            .then(pl.lit("no_reward"))
            .when(pl.col("stop_time") <= pl.col("_no_reward_start"))
            .then(pl.lit("engaged_1"))
            .when(pl.col("start_time") >= pl.col("_no_reward_stop"))
            .then(pl.lit("engaged_2"))
            .otherwise(pl.lit("transition"))
            .alias(REWARD_BLOCK_COLUMN)
        )
        .drop("_no_reward_start", "_no_reward_stop")
    )


def summarize_task_structure(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    table_index_column: str = "_table_index",
) -> pl.LazyFrame:
    """Audit ordering and the expected E1 -> NR -> E2 task structure."""

    frame = trials.lazy() if isinstance(trials, pl.DataFrame) else trials
    schema_names = set(frame.collect_schema().names())
    ordered = frame.sort(source_column, table_index_column)
    no_reward = pl.col("no_reward_epoch").fill_null(False)
    previous_no_reward = no_reward.shift(1).over(source_column).fill_null(False)
    start_time = pl.col("start_time")
    stop_time = pl.col("stop_time")
    table_index = pl.col(table_index_column)
    invalid_start_time = start_time.is_null() | ~start_time.is_finite()
    invalid_stop_time = stop_time.is_null() | ~stop_time.is_finite()
    invalid_table_index = table_index.is_null() | ~table_index.is_finite()
    nonpositive_duration = ~invalid_start_time & ~invalid_stop_time & (stop_time <= start_time)
    reward_epoch_conflicts = []
    if "no_reward_epoch_conflict" in schema_names:
        reward_epoch_conflicts.append(pl.col("no_reward_epoch_conflict").fill_null(False))
    if "companion_reward_epoch_conflict" in schema_names:
        companion_conflict = pl.col("companion_reward_epoch_conflict").fill_null(False)
        if "no_reward_epoch_source" in schema_names:
            companion_conflict &= pl.col("no_reward_epoch_source") == "companion"
        reward_epoch_conflicts.append(companion_conflict)
    reward_epoch_conflict = (
        pl.any_horizontal(*reward_epoch_conflicts) if reward_epoch_conflicts else pl.lit(False)
    )
    labeled = label_reward_blocks(
        ordered.with_columns(
            (no_reward & ~previous_no_reward).alias("_no_reward_run_start"),
            (pl.col("start_time").diff().over(source_column).fill_null(0) >= 0).alias(
                "_timestamps_monotonic"
            ),
        ),
        source_column=source_column,
    )
    structure = labeled.group_by(source_column).agg(
        pl.len().alias("n_trials"),
        pl.col(table_index_column).n_unique().alias("n_unique_table_indices"),
        (pl.len() - pl.col(table_index_column).n_unique()).alias("n_duplicate_table_indices"),
        invalid_table_index.sum().alias("n_invalid_table_indices"),
        invalid_start_time.sum().alias("n_invalid_start_times"),
        invalid_stop_time.sum().alias("n_invalid_stop_times"),
        nonpositive_duration.sum().alias("n_nonpositive_trial_durations"),
        pl.col("no_reward_epoch").is_null().sum().alias("n_unknown_reward_flags"),
        reward_epoch_conflict.sum().alias("n_reward_epoch_conflicts"),
        pl.col("_no_reward_run_start").sum().alias("n_no_reward_runs"),
        pl.col("_timestamps_monotonic").all().alias("timestamps_monotonic"),
        (pl.col(REWARD_BLOCK_COLUMN) == "engaged_1").sum().alias("n_engaged_1_trials"),
        (pl.col(REWARD_BLOCK_COLUMN) == "no_reward").sum().alias("n_no_reward_trials"),
        (pl.col(REWARD_BLOCK_COLUMN) == "engaged_2").sum().alias("n_engaged_2_trials"),
        (pl.col(REWARD_BLOCK_COLUMN) == "transition").sum().alias("n_transition_trials"),
    )
    return structure.with_columns(
        (
            pl.col("timestamps_monotonic")
            & (pl.col("n_duplicate_table_indices") == 0)
            & (pl.col("n_invalid_table_indices") == 0)
            & (pl.col("n_invalid_start_times") == 0)
            & (pl.col("n_invalid_stop_times") == 0)
            & (pl.col("n_nonpositive_trial_durations") == 0)
            & (pl.col("n_unknown_reward_flags") == 0)
            & (pl.col("n_reward_epoch_conflicts") == 0)
            & (pl.col("n_no_reward_runs") == 1)
            & (pl.col("n_engaged_1_trials") > 0)
            & (pl.col("n_no_reward_trials") > 0)
            & (pl.col("n_engaged_2_trials") > 0)
            & (pl.col("n_transition_trials") == 0)
        ).alias("task_structure_valid")
    )


def summarize_session_behavior(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    no_reward_tail_seconds: float | None = 600.0,
    final_engaged_exclusion_seconds: float = 600.0,
    event_time_column: str = "change_time",
    confidence_level: float = 0.95,
    source_column: str = SOURCE_COLUMN,
) -> pl.LazyFrame:
    """Summarize target and catch responses in reward-availability windows.

    The denominator is completed, non-auto-rewarded NWB ``go`` trials; the
    numerator is the subset carrying the NWB ``hit`` label. This label remains
    defined during the no-reward epoch and therefore measures licking without
    conflating it with reward delivery. By default, only the final ten minutes
    of no reward and all but the final ten minutes of engaged block 2 determine
    session eligibility. Window membership uses the actual change event time,
    not trial start time. Thus early extinction and possible late E2
    disengagement are not selected away.
    """

    if no_reward_tail_seconds is not None:
        _validate_nonnegative_finite("no_reward_tail_seconds", no_reward_tail_seconds)
    _validate_nonnegative_finite("final_engaged_exclusion_seconds", final_engaged_exclusion_seconds)
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    frame = label_reward_blocks(trials, source_column=source_column)
    block_bounds = frame.group_by(source_column).agg(
        *(
            expression
            for block in ("engaged_1", "no_reward", "engaged_2")
            for expression in (
                pl.col("start_time")
                .filter(pl.col(REWARD_BLOCK_COLUMN) == block)
                .min()
                .alias(f"_{block}_start"),
                pl.col("stop_time")
                .filter(pl.col(REWARD_BLOCK_COLUMN) == block)
                .max()
                .alias(f"_{block}_stop"),
            )
        )
    )
    frame = frame.join(
        block_bounds,
        on=source_column,
        how="left",
        validate="m:1",
    )
    go = pl.col("go")
    catch = pl.col("catch")
    aborted = pl.col("aborted")
    auto_rewarded = pl.col("auto_rewarded")
    hit_label = pl.col("hit")
    false_alarm_label = pl.col("false_alarm")
    core_labels_valid = pl.all_horizontal(
        go.is_not_null(),
        catch.is_not_null(),
        aborted.is_not_null(),
        auto_rewarded.is_not_null(),
    )
    candidate_go = go.fill_null(False) & ~aborted.fill_null(False) & ~auto_rewarded.fill_null(False)
    candidate_catch = (
        catch.fill_null(False) & ~aborted.fill_null(False) & ~auto_rewarded.fill_null(False)
    )
    outcome_labels_valid = (~candidate_go | hit_label.is_not_null()) & (
        ~candidate_catch | false_alarm_label.is_not_null()
    )
    behavior_labels_valid = core_labels_valid & outcome_labels_valid
    hit = pl.col("hit").fill_null(False)
    false_alarm = pl.col("false_alarm").fill_null(False)
    event_time = pl.col(event_time_column)
    event_time_valid = event_time.is_not_null() & event_time.is_finite()
    event_time_within_trial = (
        event_time_valid
        & pl.col("start_time").is_not_null()
        & pl.col("start_time").is_finite()
        & pl.col("stop_time").is_not_null()
        & pl.col("stop_time").is_finite()
        & (event_time >= pl.col("start_time"))
        & (event_time <= pl.col("stop_time"))
    )
    candidate_event = candidate_go | candidate_catch
    valid_go = candidate_go & behavior_labels_valid & event_time_within_trial
    valid_catch = candidate_catch & behavior_labels_valid & event_time_within_trial

    expressions: list[pl.Expr] = [
        (~behavior_labels_valid).sum().alias("n_invalid_behavior_labels"),
        (candidate_event & ~event_time_valid).sum().alias("n_invalid_behavior_event_times"),
        (candidate_event & event_time_valid & ~event_time_within_trial)
        .sum()
        .alias("n_behavior_event_times_outside_trial"),
    ]
    for block in ("engaged_1", "no_reward", "engaged_2"):
        in_block = pl.col(REWARD_BLOCK_COLUMN) == block
        if block == "no_reward" and no_reward_tail_seconds is not None:
            in_block &= event_time >= (pl.col("_no_reward_stop") - no_reward_tail_seconds)
        if block == "engaged_2":
            in_block &= event_time < (pl.col("_engaged_2_stop") - final_engaged_exclusion_seconds)
        expressions.extend(
            [
                (in_block & valid_go).sum().alias(f"n_{block}_go_trials"),
                (in_block & valid_go & hit).sum().alias(f"n_{block}_responses"),
                (in_block & valid_catch).sum().alias(f"n_{block}_catch_trials"),
                (in_block & valid_catch & false_alarm).sum().alias(f"n_{block}_false_alarms"),
            ]
        )

    summary = (
        frame.group_by(source_column)
        .agg(*expressions)
        .join(
            summarize_task_structure(trials, source_column=source_column),
            on=source_column,
            how="left",
            validate="1:1",
        )
    )
    rate_expressions = []
    interval_expressions = []
    z_score = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2)
    for block in ("engaged_1", "no_reward", "engaged_2"):
        n_trials = pl.col(f"n_{block}_go_trials")
        n_responses = pl.col(f"n_{block}_responses")
        rate_expressions.append(
            pl.when(n_trials > 0)
            .then(n_responses / n_trials)
            .otherwise(None)
            .alias(f"{block}_response_rate")
        )
        interval_expressions.extend(
            _wilson_interval_exprs(
                n_responses,
                n_trials,
                z_score=z_score,
                name=f"{block}_response_rate",
            )
        )
        n_catch = pl.col(f"n_{block}_catch_trials")
        n_false_alarms = pl.col(f"n_{block}_false_alarms")
        rate_expressions.append(
            pl.when(n_catch > 0)
            .then(n_false_alarms / n_catch)
            .otherwise(None)
            .alias(f"{block}_false_alarm_rate")
        )
        interval_expressions.extend(
            _wilson_interval_exprs(
                n_false_alarms,
                n_catch,
                z_score=z_score,
                name=f"{block}_false_alarm_rate",
            )
        )
    summary = summary.with_columns(*rate_expressions)
    return summary.with_columns(
        *interval_expressions,
        *(_loglinear_dprime_expr(block) for block in ("engaged_1", "no_reward", "engaged_2")),
        (
            pl.min_horizontal("engaged_1_response_rate", "engaged_2_response_rate")
            - pl.col("no_reward_response_rate")
        ).alias("reward_suppression_drop"),
        pl.lit(confidence_level).alias("behavior_confidence_level"),
    )


def good_session_expr(
    thresholds: SessionQualityThresholds = DEFAULT_SESSION_QUALITY_THRESHOLDS,
) -> pl.Expr:
    """Build a predicate for reliable engaged licking and no-reward stopping."""

    enough_trials = pl.all_horizontal(
        *(
            pl.col(f"n_{block}_go_trials") >= thresholds.minimum_go_trials_per_block
            for block in ("engaged_1", "no_reward", "engaged_2")
        )
    )
    enough_engaged_catches = pl.all_horizontal(
        *(
            pl.col(f"n_{block}_catch_trials") >= thresholds.minimum_catch_trials_per_engaged_block
            for block in ("engaged_1", "engaged_2")
        )
    )
    return (
        pl.col("task_structure_valid").fill_null(False)
        & (pl.col("n_invalid_behavior_labels") == 0)
        & (pl.col("n_invalid_behavior_event_times") == 0)
        & (pl.col("n_behavior_event_times_outside_trial") == 0)
        & enough_trials
        & enough_engaged_catches
        & (pl.col("engaged_1_response_rate") >= thresholds.minimum_engaged_response_rate)
        & (pl.col("engaged_2_response_rate") >= thresholds.minimum_engaged_response_rate)
        & (pl.col("engaged_1_dprime") >= thresholds.minimum_engaged_dprime)
        & (pl.col("engaged_2_dprime") >= thresholds.minimum_engaged_dprime)
        & (pl.col("no_reward_response_rate") <= thresholds.maximum_no_reward_response_rate)
        & (pl.col("reward_suppression_drop") >= thresholds.minimum_reward_suppression_drop)
    )


def add_good_session_flag(
    summaries: FrameT,
    *,
    thresholds: SessionQualityThresholds = DEFAULT_SESSION_QUALITY_THRESHOLDS,
) -> FrameT:
    """Add an ``is_good_session`` column to behavioral summaries."""

    enough_trials = pl.all_horizontal(
        *(
            pl.col(f"n_{block}_go_trials") >= thresholds.minimum_go_trials_per_block
            for block in ("engaged_1", "no_reward", "engaged_2")
        )
    ).fill_null(False)
    enough_engaged_catches = pl.all_horizontal(
        *(
            pl.col(f"n_{block}_catch_trials") >= thresholds.minimum_catch_trials_per_engaged_block
            for block in ("engaged_1", "engaged_2")
        )
    ).fill_null(False)
    result = summaries.with_columns(
        (pl.col("n_invalid_behavior_labels") == 0).fill_null(False).alias("behavior_labels_pass"),
        (
            (pl.col("n_invalid_behavior_event_times") == 0)
            & (pl.col("n_behavior_event_times_outside_trial") == 0)
        )
        .fill_null(False)
        .alias("behavior_event_times_pass"),
        enough_trials.alias("minimum_go_trials_pass"),
        enough_engaged_catches.alias("minimum_engaged_catch_trials_pass"),
        (pl.col("engaged_1_response_rate") >= thresholds.minimum_engaged_response_rate)
        .fill_null(False)
        .alias("engaged_1_response_pass"),
        (pl.col("engaged_2_response_rate") >= thresholds.minimum_engaged_response_rate)
        .fill_null(False)
        .alias("engaged_2_response_pass"),
        (pl.col("no_reward_response_rate") <= thresholds.maximum_no_reward_response_rate)
        .fill_null(False)
        .alias("no_reward_response_pass"),
        (pl.col("engaged_1_dprime") >= thresholds.minimum_engaged_dprime)
        .fill_null(False)
        .alias("engaged_1_dprime_pass"),
        (pl.col("engaged_2_dprime") >= thresholds.minimum_engaged_dprime)
        .fill_null(False)
        .alias("engaged_2_dprime_pass"),
        (pl.col("reward_suppression_drop") >= thresholds.minimum_reward_suppression_drop)
        .fill_null(False)
        .alias("reward_suppression_drop_pass"),
        pl.lit(thresholds.minimum_go_trials_per_block).alias("minimum_go_trials_threshold"),
        pl.lit(thresholds.minimum_catch_trials_per_engaged_block).alias(
            "minimum_engaged_catch_trials_threshold"
        ),
        pl.lit(thresholds.minimum_engaged_response_rate).alias(
            "minimum_engaged_response_rate_threshold"
        ),
        pl.lit(thresholds.minimum_engaged_dprime).alias("minimum_engaged_dprime_threshold"),
        pl.lit(thresholds.maximum_no_reward_response_rate).alias(
            "maximum_no_reward_response_rate_threshold"
        ),
        pl.lit(thresholds.minimum_reward_suppression_drop).alias(
            "minimum_reward_suppression_drop_threshold"
        ),
    ).with_columns(good_session_expr(thresholds).fill_null(False).alias("is_good_session"))
    return result.with_columns(
        pl.concat_str(
            pl.when(~pl.col("task_structure_valid").fill_null(False))
            .then(pl.lit("invalid_task_structure;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("behavior_labels_pass"))
            .then(pl.lit("invalid_behavior_labels;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("behavior_event_times_pass"))
            .then(pl.lit("invalid_behavior_event_times;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("minimum_go_trials_pass"))
            .then(pl.lit("insufficient_go_trials;"))
            .otherwise(pl.lit("")),
            pl.when(~pl.col("minimum_engaged_catch_trials_pass"))
            .then(pl.lit("insufficient_engaged_catch_trials;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("engaged_1_response_rate").is_null())
            .then(pl.lit("engaged_1_response_unevaluable;"))
            .when(~pl.col("engaged_1_response_pass"))
            .then(pl.lit("engaged_1_response_below_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("engaged_2_response_rate").is_null())
            .then(pl.lit("engaged_2_response_unevaluable;"))
            .when(~pl.col("engaged_2_response_pass"))
            .then(pl.lit("engaged_2_response_below_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("engaged_1_dprime").is_null())
            .then(pl.lit("engaged_1_dprime_unevaluable;"))
            .when(~pl.col("engaged_1_dprime_pass"))
            .then(pl.lit("engaged_1_dprime_below_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("engaged_2_dprime").is_null())
            .then(pl.lit("engaged_2_dprime_unevaluable;"))
            .when(~pl.col("engaged_2_dprime_pass"))
            .then(pl.lit("engaged_2_dprime_below_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("no_reward_response_rate").is_null())
            .then(pl.lit("no_reward_response_unevaluable;"))
            .when(~pl.col("no_reward_response_pass"))
            .then(pl.lit("no_reward_response_above_threshold;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("reward_suppression_drop").is_null())
            .then(pl.lit("reward_suppression_drop_unevaluable;"))
            .when(~pl.col("reward_suppression_drop_pass"))
            .then(pl.lit("reward_suppression_drop_below_threshold;"))
            .otherwise(pl.lit("")),
        )
        .str.strip_chars_end(";")
        .alias("session_exclusion_reasons")
    )


def filter_good_sessions(
    summaries: FrameT,
    *,
    thresholds: SessionQualityThresholds = DEFAULT_SESSION_QUALITY_THRESHOLDS,
) -> FrameT:
    """Keep sessions passing the explicit behavioral gating criteria."""

    return summaries.filter(good_session_expr(thresholds).fill_null(False))


def get_engaged_block_windows(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    final_engaged_exclusion_seconds: float = 600.0,
    source_column: str = SOURCE_COLUMN,
) -> pl.LazyFrame:
    """Get per-session windows for unit-stability assessment.

    The final ten minutes of the second engaged block are excluded by default,
    as requested, so late behavioral disengagement does not by itself reject an
    otherwise stable unit. A second block shorter than the exclusion yields a
    null analysis stop and will fail stability filtering.
    """

    if not math.isfinite(final_engaged_exclusion_seconds):
        raise ValueError("final_engaged_exclusion_seconds must be finite")
    if final_engaged_exclusion_seconds < 0:
        raise ValueError("final_engaged_exclusion_seconds must be non-negative")

    frame = label_reward_blocks(trials, source_column=source_column)
    engaged_1 = pl.col(REWARD_BLOCK_COLUMN) == "engaged_1"
    engaged_2 = pl.col(REWARD_BLOCK_COLUMN) == "engaged_2"
    recorded_bounds = frame.group_by(source_column).agg(
        pl.col("start_time").filter(engaged_1).min().alias("engaged_1_recorded_start"),
        pl.col("stop_time").filter(engaged_1).max().alias("engaged_1_recorded_stop"),
        pl.col("start_time").filter(engaged_2).min().alias("engaged_2_recorded_start"),
        pl.col("stop_time").filter(engaged_2).max().alias("engaged_2_recorded_stop"),
    )
    windows = recorded_bounds.join(
        summarize_task_structure(trials, source_column=source_column),
        on=source_column,
        how="left",
        validate="1:1",
    ).with_columns(
        (pl.col("engaged_2_recorded_stop") - final_engaged_exclusion_seconds).alias(
            "_engaged_2_candidate_stop"
        ),
        pl.lit(final_engaged_exclusion_seconds).alias("engaged_2_excluded_tail_seconds"),
    )
    recorded_bounds_finite = pl.all_horizontal(
        *(
            pl.col(column).is_not_null() & pl.col(column).is_finite()
            for column in (
                "engaged_1_recorded_start",
                "engaged_1_recorded_stop",
                "engaged_2_recorded_start",
                "engaged_2_recorded_stop",
                "_engaged_2_candidate_stop",
            )
        )
    )
    positive_windows = (pl.col("engaged_1_recorded_stop") > pl.col("engaged_1_recorded_start")) & (
        pl.col("_engaged_2_candidate_stop") > pl.col("engaged_2_recorded_start")
    )
    windows = windows.with_columns(
        (
            pl.col("task_structure_valid").fill_null(False)
            & recorded_bounds_finite
            & positive_windows
        ).alias("engaged_window_valid"),
        pl.when(~pl.col("task_structure_valid").fill_null(False))
        .then(pl.lit("invalid_task_structure"))
        .when(~recorded_bounds_finite)
        .then(pl.lit("invalid_engaged_bounds"))
        .when(pl.col("engaged_1_recorded_stop") <= pl.col("engaged_1_recorded_start"))
        .then(pl.lit("invalid_engaged_1_window"))
        .when(pl.col("_engaged_2_candidate_stop") <= pl.col("engaged_2_recorded_start"))
        .then(pl.lit("engaged_2_exclusion_leaves_no_data"))
        .otherwise(pl.lit("pass"))
        .alias("engaged_window_status"),
    )
    return (
        windows.with_columns(
            pl.when(pl.col("engaged_window_valid"))
            .then(pl.col("engaged_1_recorded_start"))
            .otherwise(None)
            .alias("engaged_1_start"),
            pl.when(pl.col("engaged_window_valid"))
            .then(pl.col("engaged_1_recorded_stop"))
            .otherwise(None)
            .alias("engaged_1_stop"),
            pl.when(pl.col("engaged_window_valid"))
            .then(pl.col("engaged_2_recorded_start"))
            .otherwise(None)
            .alias("engaged_2_start"),
            pl.when(pl.col("engaged_window_valid"))
            .then(pl.col("_engaged_2_candidate_stop"))
            .otherwise(None)
            .alias("engaged_2_stop"),
        )
        .with_columns(
            (pl.col("engaged_1_stop") - pl.col("engaged_1_start")).alias("engaged_1_duration"),
            (pl.col("engaged_2_stop") - pl.col("engaged_2_start")).alias("engaged_2_duration"),
        )
        .drop("_engaged_2_candidate_stop")
    )


def summarize_coarse_engaged_rate_stability(
    units: pl.DataFrame,
    windows: pl.DataFrame,
    *,
    thresholds: CoarseUnitStabilityThresholds = DEFAULT_COARSE_UNIT_STABILITY_THRESHOLDS,
    source_column: str = SOURCE_COLUMN,
    unit_id_column: str = "id",
    spike_times_column: str = "spike_times",
    validate_spike_times: bool = True,
) -> pl.DataFrame:
    """Compute a provisional whole-block firing-rate consistency screen.

    Spike counts use binary searches on the sorted NWB spike-time vectors, so
    cost scales with units rather than with all spikes. The symmetric rate ratio
    is ``min(rate_1, rate_2) / max(rate_1, rate_2)``.

    This coarse screen includes task events and behavior. It is suitable for an
    initial drift diagnostic and sensitivity analysis, but it is not the
    condition-matched, lick-free, noise-calibrated primary stability filter
    proposed in ``ANALYSIS_PLAN.md``.
    """

    required_unit_columns = {source_column, unit_id_column, spike_times_column}
    missing_unit_columns = required_unit_columns.difference(units.columns)
    if missing_unit_columns:
        raise ValueError(f"units is missing columns: {sorted(missing_unit_columns)}")

    required_window_columns = {
        source_column,
        "engaged_1_start",
        "engaged_1_stop",
        "engaged_2_start",
        "engaged_2_stop",
    }
    missing_window_columns = required_window_columns.difference(windows.columns)
    if missing_window_columns:
        raise ValueError(f"windows is missing columns: {sorted(missing_window_columns)}")
    if windows.get_column(source_column).n_unique() != windows.height:
        raise ValueError(f"windows must have exactly one row per {source_column}")
    unit_keys = units.select(source_column, unit_id_column)
    if unit_keys.is_duplicated().any():
        raise ValueError(f"units must have unique ({source_column}, {unit_id_column}) keys")

    optional_window_columns = [
        column
        for column in ("engaged_window_valid", "engaged_window_status")
        if column in windows.columns
    ]
    window_by_source = {
        row[source_column]: row
        for row in windows.select(*required_window_columns, *optional_window_columns).iter_rows(
            named=True
        )
    }
    records: list[dict[str, Any]] = []
    for row in units.select(source_column, unit_id_column, spike_times_column).iter_rows(
        named=True
    ):
        source = row[source_column]
        raw_spikes = row[spike_times_column]
        spikes = None if raw_spikes is None else np.asarray(raw_spikes, dtype=float)
        if spikes is not None and validate_spike_times:
            try:
                _validate_spike_times(spikes)
            except ValueError as error:
                raise ValueError(
                    f"invalid {spike_times_column} for source {source!r}, "
                    f"unit {row[unit_id_column]!r}: {error}"
                ) from error

        window = window_by_source.get(source)
        if window is None:
            valid_windows = False
            status = "missing_window"
            first_start = first_stop = second_start = second_stop = None
        else:
            first_start = window["engaged_1_start"]
            first_stop = window["engaged_1_stop"]
            second_start = window["engaged_2_start"]
            second_stop = window["engaged_2_stop"]
            bounds = (first_start, first_stop, second_start, second_stop)
            bounds_are_finite = all(_is_finite_number(value) for value in bounds)
            bounds_are_ordered = bounds_are_finite and (
                first_stop > first_start and second_stop > second_start
            )
            marked_valid = window.get("engaged_window_valid", True)
            valid_windows = bounds_are_ordered and marked_valid is True
            status = "invalid_window"

        if valid_windows and spikes is None:
            status = "missing_spike_times"
        if valid_windows and spikes is not None:
            first_count = _count_spikes(spikes, first_start, first_stop)
            second_count = _count_spikes(spikes, second_start, second_stop)
            first_rate = first_count / (first_stop - first_start)
            second_rate = second_count / (second_stop - second_start)
            maximum_rate = max(first_rate, second_rate)
            rate_ratio = (
                min(first_rate, second_rate) / maximum_rate if maximum_rate > 0 else math.nan
            )
        else:
            first_count = None
            second_count = None
            first_rate = None
            second_rate = None
            rate_ratio = None

        consistent = False
        if valid_windows and spikes is not None:
            consistent = (
                first_count >= thresholds.minimum_spikes_per_block
                and second_count >= thresholds.minimum_spikes_per_block
                and math.isfinite(rate_ratio)
                and rate_ratio >= thresholds.minimum_engaged_rate_ratio
            )
            if (
                first_count < thresholds.minimum_spikes_per_block
                or second_count < thresholds.minimum_spikes_per_block
            ):
                status = "insufficient_spikes"
            elif not math.isfinite(rate_ratio):
                status = "indeterminate_rate_ratio"
            elif rate_ratio < thresholds.minimum_engaged_rate_ratio:
                status = "rate_ratio_below_threshold"
            else:
                status = "pass"
        records.append(
            {
                source_column: source,
                unit_id_column: row[unit_id_column],
                "engaged_1_spike_count": first_count,
                "engaged_2_spike_count": second_count,
                "engaged_1_rate_hz": first_rate,
                "engaged_2_rate_hz": second_rate,
                "engaged_rate_ratio": rate_ratio,
                "coarse_engaged_rate_status": status,
                COARSE_UNIT_STABILITY_COLUMN: consistent,
            }
        )

    unit_id_dtype = units.schema[unit_id_column]
    output_schema = {
        source_column: units.schema[source_column],
        unit_id_column: unit_id_dtype,
        "engaged_1_spike_count": pl.Int64,
        "engaged_2_spike_count": pl.Int64,
        "engaged_1_rate_hz": pl.Float64,
        "engaged_2_rate_hz": pl.Float64,
        "engaged_rate_ratio": pl.Float64,
        "coarse_engaged_rate_status": pl.String,
        COARSE_UNIT_STABILITY_COLUMN: pl.Boolean,
    }
    if not records:
        return pl.DataFrame(schema=output_schema)
    return pl.DataFrame(
        records,
        schema_overrides=output_schema,
    )


def filter_coarse_engaged_rate_consistent_units(
    units: pl.DataFrame | pl.LazyFrame,
    stability: pl.DataFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    unit_id_column: str = "id",
) -> pl.DataFrame | pl.LazyFrame:
    """Keep units passing the provisional coarse engaged-rate screen."""

    keys = [source_column, unit_id_column]
    if isinstance(units, pl.LazyFrame):
        return units.join(stability.lazy(), on=keys, how="inner", validate="1:1").filter(
            pl.col(COARSE_UNIT_STABILITY_COLUMN)
        )
    return units.join(stability, on=keys, how="inner", validate="1:1").filter(
        pl.col(COARSE_UNIT_STABILITY_COLUMN)
    )


def _count_spikes(spikes: np.ndarray, start: float, stop: float) -> int:
    if stop <= start:
        raise ValueError("engaged block stops must be greater than starts")
    left = np.searchsorted(spikes, start, side="left")
    right = np.searchsorted(spikes, stop, side="left")
    return int(right - left)


def _validate_spike_times(spikes: np.ndarray) -> None:
    if spikes.ndim != 1:
        raise ValueError("spike_times must be one-dimensional")
    if not np.isfinite(spikes).all():
        raise ValueError("spike_times must contain only finite values")
    if spikes.size > 1 and np.any(np.diff(spikes) < 0):
        raise ValueError("spike_times must be sorted in nondecreasing order")


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _wilson_interval_exprs(
    successes: pl.Expr,
    trials: pl.Expr,
    *,
    z_score: float,
    name: str,
) -> tuple[pl.Expr, pl.Expr]:
    """Return lower and upper Wilson score interval expressions."""

    proportion = successes / trials
    z_squared = z_score**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        z_score
        * (proportion * (1 - proportion) / trials + z_squared / (4 * trials**2)).sqrt()
        / denominator
    )
    valid = trials > 0
    return (
        pl.when(valid)
        .then((center - half_width).clip(0, 1))
        .otherwise(None)
        .alias(f"{name}_ci_low"),
        pl.when(valid)
        .then((center + half_width).clip(0, 1))
        .otherwise(None)
        .alias(f"{name}_ci_high"),
    )


def _loglinear_dprime_expr(block: str) -> pl.Expr:
    """Return a finite-sample-corrected d-prime expression for one block."""

    n_go = pl.col(f"n_{block}_go_trials")
    n_responses = pl.col(f"n_{block}_responses")
    n_catch = pl.col(f"n_{block}_catch_trials")
    n_false_alarms = pl.col(f"n_{block}_false_alarms")
    hit_rate = (n_responses + 0.5) / (n_go + 1.0)
    false_alarm_rate = (n_false_alarms + 0.5) / (n_catch + 1.0)
    return (
        pl.when((n_go > 0) & (n_catch > 0))
        .then(
            hit_rate.map_batches(_standard_normal_quantiles, return_dtype=pl.Float64)
            - false_alarm_rate.map_batches(
                _standard_normal_quantiles,
                return_dtype=pl.Float64,
            )
        )
        .otherwise(None)
        .alias(f"{block}_dprime")
    )


def _standard_normal_quantiles(values: pl.Series) -> pl.Series:
    normal = statistics.NormalDist()
    return pl.Series(
        [None if value is None else normal.inv_cdf(float(value)) for value in values],
        dtype=pl.Float64,
    )
