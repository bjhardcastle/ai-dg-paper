"""Publication-facing descriptive behavior tables for Dynamic Gating.

This module is deliberately limited to behavior. It reuses the audited trial
semantics in :mod:`dg.quality`, never reads neural columns, and does not perform
population inference. The outputs separate technical validity from the
threshold-selected neural-analysis cohort so that behavioral gating can be
described across all technically valid sessions.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

import polars as pl

import dg.quality

SOURCE_COLUMN = dg.quality.SOURCE_COLUMN
DEFAULT_MOUSE_COLUMN = "subject_id"

PRIMARY_BLOCKS = ("engaged_1", "no_reward_late", "engaged_2_early")
CHARACTERIZATION_BLOCK = "engaged_2_late"
BEHAVIOR_BLOCKS = (*PRIMARY_BLOCKS, CHARACTERIZATION_BLOCK)

RESPONSE_GATING_ESTIMATE_COLUMNS = (
    "reversible_gating_estimate",
    "withdrawal_suppression_estimate",
    "restoration_recovery_estimate",
    "engaged_response_drift_estimate",
)
SPECIFICITY_GATING_ESTIMATE_COLUMNS = (
    "reversible_specificity_gating_estimate",
    "withdrawal_specificity_suppression_estimate",
    "restoration_specificity_recovery_estimate",
    "engaged_specificity_drift_estimate",
)

_REQUIRED_TRIAL_COLUMNS = (
    "_table_index",
    "start_time",
    "stop_time",
    "go",
    "catch",
    "aborted",
    "auto_rewarded",
    "hit",
    "miss",
    "false_alarm",
    "correct_reject",
    "no_reward_epoch",
    "response_in_window",
)
_OPTIONAL_TRIAL_AUDIT_COLUMNS = (
    "no_reward_epoch_conflict",
    "companion_reward_epoch_conflict",
    "no_reward_epoch_source",
)
_BLOCK_COLUMNS = {
    "engaged_1": (
        "engaged_1_response_probability",
        "n_engaged_1_responses",
        "n_engaged_1_go_trials",
        "engaged_1_false_alarm_probability",
        "n_engaged_1_false_alarms",
        "n_engaged_1_catch_trials",
        "engaged_1_dprime",
    ),
    "no_reward_late": (
        "no_reward_late_response_probability",
        "n_no_reward_late_responses",
        "n_no_reward_late_go_trials",
        "no_reward_late_false_alarm_probability",
        "n_no_reward_late_false_alarms",
        "n_no_reward_late_catch_trials",
        "no_reward_late_dprime",
    ),
    "engaged_2_early": (
        "engaged_2_early_response_probability",
        "n_engaged_2_early_responses",
        "n_engaged_2_early_go_trials",
        "engaged_2_early_false_alarm_probability",
        "n_engaged_2_early_false_alarms",
        "n_engaged_2_early_catch_trials",
        "engaged_2_early_dprime",
    ),
    "engaged_2_late": (
        "engaged_2_late_response_probability",
        "n_engaged_2_late_responses",
        "n_engaged_2_late_go_trials",
        "engaged_2_late_false_alarm_probability",
        "n_engaged_2_late_false_alarms",
        "n_engaged_2_late_catch_trials",
        "engaged_2_late_dprime",
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class BehaviorTables:
    """Descriptive tables produced by :func:`build_behavior_tables`."""

    sessions: pl.DataFrame
    session_blocks: pl.DataFrame
    mouse_blocks: pl.DataFrame
    mouse_gating: pl.DataFrame
    session_attrition: pl.DataFrame


def add_response_in_window(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    response_window_start_seconds: float = 0.150,
    response_window_stop_seconds: float = 0.750,
    event_time_column: str = "change_time",
    lick_times_column: str = "lick_times",
) -> pl.DataFrame | pl.LazyFrame:
    """Compatibility wrapper for the canonical quality-layer response helper.

    Prefer :func:`dg.quality.add_trial_response_from_licks` in new code. This
    wrapper keeps the behavior workflow discoverable while delegating response
    derivation, raw-vector validation, latency, count, and status fields to the
    single canonical implementation.
    """

    return dg.quality.add_trial_response_from_licks(
        trials,
        response_window_start_seconds=response_window_start_seconds,
        response_window_stop_seconds=response_window_stop_seconds,
        event_time_column=event_time_column,
        lick_times_column=lick_times_column,
    )


def summarize_behavior_sessions(
    trials: pl.DataFrame | pl.LazyFrame,
    session_inventory: pl.DataFrame | pl.LazyFrame,
    *,
    thresholds: dg.quality.SessionQualityThresholds = (
        dg.quality.DEFAULT_SESSION_QUALITY_THRESHOLDS
    ),
    late_no_reward_seconds: float = 600.0,
    late_engaged_2_seconds: float = 600.0,
    event_time_column: str = "change_time",
    confidence_level: float = 0.95,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return one descriptive behavior row per inventoried session.

    E1, the final ``late_no_reward_seconds`` of NR, and E2 before its final
    ``late_engaged_2_seconds`` define the continuous reversible-gating score.
    The final E2 window is retained only for characterization and never enters
    ``is_good_session`` or the gating score. A positive score means responses
    were, descriptively, higher in the two reward-available windows than in
    late NR::

        0.5 * E1 - late_NR + 0.5 * early_E2

    ``response_in_window`` must be the canonical response derived from raw lick
    times with :func:`add_response_in_window`; author ``hit`` and
    ``false_alarm`` labels are retained only for agreement auditing.
    ``is_technically_valid`` uses task structure, canonical-response
    completeness, and event timing only. ``is_good_session`` additionally
    applies the explicit behavior thresholds in :mod:`dg.quality`. No
    statistical inference is performed here.

    The high-level function materializes only the required behavioral columns
    once. Any neural or other large columns in the input LazyFrame are removed
    by projection before collection.
    """

    _validate_nonnegative_finite("late_no_reward_seconds", late_no_reward_seconds)
    _validate_nonnegative_finite("late_engaged_2_seconds", late_engaged_2_seconds)

    trial_schema = _require_columns(
        trials,
        (source_column, *_REQUIRED_TRIAL_COLUMNS, event_time_column),
        frame_name="trials",
    )
    trial_columns = list(
        dict.fromkeys((source_column, *_REQUIRED_TRIAL_COLUMNS, event_time_column))
    )
    trial_columns.extend(
        column for column in _OPTIONAL_TRIAL_AUDIT_COLUMNS if column in trial_schema
    )
    trial_data = _collect_projection(trials, trial_columns)
    _validate_non_null(trial_data, source_column, frame_name="trials")
    if trial_data.schema["response_in_window"] != pl.Boolean:
        raise ValueError("trials.response_in_window must have Boolean dtype")

    _require_columns(
        session_inventory,
        (source_column, mouse_column),
        frame_name="session_inventory",
    )
    inventory = _collect_projection(session_inventory, (source_column, mouse_column))
    _validate_non_null(inventory, source_column, frame_name="session_inventory")
    _validate_non_null(inventory, mouse_column, frame_name="session_inventory")
    _validate_unique_key(inventory, (source_column,), frame_name="session_inventory")

    unknown_sources = (
        trial_data.select(source_column)
        .unique()
        .join(inventory.select(source_column), on=source_column, how="anti")
    )
    if unknown_sources.height:
        examples = unknown_sources.get_column(source_column).head(3).to_list()
        raise ValueError(f"trials contain sources absent from session_inventory: {examples!r}")

    primary = dg.quality.summarize_session_behavior(
        trial_data,
        no_reward_tail_seconds=late_no_reward_seconds,
        final_engaged_exclusion_seconds=late_engaged_2_seconds,
        event_time_column=event_time_column,
        response_column="response_in_window",
        confidence_level=confidence_level,
        source_column=source_column,
    ).collect()
    all_engaged_2 = dg.quality.summarize_session_behavior(
        trial_data,
        no_reward_tail_seconds=late_no_reward_seconds,
        final_engaged_exclusion_seconds=0.0,
        event_time_column=event_time_column,
        response_column="response_in_window",
        confidence_level=confidence_level,
        source_column=source_column,
    ).collect()
    _validate_unique_key(primary, (source_column,), frame_name="primary behavior summary")
    _validate_unique_key(all_engaged_2, (source_column,), frame_name="all-E2 behavior summary")

    label_agreement = _summarize_author_label_agreement(
        trial_data,
        source_column=source_column,
    )
    audited = dg.quality.add_good_session_flag(primary, thresholds=thresholds).join(
        label_agreement,
        on=source_column,
        how="left",
        validate="1:1",
    )
    all_e2_counts = all_engaged_2.select(
        source_column,
        pl.col("n_engaged_2_go_trials").alias("_n_engaged_2_all_go_trials"),
        pl.col("n_engaged_2_responses").alias("_n_engaged_2_all_responses"),
        pl.col("n_engaged_2_catch_trials").alias("_n_engaged_2_all_catch_trials"),
        pl.col("n_engaged_2_false_alarms").alias("_n_engaged_2_all_false_alarms"),
    )
    summarized = (
        audited.join(all_e2_counts, on=source_column, how="left", validate="1:1")
        .with_columns(
            (pl.col("_n_engaged_2_all_go_trials") - pl.col("n_engaged_2_go_trials")).alias(
                "n_engaged_2_late_go_trials"
            ),
            (pl.col("_n_engaged_2_all_responses") - pl.col("n_engaged_2_responses")).alias(
                "n_engaged_2_late_responses"
            ),
            (pl.col("_n_engaged_2_all_catch_trials") - pl.col("n_engaged_2_catch_trials")).alias(
                "n_engaged_2_late_catch_trials"
            ),
            (pl.col("_n_engaged_2_all_false_alarms") - pl.col("n_engaged_2_false_alarms")).alias(
                "n_engaged_2_late_false_alarms"
            ),
        )
        .with_columns(
            pl.col("engaged_1_response_rate").alias("engaged_1_response_probability"),
            pl.col("n_engaged_1_responses").alias("n_engaged_1_responses"),
            pl.col("engaged_1_false_alarm_rate").alias("engaged_1_false_alarm_probability"),
            pl.col("no_reward_response_rate").alias("no_reward_late_response_probability"),
            pl.col("n_no_reward_go_trials").alias("n_no_reward_late_go_trials"),
            pl.col("n_no_reward_responses").alias("n_no_reward_late_responses"),
            pl.col("no_reward_false_alarm_rate").alias("no_reward_late_false_alarm_probability"),
            pl.col("n_no_reward_catch_trials").alias("n_no_reward_late_catch_trials"),
            pl.col("n_no_reward_false_alarms").alias("n_no_reward_late_false_alarms"),
            pl.col("no_reward_dprime").alias("no_reward_late_dprime"),
            pl.col("engaged_2_response_rate").alias("engaged_2_early_response_probability"),
            pl.col("n_engaged_2_go_trials").alias("n_engaged_2_early_go_trials"),
            pl.col("n_engaged_2_responses").alias("n_engaged_2_early_responses"),
            pl.col("engaged_2_false_alarm_rate").alias("engaged_2_early_false_alarm_probability"),
            pl.col("n_engaged_2_catch_trials").alias("n_engaged_2_early_catch_trials"),
            pl.col("n_engaged_2_false_alarms").alias("n_engaged_2_early_false_alarms"),
            pl.col("engaged_2_dprime").alias("engaged_2_early_dprime"),
            pl.when(pl.col("n_engaged_2_late_go_trials") > 0)
            .then(pl.col("n_engaged_2_late_responses") / pl.col("n_engaged_2_late_go_trials"))
            .otherwise(None)
            .alias("engaged_2_late_response_probability"),
            pl.when(pl.col("n_engaged_2_late_catch_trials") > 0)
            .then(pl.col("n_engaged_2_late_false_alarms") / pl.col("n_engaged_2_late_catch_trials"))
            .otherwise(None)
            .alias("engaged_2_late_false_alarm_probability"),
            pl.lit(late_no_reward_seconds).alias("late_no_reward_seconds"),
            pl.lit(late_engaged_2_seconds).alias("late_engaged_2_seconds"),
        )
        .with_columns(_loglinear_dprime_expr("engaged_2_late").alias("engaged_2_late_dprime"))
        .with_columns(pl.lit(True).alias("_behavior_summary_available"))
    )
    negative_late_counts = summarized.filter(
        (pl.col("n_engaged_2_late_go_trials") < 0)
        | (pl.col("n_engaged_2_late_responses") < 0)
        | (pl.col("n_engaged_2_late_catch_trials") < 0)
        | (pl.col("n_engaged_2_late_false_alarms") < 0)
    )
    if negative_late_counts.height:
        raise RuntimeError("all-E2 counts were smaller than early-E2 counts")

    sessions = inventory.join(summarized, on=source_column, how="left", validate="1:1")
    available = pl.col("_behavior_summary_available").fill_null(False)
    task_structure_valid = pl.col("task_structure_valid").fill_null(False)
    behavior_labels_valid = (pl.col("n_invalid_behavior_labels") == 0).fill_null(False)
    event_times_valid = (
        (pl.col("n_invalid_behavior_event_times") == 0)
        & (pl.col("n_behavior_event_times_outside_trial") == 0)
    ).fill_null(False)
    sessions = sessions.with_columns(
        (available & task_structure_valid & behavior_labels_valid & event_times_valid).alias(
            "is_technically_valid"
        ),
        pl.concat_str(
            pl.when(~available).then(pl.lit("missing_behavior_trials;")).otherwise(pl.lit("")),
            pl.when(available & ~task_structure_valid)
            .then(pl.lit("invalid_task_structure;"))
            .otherwise(pl.lit("")),
            pl.when(available & ~behavior_labels_valid)
            .then(pl.lit("invalid_behavior_labels;"))
            .otherwise(pl.lit("")),
            pl.when(available & ~event_times_valid)
            .then(pl.lit("invalid_behavior_event_times;"))
            .otherwise(pl.lit("")),
        )
        .str.strip_chars_end(";")
        .alias("technical_exclusion_reasons"),
    ).with_columns(
        (pl.col("is_good_session").fill_null(False) & pl.col("is_technically_valid")).alias(
            "is_good_session"
        ),
        pl.when(~available)
        .then(pl.lit("missing_behavior_trials"))
        .otherwise(pl.col("session_exclusion_reasons"))
        .alias("session_exclusion_reasons"),
    )

    sessions = sessions.with_columns(
        (
            pl.col("engaged_1_response_probability") - pl.col("engaged_1_false_alarm_probability")
        ).alias("engaged_1_response_specificity"),
        (
            pl.col("no_reward_late_response_probability")
            - pl.col("no_reward_late_false_alarm_probability")
        ).alias("no_reward_late_response_specificity"),
        (
            pl.col("engaged_2_early_response_probability")
            - pl.col("engaged_2_early_false_alarm_probability")
        ).alias("engaged_2_early_response_specificity"),
        (
            pl.col("engaged_2_late_response_probability")
            - pl.col("engaged_2_late_false_alarm_probability")
        ).alias("engaged_2_late_response_specificity"),
    )

    primary_probabilities = [
        pl.col("engaged_1_response_probability"),
        pl.col("no_reward_late_response_probability"),
        pl.col("engaged_2_early_response_probability"),
    ]
    primary_probabilities_valid = pl.all_horizontal(
        *(value.is_not_null() & value.is_finite() for value in primary_probabilities)
    )
    primary_specificities = [
        pl.col("engaged_1_response_specificity"),
        pl.col("no_reward_late_response_specificity"),
        pl.col("engaged_2_early_response_specificity"),
    ]
    primary_specificities_valid = pl.all_horizontal(
        *(value.is_not_null() & value.is_finite() for value in primary_specificities)
    )
    sessions = sessions.with_columns(
        pl.when(pl.col("is_technically_valid") & primary_probabilities_valid)
        .then(
            0.5 * pl.col("engaged_1_response_probability")
            - pl.col("no_reward_late_response_probability")
            + 0.5 * pl.col("engaged_2_early_response_probability")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("reversible_gating_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_probabilities_valid)
        .then(
            pl.col("engaged_1_response_probability") - pl.col("no_reward_late_response_probability")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("withdrawal_suppression_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_probabilities_valid)
        .then(
            pl.col("engaged_2_early_response_probability")
            - pl.col("no_reward_late_response_probability")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("restoration_recovery_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_probabilities_valid)
        .then(
            pl.col("engaged_2_early_response_probability")
            - pl.col("engaged_1_response_probability")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("engaged_response_drift_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_specificities_valid)
        .then(
            0.5 * pl.col("engaged_1_response_specificity")
            - pl.col("no_reward_late_response_specificity")
            + 0.5 * pl.col("engaged_2_early_response_specificity")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("reversible_specificity_gating_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_specificities_valid)
        .then(
            pl.col("engaged_1_response_specificity") - pl.col("no_reward_late_response_specificity")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("withdrawal_specificity_suppression_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_specificities_valid)
        .then(
            pl.col("engaged_2_early_response_specificity")
            - pl.col("no_reward_late_response_specificity")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("restoration_specificity_recovery_estimate"),
        pl.when(pl.col("is_technically_valid") & primary_specificities_valid)
        .then(
            pl.col("engaged_2_early_response_specificity")
            - pl.col("engaged_1_response_specificity")
        )
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("engaged_specificity_drift_estimate"),
    ).with_columns(
        pl.col("reversible_gating_estimate").is_not_null().alias("reversible_gating_estimable"),
        pl.when(~pl.col("is_technically_valid"))
        .then(pl.lit("invalid_technical"))
        .when(pl.col("reversible_gating_estimate").is_null())
        .then(pl.lit("insufficient_primary_go_trials"))
        .otherwise(pl.lit("estimable"))
        .alias("reversible_gating_status"),
    )
    return sessions.drop(
        "_behavior_summary_available",
        "_n_engaged_2_all_go_trials",
        "_n_engaged_2_all_responses",
        "_n_engaged_2_all_catch_trials",
        "_n_engaged_2_all_false_alarms",
    ).sort(source_column)


def make_session_block_table(
    sessions: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> pl.DataFrame:
    """Reshape session response probabilities to one row per session and block."""

    required = [
        source_column,
        mouse_column,
        "is_technically_valid",
        "is_good_session",
        *(column for columns in _BLOCK_COLUMNS.values() for column in columns),
    ]
    _require_columns(sessions, required, frame_name="sessions")
    data = _collect_projection(sessions, required)
    _validate_non_null(data, source_column, frame_name="sessions")
    _validate_non_null(data, mouse_column, frame_name="sessions")
    _validate_unique_key(data, (source_column,), frame_name="sessions")

    blocks = []
    for order, block in enumerate(BEHAVIOR_BLOCKS, start=1):
        (
            response_probability,
            n_responses,
            n_go_trials,
            false_alarm_probability,
            n_false_alarms,
            n_catch_trials,
            dprime,
        ) = _BLOCK_COLUMNS[block]
        blocks.append(
            data.select(
                source_column,
                mouse_column,
                pl.lit(block).alias("behavior_block"),
                pl.lit(order, dtype=pl.Int8).alias("block_order"),
                pl.lit(block in PRIMARY_BLOCKS).alias("is_primary_gating_block"),
                pl.col(response_probability).cast(pl.Float64).alias("response_probability"),
                pl.col(n_responses).cast(pl.Int64).alias("n_responses"),
                pl.col(n_go_trials).cast(pl.Int64).alias("n_go_trials"),
                pl.col(false_alarm_probability).cast(pl.Float64).alias("false_alarm_probability"),
                pl.col(n_false_alarms).cast(pl.Int64).alias("n_false_alarms"),
                pl.col(n_catch_trials).cast(pl.Int64).alias("n_catch_trials"),
                pl.col(dprime).cast(pl.Float64).alias("dprime"),
                "is_technically_valid",
                "is_good_session",
            )
        )
    return pl.concat(blocks, how="vertical").sort(mouse_column, source_column, "block_order")


def aggregate_mouse_blocks(
    session_blocks: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> pl.DataFrame:
    """Average session-level block probabilities equally within each mouse.

    The long-form output contains both the all-technically-valid cohort used for
    the behavioral description and the threshold-selected cohort used only as
    QC characterization. Trial counts are retained as denominators but are not
    used as aggregation weights.
    """

    required = (
        source_column,
        mouse_column,
        "behavior_block",
        "block_order",
        "response_probability",
        "n_responses",
        "n_go_trials",
        "false_alarm_probability",
        "n_false_alarms",
        "n_catch_trials",
        "dprime",
        "is_technically_valid",
        "is_good_session",
    )
    _require_columns(session_blocks, required, frame_name="session_blocks")
    data = _collect_projection(session_blocks, required)
    _validate_non_null(data, source_column, frame_name="session_blocks")
    _validate_non_null(data, mouse_column, frame_name="session_blocks")
    _validate_unique_key(
        data,
        (source_column, "behavior_block"),
        frame_name="session_blocks",
    )
    _validate_session_block_identity(
        data,
        source_column=source_column,
        mouse_column=mouse_column,
    )

    block_index = data.select("behavior_block", "block_order").unique()
    _validate_unique_key(block_index, ("behavior_block",), frame_name="block index")
    mice = data.select(mouse_column).unique()
    grid = mice.join(block_index, how="cross")
    session_flags = data.unique(subset=[source_column]).select(
        source_column,
        mouse_column,
        "is_technically_valid",
        "is_good_session",
    )

    cohorts = []
    for cohort, flag_column in (
        ("technically_valid", "is_technically_valid"),
        ("threshold_selected", "is_good_session"),
    ):
        cohort_sizes = session_flags.group_by(mouse_column).agg(
            pl.len().alias("n_sessions_inventory"),
            pl.col(flag_column).sum().cast(pl.Int64).alias("n_sessions_in_cohort"),
        )
        values = (
            data.filter(pl.col(flag_column))
            .group_by(mouse_column, "behavior_block", "block_order")
            .agg(
                pl.col("response_probability").mean().alias("response_probability"),
                (
                    pl.col("response_probability").is_not_null()
                    & pl.col("response_probability").is_finite()
                )
                .sum()
                .cast(pl.Int64)
                .alias("n_sessions_response_contributing"),
                pl.col("n_responses").sum().cast(pl.Int64).alias("n_responses"),
                pl.col("n_go_trials").sum().cast(pl.Int64).alias("n_go_trials"),
                pl.col("false_alarm_probability").mean().alias("false_alarm_probability"),
                (
                    pl.col("false_alarm_probability").is_not_null()
                    & pl.col("false_alarm_probability").is_finite()
                )
                .sum()
                .cast(pl.Int64)
                .alias("n_sessions_false_alarm_contributing"),
                pl.col("n_false_alarms").sum().cast(pl.Int64).alias("n_false_alarms"),
                pl.col("n_catch_trials").sum().cast(pl.Int64).alias("n_catch_trials"),
                pl.col("dprime").mean().alias("dprime"),
                (pl.col("dprime").is_not_null() & pl.col("dprime").is_finite())
                .sum()
                .cast(pl.Int64)
                .alias("n_sessions_dprime_contributing"),
            )
        )
        cohorts.append(
            grid.join(
                values,
                on=[mouse_column, "behavior_block", "block_order"],
                how="left",
                validate="1:1",
            )
            .join(cohort_sizes, on=mouse_column, how="left", validate="m:1")
            .with_columns(
                pl.lit(cohort).alias("session_cohort"),
                pl.col("n_sessions_response_contributing").fill_null(0),
                pl.col("n_sessions_false_alarm_contributing").fill_null(0),
                pl.col("n_sessions_dprime_contributing").fill_null(0),
                pl.col("n_responses").fill_null(0),
                pl.col("n_go_trials").fill_null(0),
                pl.col("n_false_alarms").fill_null(0),
                pl.col("n_catch_trials").fill_null(0),
                pl.lit("equal_session_mean").alias("aggregation_method"),
            )
            .with_columns(
                pl.col("n_sessions_response_contributing").alias("n_sessions_contributing")
            )
        )
    return pl.concat(cohorts, how="vertical").sort(mouse_column, "session_cohort", "block_order")


def aggregate_mouse_gating(
    sessions: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> pl.DataFrame:
    """Average complete per-session reversible-gating estimates within mouse."""

    required = (
        source_column,
        mouse_column,
        "is_technically_valid",
        "is_good_session",
        *RESPONSE_GATING_ESTIMATE_COLUMNS,
        *SPECIFICITY_GATING_ESTIMATE_COLUMNS,
    )
    _require_columns(sessions, required, frame_name="sessions")
    data = _collect_projection(sessions, required)
    _validate_non_null(data, source_column, frame_name="sessions")
    _validate_non_null(data, mouse_column, frame_name="sessions")
    _validate_unique_key(data, (source_column,), frame_name="sessions")
    mice = data.select(mouse_column).unique()

    cohorts = []
    for cohort, flag_column in (
        ("technically_valid", "is_technically_valid"),
        ("threshold_selected", "is_good_session"),
    ):
        cohort_sizes = data.group_by(mouse_column).agg(
            pl.len().alias("n_sessions_inventory"),
            pl.col(flag_column).sum().cast(pl.Int64).alias("n_sessions_in_cohort"),
        )
        estimates = (
            data.filter(
                pl.col(flag_column)
                & pl.col("reversible_gating_estimate").is_not_null()
                & pl.col("reversible_gating_estimate").is_finite()
            )
            .group_by(mouse_column)
            .agg(
                *(
                    pl.col(column).mean().alias(column)
                    for column in RESPONSE_GATING_ESTIMATE_COLUMNS
                ),
                pl.len().cast(pl.Int64).alias("n_sessions_contributing"),
                *(
                    pl.col(column).mean().alias(column)
                    for column in SPECIFICITY_GATING_ESTIMATE_COLUMNS
                ),
                pl.col("reversible_specificity_gating_estimate")
                .is_not_null()
                .sum()
                .cast(pl.Int64)
                .alias("n_sessions_specificity_contributing"),
            )
        )
        cohorts.append(
            mice.join(estimates, on=mouse_column, how="left", validate="1:1")
            .join(cohort_sizes, on=mouse_column, how="left", validate="1:1")
            .with_columns(
                pl.lit(cohort).alias("session_cohort"),
                pl.col("n_sessions_contributing").fill_null(0),
                pl.col("n_sessions_specificity_contributing").fill_null(0),
                pl.lit("equal_session_mean").alias("aggregation_method"),
            )
        )
    return pl.concat(cohorts, how="vertical").sort(mouse_column, "session_cohort")


def summarize_session_attrition(
    sessions: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return cumulative session counts for technical and behavior gates."""

    required = (
        source_column,
        mouse_column,
        "is_technically_valid",
        "is_good_session",
        "reversible_gating_estimable",
    )
    _require_columns(sessions, required, frame_name="sessions")
    data = _collect_projection(sessions, required)
    _validate_non_null(data, source_column, frame_name="sessions")
    _validate_non_null(data, mouse_column, frame_name="sessions")
    _validate_unique_key(data, (source_column,), frame_name="sessions")

    stages = (
        ("inventory", "inventory", pl.lit(True)),
        ("technically_valid", "technical", pl.col("is_technically_valid")),
        (
            "reversible_gating_estimable",
            "estimability",
            pl.col("is_technically_valid") & pl.col("reversible_gating_estimable"),
        ),
        ("threshold_selected", "behavior_threshold", pl.col("is_good_session")),
    )
    n_inventory = data.height
    previous_count = n_inventory
    rows = []
    for order, (stage, criterion_type, predicate) in enumerate(stages):
        selected = data.filter(predicate)
        n_sessions = selected.height
        rows.append(
            {
                "stage_order": order,
                "attrition_stage": stage,
                "criterion_type": criterion_type,
                "n_sessions": n_sessions,
                "n_mice": selected.get_column(mouse_column).n_unique(),
                "n_sessions_removed_at_stage": previous_count - n_sessions,
                "n_sessions_removed_from_inventory": n_inventory - n_sessions,
                "fraction_of_inventory": (n_sessions / n_inventory if n_inventory else None),
            }
        )
        previous_count = n_sessions
    return pl.DataFrame(rows, infer_schema_length=None)


def build_behavior_tables(
    trials: pl.DataFrame | pl.LazyFrame,
    session_inventory: pl.DataFrame | pl.LazyFrame,
    *,
    thresholds: dg.quality.SessionQualityThresholds = (
        dg.quality.DEFAULT_SESSION_QUALITY_THRESHOLDS
    ),
    late_no_reward_seconds: float = 600.0,
    late_engaged_2_seconds: float = 600.0,
    event_time_column: str = "change_time",
    confidence_level: float = 0.95,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = DEFAULT_MOUSE_COLUMN,
) -> BehaviorTables:
    """Build the complete descriptive behavior vertical slice without inference."""

    sessions = summarize_behavior_sessions(
        trials,
        session_inventory,
        thresholds=thresholds,
        late_no_reward_seconds=late_no_reward_seconds,
        late_engaged_2_seconds=late_engaged_2_seconds,
        event_time_column=event_time_column,
        confidence_level=confidence_level,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    session_blocks = make_session_block_table(
        sessions,
        source_column=source_column,
        mouse_column=mouse_column,
    )
    return BehaviorTables(
        sessions=sessions,
        session_blocks=session_blocks,
        mouse_blocks=aggregate_mouse_blocks(
            session_blocks,
            source_column=source_column,
            mouse_column=mouse_column,
        ),
        mouse_gating=aggregate_mouse_gating(
            sessions,
            source_column=source_column,
            mouse_column=mouse_column,
        ),
        session_attrition=summarize_session_attrition(
            sessions,
            source_column=source_column,
            mouse_column=mouse_column,
        ),
    )


def _require_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    required: tuple[str, ...] | list[str],
    *,
    frame_name: str,
) -> set[str]:
    schema = (
        set(frame.collect_schema().names())
        if isinstance(frame, pl.LazyFrame)
        else set(frame.columns)
    )
    missing = set(required).difference(schema)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")
    return schema


def _collect_projection(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: tuple[str, ...] | list[str],
) -> pl.DataFrame:
    lazy = frame if isinstance(frame, pl.LazyFrame) else frame.lazy()
    return lazy.select(*columns).collect()


def _validate_non_null(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    if frame.get_column(column).null_count():
        raise ValueError(f"{frame_name}.{column} must not contain null values")


def _validate_unique_key(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    if frame.select(*columns).n_unique() != frame.height:
        raise ValueError(f"{frame_name} must be unique by {list(columns)!r}")


def _summarize_author_label_agreement(
    trials: pl.DataFrame,
    *,
    source_column: str,
) -> pl.DataFrame:
    """Audit author outcome labels against canonical lick-derived responses."""

    labeled = dg.quality.label_reward_blocks(trials, source_column=source_column)
    candidate = (
        (pl.col("go").fill_null(False) | pl.col("catch").fill_null(False))
        & ~pl.col("aborted").fill_null(False)
        & ~pl.col("auto_rewarded").fill_null(False)
    )
    hit = pl.col("hit")
    miss = pl.col("miss")
    false_alarm = pl.col("false_alarm")
    correct_reject = pl.col("correct_reject")
    go_outcome_available = hit.is_not_null() & miss.is_not_null() & (hit != miss)
    catch_outcome_available = (
        false_alarm.is_not_null() & correct_reject.is_not_null() & (false_alarm != correct_reject)
    )
    online_outcome_available = (pl.col("go").fill_null(False) & go_outcome_available) | (
        pl.col("catch").fill_null(False) & catch_outcome_available
    )
    online_outcome_absent = (
        pl.col("go").fill_null(False)
        & (hit.is_null() | miss.is_null() | (~hit.fill_null(False) & ~miss.fill_null(False)))
    ) | (
        pl.col("catch").fill_null(False)
        & (
            false_alarm.is_null()
            | correct_reject.is_null()
            | (~false_alarm.fill_null(False) & ~correct_reject.fill_null(False))
        )
    )
    online_outcome_conflict = (
        pl.col("go").fill_null(False) & hit.fill_null(False) & miss.fill_null(False)
    ) | (
        pl.col("catch").fill_null(False)
        & false_alarm.fill_null(False)
        & correct_reject.fill_null(False)
    )
    author_response = (
        pl.when(pl.col("go").fill_null(False) & go_outcome_available)
        .then(hit)
        .when(pl.col("catch").fill_null(False) & catch_outcome_available)
        .then(false_alarm)
        .otherwise(pl.lit(None, dtype=pl.Boolean))
    )
    canonical_response = pl.col("response_in_window")
    comparable = candidate & online_outcome_available & canonical_response.is_not_null()

    expressions = []
    for block in ("engaged_1", "no_reward", "engaged_2"):
        in_block = pl.col(dg.quality.REWARD_BLOCK_COLUMN) == block
        expressions.extend(
            [
                (in_block & candidate).sum().alias(f"n_{block}_response_label_trials"),
                (in_block & candidate & (author_response.is_null() | canonical_response.is_null()))
                .sum()
                .alias(f"n_{block}_response_label_missing"),
                (in_block & candidate & online_outcome_absent)
                .sum()
                .alias(f"n_{block}_online_outcome_absent"),
                (in_block & candidate & online_outcome_conflict)
                .sum()
                .alias(f"n_{block}_online_outcome_conflicts"),
                (in_block & candidate & canonical_response.is_null())
                .sum()
                .alias(f"n_{block}_canonical_response_missing"),
                (in_block & comparable & (author_response != canonical_response))
                .sum()
                .alias(f"n_{block}_response_label_disagreements"),
                (in_block & comparable).sum().alias(f"n_{block}_response_label_comparisons"),
            ]
        )
    summary = labeled.group_by(source_column).agg(*expressions).collect()
    agreement_rates = []
    for block in ("engaged_1", "no_reward", "engaged_2"):
        n_compared = pl.col(f"n_{block}_response_label_comparisons")
        n_disagreements = pl.col(f"n_{block}_response_label_disagreements")
        agreement_rates.append(
            pl.when(n_compared > 0)
            .then(1 - n_disagreements / n_compared)
            .otherwise(None)
            .alias(f"{block}_response_label_agreement_rate")
        )
    return summary.with_columns(*agreement_rates)


def _loglinear_dprime_expr(block: str) -> pl.Expr:
    """Calculate the same finite-sample d-prime used by ``dg.quality``."""

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
    )


def _standard_normal_quantiles(values: pl.Series) -> pl.Series:
    normal = statistics.NormalDist()
    return pl.Series(
        [None if value is None else normal.inv_cdf(float(value)) for value in values],
        dtype=pl.Float64,
    )


def _validate_session_block_identity(
    frame: pl.DataFrame,
    *,
    source_column: str,
    mouse_column: str,
) -> None:
    inconsistent = (
        frame.group_by(source_column)
        .agg(
            pl.col(mouse_column).n_unique().alias("n_mouse_values"),
            pl.col("is_technically_valid").n_unique().alias("n_technical_values"),
            pl.col("is_good_session").n_unique().alias("n_good_session_values"),
        )
        .filter(
            (pl.col("n_mouse_values") != 1)
            | (pl.col("n_technical_values") != 1)
            | (pl.col("n_good_session_values") != 1)
        )
    )
    if inconsistent.height:
        raise ValueError("session_blocks contain inconsistent session identity or cohort flags")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
