"""Validated source-data preparation for the behavior-transition QC figure."""

from __future__ import annotations

import dataclasses
import math

import polars as pl

TRANSITIONS = (
    ("reward_withdrawal", 1, "Reward withdrawal"),
    ("reward_restoration", 2, "Reward restoration"),
)
CONDITIONS = (
    ("go", 1, "Go response"),
    ("go_minus_catch", 2, "Go − catch"),
)
MULTIPLICITY_FAMILIES = {
    "go": "behavior_transition_primary_go",
    "go_minus_catch": "behavior_transition_primary_specificity",
}
TRAJECTORY_CONDITIONS = ("go", "catch")
EXPECTED_TIME_BINS = (-3, -2, -1, 0, 1, 2)
ANALYSIS_ID = "behavior_transition_qc"


@dataclasses.dataclass(frozen=True, slots=True)
class BehaviorTransitionFigureData:
    """Publication-facing rows used by the behavior-transition QC figure."""

    trajectories: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistics: pl.DataFrame


def prepare_behavior_transition_figure_data(
    time_trajectories: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    statistics: pl.DataFrame,
) -> BehaviorTransitionFigureData:
    """Validate and select real-boundary trajectories and controlled effects."""

    for name, frame in (
        ("time_trajectories", time_trajectories),
        ("mouse_effects", mouse_effects),
        ("statistics", statistics),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    _require_columns(
        time_trajectories,
        (
            "transition_id",
            "transition_order",
            "anchor_type",
            "condition",
            "bin_index",
            "analysis_block",
            "relative_bin_start_seconds",
            "relative_bin_stop_seconds",
            "relative_bin_center_seconds",
            "n_mice_response_contributing",
            "n_sessions_response_contributing",
            "n_trials_contributing",
            "response_probability",
            "response_probability_ci_low",
            "response_probability_ci_high",
            "aggregation",
        ),
        frame_name="time_trajectories",
    )
    _require_columns(
        mouse_effects,
        (
            "subject_id",
            "transition_id",
            "transition_order",
            "condition",
            "pseudo_controlled_step",
            "n_sessions_pseudo_controlled_step",
            "n_controlled_window_trials",
            "within_mouse_aggregation",
        ),
        frame_name="mouse_effects",
    )
    _require_columns(
        statistics,
        (
            "analysis_id",
            "result_id",
            "contrast_id",
            "estimate",
            "ci_low",
            "ci_high",
            "confidence_level",
            "p_value",
            "adjusted_p_value",
            "adjustment_method",
            "multiplicity_family",
            "n_mice",
            "n_sessions",
            "aggregation",
            "status",
            "reason",
        ),
        frame_name="statistics",
    )

    transition_lookup = pl.DataFrame(
        {
            "transition_id": [row[0] for row in TRANSITIONS],
            "expected_transition_order": [row[1] for row in TRANSITIONS],
            "transition_label": [row[2] for row in TRANSITIONS],
        }
    )
    condition_lookup = pl.DataFrame(
        {
            "condition": [row[0] for row in CONDITIONS],
            "condition_order": [row[1] for row in CONDITIONS],
            "condition_label": [row[2] for row in CONDITIONS],
        }
    )

    trajectories = (
        time_trajectories.filter(
            (pl.col("anchor_type") == "real")
            & pl.col("transition_id").is_in([row[0] for row in TRANSITIONS])
            & pl.col("condition").is_in(TRAJECTORY_CONDITIONS)
        )
        .select(
            pl.col("transition_id").cast(pl.String),
            pl.col("transition_order").cast(pl.Int64),
            pl.col("anchor_type").cast(pl.String),
            pl.col("condition").cast(pl.String),
            pl.col("bin_index").cast(pl.Int64),
            pl.col("analysis_block").cast(pl.String),
            pl.col("relative_bin_start_seconds").cast(pl.Float64),
            pl.col("relative_bin_stop_seconds").cast(pl.Float64),
            pl.col("relative_bin_center_seconds").cast(pl.Float64),
            pl.col("n_mice_response_contributing").cast(pl.Int64),
            pl.col("n_sessions_response_contributing").cast(pl.Int64),
            pl.col("n_trials_contributing").cast(pl.Int64),
            pl.col("response_probability").cast(pl.Float64),
            pl.col("response_probability_ci_low").cast(pl.Float64),
            pl.col("response_probability_ci_high").cast(pl.Float64),
            pl.col("aggregation").cast(pl.String),
        )
        .join(transition_lookup, on="transition_id", validate="m:1")
        .sort("transition_order", "condition", "bin_index")
    )
    expected_trajectory_rows = (
        len(TRANSITIONS) * len(TRAJECTORY_CONDITIONS) * len(EXPECTED_TIME_BINS)
    )
    if trajectories.height != expected_trajectory_rows:
        raise ValueError("real transition trajectories do not have the expected complete row set")
    _validate_unique(
        trajectories,
        ("transition_id", "condition", "bin_index"),
        frame_name="selected time trajectories",
    )
    if trajectories.filter(
        pl.col("transition_order") != pl.col("expected_transition_order")
    ).height:
        raise ValueError("transition order differs from the locked behavior-transition contract")
    for transition_id, _, _ in TRANSITIONS:
        for condition in TRAJECTORY_CONDITIONS:
            observed_bins = (
                trajectories.filter(
                    (pl.col("transition_id") == transition_id) & (pl.col("condition") == condition)
                )
                .get_column("bin_index")
                .to_list()
            )
            if tuple(observed_bins) != EXPECTED_TIME_BINS:
                raise ValueError(
                    f"{transition_id}/{condition} has unexpected or unordered time bins"
                )
    if trajectories.filter(
        (pl.col("relative_bin_start_seconds") >= pl.col("relative_bin_stop_seconds"))
        | (
            pl.col("relative_bin_center_seconds")
            != (pl.col("relative_bin_start_seconds") + pl.col("relative_bin_stop_seconds")) / 2.0
        )
        | (pl.col("n_mice_response_contributing") <= 0)
        | (pl.col("n_sessions_response_contributing") <= 0)
        | (pl.col("n_trials_contributing") <= 0)
    ).height:
        raise ValueError("time trajectories contain invalid bins or denominators")
    for column in (
        "response_probability",
        "response_probability_ci_low",
        "response_probability_ci_high",
    ):
        _validate_finite(trajectories, column, frame_name="selected time trajectories")
    if trajectories.filter(
        (pl.col("response_probability_ci_low") < 0.0)
        | (pl.col("response_probability_ci_high") > 1.0)
        | (pl.col("response_probability") < pl.col("response_probability_ci_low"))
        | (pl.col("response_probability") > pl.col("response_probability_ci_high"))
    ).height:
        raise ValueError("trajectory probabilities or confidence intervals are invalid")

    expected_contrasts = {
        f"{transition_id}_{condition}_pseudo_controlled_step"
        for transition_id, _, _ in TRANSITIONS
        for condition, _, _ in CONDITIONS
    }
    selected_statistics = (
        statistics.filter(pl.col("contrast_id").is_in(sorted(expected_contrasts)))
        .select(
            "analysis_id",
            "result_id",
            "contrast_id",
            pl.col("estimate").cast(pl.Float64),
            pl.col("ci_low").cast(pl.Float64),
            pl.col("ci_high").cast(pl.Float64),
            pl.col("confidence_level").cast(pl.Float64),
            pl.col("p_value").cast(pl.Float64),
            pl.col("adjusted_p_value").cast(pl.Float64),
            "adjustment_method",
            "multiplicity_family",
            pl.col("n_mice").cast(pl.Int64),
            pl.col("n_sessions").cast(pl.Int64),
            "aggregation",
            "status",
            "reason",
        )
        .with_columns(
            pl.col("contrast_id")
            .str.replace("_pseudo_controlled_step$", "")
            .alias("transition_condition")
        )
    )
    if selected_statistics.height != len(expected_contrasts):
        raise ValueError("statistics lack exactly one row for each primary controlled contrast")
    _validate_unique(selected_statistics, ("contrast_id",), frame_name="selected statistics")
    if set(selected_statistics.get_column("contrast_id")) != expected_contrasts:
        raise ValueError("selected controlled contrasts differ from the expected family")
    invalid_identity = selected_statistics.filter(
        (pl.col("analysis_id") != ANALYSIS_ID)
        | (pl.col("result_id") != pl.concat_str("contrast_id", pl.lit("_mouse_mean")))
    )
    if invalid_identity.height:
        raise ValueError(
            "primary behavior-transition statistics have unexpected analysis/result identities"
        )
    if selected_statistics.filter(pl.col("status") != "pass").height:
        raise ValueError("a primary behavior-transition contrast did not pass")
    expected_family_by_contrast = {
        f"{transition_id}_{condition}_pseudo_controlled_step": MULTIPLICITY_FAMILIES[condition]
        for transition_id, _, _ in TRANSITIONS
        for condition, _, _ in CONDITIONS
    }
    invalid_adjustment = selected_statistics.filter(
        (
            (pl.col("adjustment_method") != "holm")
            | (
                pl.col("multiplicity_family")
                != pl.col("contrast_id").replace_strict(expected_family_by_contrast)
            )
        ).fill_null(True)
    )
    if invalid_adjustment.height:
        raise ValueError(
            "primary behavior-transition contrasts do not use the exact locked Holm families"
        )
    for column in ("estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"):
        _validate_finite(selected_statistics, column, frame_name="selected statistics")
    if selected_statistics.filter(
        (pl.col("ci_low") > pl.col("estimate"))
        | (pl.col("ci_high") < pl.col("estimate"))
        | (pl.col("n_mice") <= 0)
        | (pl.col("n_sessions") <= 0)
    ).height:
        raise ValueError("controlled statistics contain invalid intervals or denominators")

    selected_mouse_effects = (
        mouse_effects.filter(
            pl.col("transition_id").is_in([row[0] for row in TRANSITIONS])
            & pl.col("condition").is_in([row[0] for row in CONDITIONS])
            & pl.col("pseudo_controlled_step").is_not_null()
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("transition_id").cast(pl.String),
            pl.col("transition_order").cast(pl.Int64),
            pl.col("condition").cast(pl.String),
            pl.col("pseudo_controlled_step").cast(pl.Float64),
            pl.col("n_sessions_pseudo_controlled_step").cast(pl.Int64),
            pl.col("n_controlled_window_trials").cast(pl.Int64),
            pl.col("within_mouse_aggregation").cast(pl.String),
        )
        .join(transition_lookup, on="transition_id", validate="m:1")
        .join(condition_lookup, on="condition", validate="m:1")
        .sort("transition_order", "condition_order", "subject_id")
    )
    if selected_mouse_effects.is_empty():
        raise ValueError("mouse effects contain no estimable controlled contrasts")
    _validate_unique(
        selected_mouse_effects,
        ("subject_id", "transition_id", "condition"),
        frame_name="selected mouse effects",
    )
    _validate_finite(
        selected_mouse_effects,
        "pseudo_controlled_step",
        frame_name="selected mouse effects",
    )
    if selected_mouse_effects.filter(
        (pl.col("transition_order") != pl.col("expected_transition_order"))
        | (pl.col("n_sessions_pseudo_controlled_step") <= 0)
        | (pl.col("n_controlled_window_trials") <= 0)
    ).height:
        raise ValueError("mouse effects contain invalid orders or denominators")

    statistic_keys = []
    for transition_id, transition_order, transition_label in TRANSITIONS:
        for condition, condition_order, condition_label in CONDITIONS:
            statistic_keys.append(
                {
                    "transition_id": transition_id,
                    "transition_order": transition_order,
                    "transition_label": transition_label,
                    "condition": condition,
                    "condition_order": condition_order,
                    "condition_label": condition_label,
                    "contrast_id": f"{transition_id}_{condition}_pseudo_controlled_step",
                }
            )
    selected_statistics = (
        pl.DataFrame(statistic_keys)
        .join(selected_statistics.drop("transition_condition"), on="contrast_id", validate="1:1")
        .sort("transition_order", "condition_order")
    )
    checks = (
        selected_mouse_effects.group_by("transition_id", "condition")
        .agg(
            pl.len().alias("observed_n_mice"),
            pl.col("pseudo_controlled_step").mean().alias("observed_mouse_mean"),
        )
        .join(
            selected_statistics.select("transition_id", "condition", "n_mice", "estimate"),
            on=("transition_id", "condition"),
            validate="1:1",
        )
    )
    if checks.filter(pl.col("observed_n_mice") != pl.col("n_mice")).height:
        raise ValueError("mouse-effect counts do not agree with the statistics table")
    differences = checks.select(
        (pl.col("observed_mouse_mean") - pl.col("estimate")).abs().alias("difference")
    ).get_column("difference")
    if differences.max() > 1e-10:
        raise ValueError("mouse-effect means do not reconstruct the reported estimates")

    return BehaviorTransitionFigureData(
        trajectories=trajectories.drop("expected_transition_order"),
        mouse_effects=selected_mouse_effects.drop("expected_transition_order"),
        statistics=selected_statistics,
    )


def _require_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} lacks columns: {missing}")


def _validate_unique(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    if frame.select(columns).n_unique() != frame.height:
        raise ValueError(f"{frame_name} has duplicate keys for {columns}")


def _validate_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    values = frame.get_column(column)
    if values.null_count() or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{frame_name}.{column} must be finite and non-null")
