"""Validated source-data preparation for the manuscript behavior figure.

The plotting script is intentionally thin.  This module owns the testable
contract that connects behavior summaries, mouse-level inference, and Figure 1.
It never fabricates missing rows or silently drops an incomplete mouse.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import polars as pl

TECHNICALLY_VALID_COHORT = "technically_valid"
SYMMETRIC_SUMMARY_ANALYSIS_ID = "figure_1_behavior"
SYMMETRIC_SUMMARY_RESULT_ID = "reversible_gating_mouse_mean"
PRIMARY_BLOCKS = (
    ("engaged_1", 1),
    ("no_reward_late", 2),
    ("engaged_2_early", 3),
)
ATTRITION_STAGES = (
    ("inventory", 0),
    ("technically_valid", 1),
    ("reversible_gating_estimable", 2),
    ("threshold_selected", 3),
)
MANUSCRIPT_STATISTIC_CONTRACTS = (
    (
        "reversible_gating_mouse_mean",
        "0.5*engaged_1-no_reward_late+0.5*engaged_2_early",
        None,
    ),
    (
        "withdrawal_suppression_mouse_mean",
        "engaged_1-no_reward_late",
        "figure_1_primary_behavior",
    ),
    (
        "restoration_recovery_mouse_mean",
        "engaged_2_early-no_reward_late",
        "figure_1_primary_behavior",
    ),
    (
        "engaged_response_drift_mouse_mean",
        "engaged_2_early-engaged_1",
        "figure_1_response_drift_sensitivity",
    ),
    (
        "withdrawal_specificity_suppression_mouse_mean",
        "engaged_1_specificity-no_reward_late_specificity",
        "figure_1_specificity_behavior",
    ),
    (
        "restoration_specificity_recovery_mouse_mean",
        "engaged_2_early_specificity-no_reward_late_specificity",
        "figure_1_specificity_behavior",
    ),
    (
        "engaged_1_response_probability_mouse_mean",
        "engaged_1_response_probability",
        None,
    ),
    (
        "no_reward_late_response_probability_mouse_mean",
        "no_reward_late_response_probability",
        None,
    ),
    (
        "engaged_2_early_response_probability_mouse_mean",
        "engaged_2_early_response_probability",
        None,
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class BehaviorFigureData:
    """Validated, publication-facing Figure 1 source tables."""

    session_timing: pl.DataFrame
    timing_summary: pl.DataFrame
    trajectories: pl.DataFrame
    gating: pl.DataFrame
    attrition: pl.DataFrame
    gating_summary_statistic: pl.DataFrame


def prepare_behavior_figure_data(
    mouse_blocks: pl.DataFrame,
    mouse_gating: pl.DataFrame,
    session_block_timing: pl.DataFrame,
    session_attrition: pl.DataFrame,
    behavior_statistics: pl.DataFrame,
) -> BehaviorFigureData:
    """Validate and select the exact source rows used in behavior Figure 1.

    Mouse-level panels use only the threshold-independent, technically valid
    cohort.  Available block estimates are retained even when denominators
    differ, but lines connect only mice whose three block denominators match
    the complete-session gating denominator.  Whenever those denominators do
    match, the reported reversible contrast must reconstruct from the plotted
    trajectory. The symmetric gating-summary row must agree with the plotted
    contrast mean and sample size; it is not labeled as a primary inferential
    test.
    """

    for name, frame in (
        ("mouse_blocks", mouse_blocks),
        ("mouse_gating", mouse_gating),
        ("session_block_timing", session_block_timing),
        ("session_attrition", session_attrition),
        ("behavior_statistics", behavior_statistics),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    _require_columns(
        mouse_blocks,
        (
            "subject_id",
            "behavior_block",
            "block_order",
            "response_probability",
            "false_alarm_probability",
            "n_sessions_contributing",
            "n_sessions_false_alarm_contributing",
            "n_sessions_in_cohort",
            "session_cohort",
        ),
        frame_name="mouse_blocks",
    )
    _require_columns(
        mouse_gating,
        (
            "subject_id",
            "reversible_gating_estimate",
            "n_sessions_contributing",
            "session_cohort",
        ),
        frame_name="mouse_gating",
    )
    _require_columns(
        session_block_timing,
        (
            "_nwb_path",
            "subject_id",
            "reward_block",
            "block_order",
            "block_start_from_session_seconds",
            "block_stop_from_session_seconds",
            "block_duration_seconds",
            "n_trials",
            "is_technically_valid",
        ),
        frame_name="session_block_timing",
    )
    _require_columns(
        session_attrition,
        (
            "attrition_stage",
            "stage_order",
            "n_sessions",
            "n_mice",
            "criterion_type",
            "fraction_of_inventory",
        ),
        frame_name="session_attrition",
    )
    _require_columns(
        behavior_statistics,
        (
            "analysis_id",
            "result_id",
            "contrast_id",
            "estimate",
            "ci_low",
            "ci_high",
            "confidence_level",
            "test_method",
            "p_value",
            "adjusted_p_value",
            "adjustment_method",
            "multiplicity_family",
            "sidedness",
            "n_mice",
            "n_sessions",
            "n_trials",
            "status",
        ),
        frame_name="behavior_statistics",
    )

    gating = (
        mouse_gating.filter(
            (pl.col("session_cohort") == TECHNICALLY_VALID_COHORT)
            & (pl.col("n_sessions_contributing") > 0)
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("reversible_gating_estimate").cast(pl.Float64),
            pl.col("n_sessions_contributing").cast(pl.Int64),
            pl.col("session_cohort").cast(pl.String),
        )
        .sort("subject_id")
    )
    if gating.is_empty():
        raise ValueError("mouse_gating has no contributing technically-valid mice")
    _validate_unique(gating, ("subject_id",), frame_name="selected mouse_gating")
    _validate_finite(gating, "reversible_gating_estimate", frame_name="selected mouse_gating")
    _validate_range(
        gating,
        "reversible_gating_estimate",
        lower=-1.0,
        upper=1.0,
        frame_name="selected mouse_gating",
    )

    block_names = [block for block, _ in PRIMARY_BLOCKS]
    trajectories = (
        mouse_blocks.filter(
            (pl.col("session_cohort") == TECHNICALLY_VALID_COHORT)
            & pl.col("behavior_block").is_in(block_names)
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("behavior_block").cast(pl.String),
            pl.col("block_order").cast(pl.Int64),
            pl.col("response_probability").cast(pl.Float64),
            pl.col("false_alarm_probability").cast(pl.Float64),
            pl.col("n_sessions_contributing").cast(pl.Int64),
            pl.col("n_sessions_false_alarm_contributing").cast(pl.Int64),
            pl.col("n_sessions_in_cohort").cast(pl.Int64),
            pl.col("session_cohort").cast(pl.String),
        )
        .filter(pl.col("n_sessions_in_cohort") > 0)
        .sort("subject_id", "block_order")
    )
    _validate_unique(
        trajectories,
        ("subject_id", "behavior_block"),
        frame_name="selected mouse_blocks",
    )
    trajectory_mice = trajectories.get_column("subject_id").n_unique()
    expected_rows = trajectory_mice * len(PRIMARY_BLOCKS)
    if trajectories.height != expected_rows:
        raise ValueError(
            "each technically-valid mouse must have exactly one row for every "
            "primary behavior block"
        )
    missing_gating_mice = gating.select("subject_id").join(
        trajectories.select("subject_id").unique(),
        on="subject_id",
        how="anti",
    )
    if missing_gating_mice.height:
        raise ValueError("a mouse with an estimable gating contrast is absent from mouse_blocks")
    observed_blocks = set(trajectories.select("behavior_block", "block_order").unique().iter_rows())
    if observed_blocks != set(PRIMARY_BLOCKS):
        raise ValueError(
            "primary behavior block names/orders differ from the locked contract: "
            f"{observed_blocks}"
        )
    invalid_response_missingness = trajectories.filter(
        ((pl.col("n_sessions_contributing") > 0) & pl.col("response_probability").is_null())
        | ((pl.col("n_sessions_contributing") == 0) & pl.col("response_probability").is_not_null())
        | (pl.col("n_sessions_contributing") < 0)
    )
    if invalid_response_missingness.height:
        raise ValueError(
            "mouse_blocks response availability disagrees with n_sessions_contributing"
        )
    _validate_optional_finite(
        trajectories,
        "response_probability",
        frame_name="selected mouse_blocks",
    )
    _validate_range(
        trajectories,
        "response_probability",
        lower=0.0,
        upper=1.0,
        frame_name="selected mouse_blocks",
    )
    _validate_optional_probability(
        trajectories,
        "false_alarm_probability",
        frame_name="selected mouse_blocks",
    )
    invalid_catch_missingness = trajectories.filter(
        (
            (pl.col("n_sessions_false_alarm_contributing") > 0)
            & pl.col("false_alarm_probability").is_null()
        )
        | (
            (pl.col("n_sessions_false_alarm_contributing") == 0)
            & pl.col("false_alarm_probability").is_not_null()
        )
        | (pl.col("n_sessions_false_alarm_contributing") < 0)
    )
    if invalid_catch_missingness.height:
        raise ValueError(
            "mouse_blocks catch-response availability disagrees with its session denominator"
        )
    if any(
        trajectories.filter(pl.col("behavior_block") == block)
        .get_column("false_alarm_probability")
        .drop_nulls()
        .is_empty()
        for block, _ in PRIMARY_BLOCKS
    ):
        raise ValueError(
            "each primary block needs at least one mouse-level catch-response estimate"
        )

    contrast_check = gating
    for block, _ in PRIMARY_BLOCKS:
        block_values = trajectories.filter(pl.col("behavior_block") == block).select(
            "subject_id",
            pl.col("response_probability").alias(f"{block}_response_probability"),
            pl.col("n_sessions_contributing").alias(f"{block}_n_sessions"),
        )
        contrast_check = contrast_check.join(
            block_values,
            on="subject_id",
            how="left",
            validate="1:1",
        )
    denominators_align = pl.all_horizontal(
        *(
            pl.col(f"{block}_n_sessions") == pl.col("n_sessions_contributing")
            for block, _ in PRIMARY_BLOCKS
        )
    )
    contrast_check = (
        contrast_check.with_columns(
            denominators_align.fill_null(False).alias("block_denominators_match_gating")
        )
        .with_columns(
            pl.when(pl.col("block_denominators_match_gating"))
            .then(
                0.5 * pl.col("engaged_1_response_probability")
                - pl.col("no_reward_late_response_probability")
                + 0.5 * pl.col("engaged_2_early_response_probability")
            )
            .otherwise(None)
            .alias("reconstructed_gating_estimate")
        )
        .with_columns(
            (pl.col("reversible_gating_estimate") - pl.col("reconstructed_gating_estimate"))
            .abs()
            .alias("absolute_difference")
        )
    )
    aligned_differences = contrast_check.filter(
        pl.col("block_denominators_match_gating")
    ).get_column("absolute_difference")
    if aligned_differences.len() and aligned_differences.max() > 1e-10:
        raise ValueError(
            "mouse_gating estimates do not reconstruct from the plotted block probabilities"
        )
    gating = gating.join(
        contrast_check.select("subject_id", "block_denominators_match_gating"),
        on="subject_id",
        validate="1:1",
    )
    trajectories = trajectories.join(
        gating.select("subject_id", "block_denominators_match_gating"),
        on="subject_id",
        how="left",
        validate="m:1",
    )

    session_timing = (
        session_block_timing.filter(pl.col("is_technically_valid"))
        .select(
            pl.col("_nwb_path").cast(pl.String),
            pl.col("subject_id").cast(pl.String),
            pl.col("reward_block").cast(pl.String),
            pl.col("block_order").cast(pl.Int64),
            pl.col("block_start_from_session_seconds").cast(pl.Float64),
            pl.col("block_stop_from_session_seconds").cast(pl.Float64),
            pl.col("block_duration_seconds").cast(pl.Float64),
            pl.col("n_trials").cast(pl.Int64),
            pl.col("is_technically_valid").cast(pl.Boolean),
        )
        .sort("subject_id", "_nwb_path", "block_order")
    )
    if session_timing.is_empty():
        raise ValueError("session_block_timing has no technically-valid sessions")
    _validate_unique(
        session_timing,
        ("_nwb_path", "reward_block"),
        frame_name="selected session_block_timing",
    )
    timing_blocks = (("engaged_1", 1), ("no_reward", 2), ("engaged_2", 3))
    observed_timing_blocks = set(
        session_timing.select("reward_block", "block_order").unique().iter_rows()
    )
    if observed_timing_blocks != set(timing_blocks):
        raise ValueError(
            f"timing block names/orders differ from the locked contract: {observed_timing_blocks}"
        )
    n_timing_sessions = session_timing.get_column("_nwb_path").n_unique()
    if session_timing.height != n_timing_sessions * len(timing_blocks):
        raise ValueError("every technically-valid session needs one row for each timing block")
    for column in (
        "block_start_from_session_seconds",
        "block_stop_from_session_seconds",
        "block_duration_seconds",
    ):
        _validate_finite(session_timing, column, frame_name="selected session_block_timing")
    invalid_timing = session_timing.filter(
        (pl.col("block_start_from_session_seconds") < 0.0)
        | (pl.col("block_stop_from_session_seconds") <= pl.col("block_start_from_session_seconds"))
        | (pl.col("block_duration_seconds") <= 0.0)
        | (pl.col("n_trials") <= 0)
        | (
            (
                pl.col("block_duration_seconds")
                - (
                    pl.col("block_stop_from_session_seconds")
                    - pl.col("block_start_from_session_seconds")
                )
            ).abs()
            > 1e-6
        )
    )
    if invalid_timing.height:
        raise ValueError(
            "session block timing contains invalid starts, stops, durations, or counts"
        )
    timing_wide = session_timing.pivot(
        on="reward_block",
        index="_nwb_path",
        values=("block_start_from_session_seconds", "block_stop_from_session_seconds"),
    )
    invalid_order = timing_wide.filter(
        (
            pl.col("block_stop_from_session_seconds_engaged_1")
            > pl.col("block_start_from_session_seconds_no_reward")
        )
        | (
            pl.col("block_stop_from_session_seconds_no_reward")
            > pl.col("block_start_from_session_seconds_engaged_2")
        )
    )
    if invalid_order.height:
        raise ValueError("session block timing is overlapping or out of E1/NR/E2 order")
    timing_summary = (
        session_timing.group_by("reward_block", "block_order")
        .agg(
            pl.col("block_start_from_session_seconds").median().alias("median_start_seconds"),
            pl.col("block_stop_from_session_seconds").median().alias("median_stop_seconds"),
            pl.col("block_duration_seconds").median().alias("median_duration_seconds"),
            pl.col("block_start_from_session_seconds")
            .quantile(0.25, interpolation="linear")
            .alias("start_q25_seconds"),
            pl.col("block_start_from_session_seconds")
            .quantile(0.75, interpolation="linear")
            .alias("start_q75_seconds"),
            pl.col("block_stop_from_session_seconds")
            .quantile(0.25, interpolation="linear")
            .alias("stop_q25_seconds"),
            pl.col("block_stop_from_session_seconds")
            .quantile(0.75, interpolation="linear")
            .alias("stop_q75_seconds"),
            pl.col("_nwb_path").n_unique().cast(pl.Int64).alias("n_sessions"),
            pl.col("subject_id").n_unique().cast(pl.Int64).alias("n_mice"),
            pl.col("n_trials").sum().cast(pl.Int64).alias("n_trials"),
        )
        .sort("block_order")
    )

    attrition = session_attrition.select(
        pl.col("attrition_stage").cast(pl.String),
        pl.col("stage_order").cast(pl.Int64),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_mice").cast(pl.Int64),
        pl.col("criterion_type").cast(pl.String),
        pl.col("fraction_of_inventory").cast(pl.Float64),
    ).sort("stage_order")
    _validate_unique(attrition, ("attrition_stage",), frame_name="session_attrition")
    observed_stages = set(attrition.select("attrition_stage", "stage_order").iter_rows())
    if observed_stages != set(ATTRITION_STAGES) or attrition.height != len(ATTRITION_STAGES):
        raise ValueError(f"attrition stages differ from the locked contract: {observed_stages}")
    if attrition.filter((pl.col("n_sessions") < 0) | (pl.col("n_mice") < 0)).height:
        raise ValueError("attrition counts must be non-negative")
    session_counts = attrition.get_column("n_sessions").to_list()
    mouse_counts = attrition.get_column("n_mice").to_list()
    if any(
        later > earlier for earlier, later in zip(session_counts, session_counts[1:], strict=False)
    ):
        raise ValueError("session attrition counts must be non-increasing")
    if any(later > earlier for earlier, later in zip(mouse_counts, mouse_counts[1:], strict=False)):
        raise ValueError("mouse attrition counts must be non-increasing")
    inventory_sessions = session_counts[0]
    if inventory_sessions <= 0:
        raise ValueError("session attrition inventory must be non-empty")
    expected_fractions = np.asarray(session_counts, dtype=float) / inventory_sessions
    observed_fractions = attrition.get_column("fraction_of_inventory").to_numpy()
    if not np.all(np.isfinite(observed_fractions)) or not np.allclose(
        observed_fractions,
        expected_fractions,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("fraction_of_inventory does not agree with session counts")
    technical_attrition = attrition.filter(pl.col("attrition_stage") == "technically_valid")
    if (
        technical_attrition.get_column("n_sessions").item() != n_timing_sessions
        or technical_attrition.get_column("n_mice").item()
        != session_timing.get_column("subject_id").n_unique()
    ):
        raise ValueError("technically-valid timing sessions/mice disagree with the attrition table")

    reporting_statistics = _validate_manuscript_statistics(behavior_statistics)
    gating_summary_statistic = reporting_statistics.filter(
        (pl.col("analysis_id") == SYMMETRIC_SUMMARY_ANALYSIS_ID)
        & (pl.col("result_id") == SYMMETRIC_SUMMARY_RESULT_ID)
    )
    if gating_summary_statistic.height != 1:
        raise ValueError(
            "behavior_statistics must contain exactly one figure_1_behavior / "
            "reversible_gating_mouse_mean symmetric-summary row"
        )
    statistic = gating_summary_statistic.row(0, named=True)
    for column in ("estimate", "ci_low", "ci_high", "confidence_level"):
        value = statistic[column]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"symmetric gating summary requires finite {column}")
    if not 0.0 < float(statistic["confidence_level"]) < 1.0:
        raise ValueError("symmetric gating summary confidence_level must be between zero and one")
    if float(statistic["ci_low"]) > float(statistic["ci_high"]):
        raise ValueError("symmetric gating summary confidence interval is reversed")
    if statistic["n_mice"] is None or int(statistic["n_mice"]) != gating.height:
        raise ValueError("symmetric gating summary n_mice disagrees with plotted mice")
    mouse_mean = gating.get_column("reversible_gating_estimate").mean()
    if not math.isclose(float(statistic["estimate"]), mouse_mean, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("symmetric gating summary estimate disagrees with plotted mouse mean")

    return BehaviorFigureData(
        session_timing=session_timing,
        timing_summary=timing_summary,
        trajectories=trajectories,
        gating=gating,
        attrition=attrition,
        gating_summary_statistic=gating_summary_statistic,
    )


def _validate_manuscript_statistics(behavior_statistics: pl.DataFrame) -> pl.DataFrame:
    """Authenticate the exact Figure 1 rows cited by the preliminary manuscript."""

    expected = pl.DataFrame(
        {
            "result_id": [row[0] for row in MANUSCRIPT_STATISTIC_CONTRACTS],
            "expected_contrast_id": [row[1] for row in MANUSCRIPT_STATISTIC_CONTRACTS],
            "expected_multiplicity_family": [row[2] for row in MANUSCRIPT_STATISTIC_CONTRACTS],
        },
        schema={
            "result_id": pl.String,
            "expected_contrast_id": pl.String,
            "expected_multiplicity_family": pl.String,
        },
    )
    selected = behavior_statistics.filter(
        (pl.col("analysis_id") == SYMMETRIC_SUMMARY_ANALYSIS_ID)
        & pl.col("result_id").is_in(expected.get_column("result_id").to_list())
    )
    if selected.height != expected.height:
        raise ValueError(
            "behavior_statistics must contain exactly one row for every Figure 1 "
            "manuscript statistic"
        )
    _validate_unique(selected, ("result_id",), frame_name="Figure 1 manuscript statistics")
    selected = selected.join(expected, on="result_id", validate="1:1")
    if selected.filter(
        (pl.col("contrast_id") != pl.col("expected_contrast_id"))
        | (pl.col("status") != "pass")
    ).height:
        raise ValueError(
            "Figure 1 manuscript statistics have unexpected contrast identities or status"
        )

    for column in ("estimate", "ci_low", "ci_high", "confidence_level"):
        _validate_finite(selected, column, frame_name="Figure 1 manuscript statistics")
    if selected.filter(
        (pl.col("confidence_level") <= 0.0)
        | (pl.col("confidence_level") >= 1.0)
        | (pl.col("ci_low") > pl.col("estimate"))
        | (pl.col("ci_high") < pl.col("estimate"))
        | (pl.col("n_mice") <= 0)
        | (pl.col("n_sessions") <= 0)
        | (pl.col("n_trials") <= 0)
    ).height:
        raise ValueError("Figure 1 manuscript statistics have invalid intervals or denominators")

    inferential = selected.filter(pl.col("expected_multiplicity_family").is_not_null())
    invalid_inference = (
        pl.col("p_value").is_null()
        | ~pl.col("p_value").is_finite()
        | (pl.col("p_value") < 0.0)
        | (pl.col("p_value") > 1.0)
        | pl.col("adjusted_p_value").is_null()
        | ~pl.col("adjusted_p_value").is_finite()
        | (pl.col("adjusted_p_value") < pl.col("p_value"))
        | (pl.col("adjusted_p_value") > 1.0)
        | (pl.col("adjustment_method") != "holm")
        | (pl.col("multiplicity_family") != pl.col("expected_multiplicity_family"))
        | (pl.col("sidedness") != "two-sided")
        | ~pl.col("test_method").str.contains("two_sided_sign_flip")
    ).fill_null(True)
    if inferential.filter(invalid_inference).height:
        raise ValueError("Figure 1 inferential statistics violate the locked test families")

    descriptive = selected.filter(pl.col("expected_multiplicity_family").is_null())
    if descriptive.filter(
        pl.any_horizontal(
            pl.col("p_value").is_not_null(),
            pl.col("adjusted_p_value").is_not_null(),
            pl.col("adjustment_method").is_not_null(),
            pl.col("multiplicity_family").is_not_null(),
            pl.col("sidedness").is_not_null(),
        )
    ).height:
        raise ValueError("Figure 1 descriptive statistics must remain non-inferential")
    return selected.drop("expected_contrast_id", "expected_multiplicity_family")


def deterministic_strip_offsets(n_points: int, *, width: float = 0.24) -> np.ndarray:
    """Return centered, outcome-independent offsets for a strip plot."""

    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    if not math.isfinite(width) or width < 0:
        raise ValueError("width must be finite and non-negative")
    if n_points == 1:
        return np.asarray([0.0])
    evenly_spaced = np.linspace(-width / 2.0, width / 2.0, n_points)
    permutation = np.random.default_rng(1051).permutation(n_points)
    return evenly_spaced[permutation]


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
        raise ValueError(f"{frame_name} must be unique by {list(columns)!r}")


def _validate_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    if frame.filter(pl.col(column).is_null() | ~pl.col(column).is_finite()).height:
        raise ValueError(f"{frame_name}.{column} must contain only finite values")


def _validate_optional_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    if frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite()).height:
        raise ValueError(f"{frame_name}.{column} must contain finite values or nulls")


def _validate_range(
    frame: pl.DataFrame,
    column: str,
    *,
    lower: float,
    upper: float,
    frame_name: str,
) -> None:
    if frame.filter((pl.col(column) < lower) | (pl.col(column) > upper)).height:
        raise ValueError(f"{frame_name}.{column} must lie in [{lower}, {upper}]")


def _validate_optional_probability(
    frame: pl.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> None:
    invalid = frame.filter(
        pl.col(column).is_not_null()
        & (~pl.col(column).is_finite() | (pl.col(column) < 0.0) | (pl.col(column) > 1.0))
    )
    if invalid.height:
        raise ValueError(f"{frame_name}.{column} must contain probabilities or null values")
