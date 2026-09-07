"""Tests for the validated Figure 1 source-data contract."""

import numpy as np
import polars as pl
import pytest

import dg.figure_behavior


def _mouse_blocks() -> pl.DataFrame:
    rows = []
    values = {
        "m1": ((0.8, 0.1), (0.2, 0.1), (0.7, 0.1)),
        "m2": ((0.6, 0.2), (0.1, 0.2), (0.9, 0.2)),
    }
    for subject_id, block_values in values.items():
        for (behavior_block, block_order), (response, false_alarm) in zip(
            dg.figure_behavior.PRIMARY_BLOCKS,
            block_values,
            strict=True,
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "behavior_block": behavior_block,
                    "block_order": block_order,
                    "response_probability": response,
                    "false_alarm_probability": false_alarm,
                    "n_sessions_contributing": 2,
                    "n_sessions_false_alarm_contributing": 2,
                    "n_sessions_in_cohort": 2,
                    "session_cohort": "technically_valid",
                }
            )
    rows.append(
        {
            "subject_id": "m1",
            "behavior_block": "engaged_1",
            "block_order": 1,
            "response_probability": 1.0,
            "false_alarm_probability": 0.0,
            "n_sessions_contributing": 1,
            "n_sessions_false_alarm_contributing": 1,
            "n_sessions_in_cohort": 1,
            "session_cohort": "threshold_selected",
        }
    )
    return pl.DataFrame(rows)


def _mouse_gating() -> pl.DataFrame:
    # m1: .5*.8 - .2 + .5*.7 = .55; m2: .5*.6 - .1 + .5*.9 = .65
    return pl.DataFrame(
        {
            "subject_id": ["m1", "m2", "m3"],
            "reversible_gating_estimate": [0.55, 0.65, None],
            "n_sessions_contributing": [2, 2, 0],
            "session_cohort": ["technically_valid"] * 3,
        }
    )


def _attrition() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "attrition_stage": [stage for stage, _ in dg.figure_behavior.ATTRITION_STAGES],
            "stage_order": [order for _, order in dg.figure_behavior.ATTRITION_STAGES],
            "n_sessions": [3, 2, 2, 1],
            "n_mice": [2, 2, 2, 1],
            "criterion_type": [
                "inventory",
                "technical",
                "estimability",
                "behavior_threshold",
            ],
            "fraction_of_inventory": [1.0, 2 / 3, 2 / 3, 1 / 3],
        }
    )


def _session_block_timing() -> pl.DataFrame:
    rows = []
    for source, subject, offset in (("session-1", "m1", 0.0), ("session-2", "m2", 60.0)):
        for reward_block, block_order, start, stop in (
            ("engaged_1", 1, 0.0, 600.0),
            ("no_reward", 2, 600.0, 1200.0),
            ("engaged_2", 3, 1200.0, 1800.0),
        ):
            rows.append(
                {
                    "_nwb_path": source,
                    "subject_id": subject,
                    "reward_block": reward_block,
                    "block_order": block_order,
                    "block_start_from_session_seconds": start + offset,
                    "block_stop_from_session_seconds": stop + offset,
                    "block_duration_seconds": stop - start,
                    "n_trials": 50,
                    "is_technically_valid": True,
                }
            )
    return pl.DataFrame(rows)


def _statistics() -> pl.DataFrame:
    estimates = {
        "reversible_gating_mouse_mean": 0.6,
        "withdrawal_suppression_mouse_mean": 0.55,
        "restoration_recovery_mouse_mean": 0.65,
        "engaged_response_drift_mouse_mean": 0.1,
        "withdrawal_specificity_suppression_mouse_mean": 0.55,
        "restoration_specificity_recovery_mouse_mean": 0.65,
        "engaged_1_response_probability_mouse_mean": 0.7,
        "no_reward_late_response_probability_mouse_mean": 0.15,
        "engaged_2_early_response_probability_mouse_mean": 0.8,
    }
    rows = []
    for result_id, contrast_id, family in dg.figure_behavior.MANUSCRIPT_STATISTIC_CONTRACTS:
        estimate = estimates[result_id]
        inferential = family is not None
        rows.append(
            {
                "analysis_id": "figure_1_behavior",
                "result_id": result_id,
                "contrast_id": contrast_id,
                "estimate": estimate,
                "ci_low": estimate - 0.1,
                "ci_high": estimate + 0.1,
                "confidence_level": 0.95,
                "test_method": (
                    "monte_carlo_two_sided_sign_flip;n_permutations=100000"
                    if inferential
                    else None
                ),
                "p_value": 0.01 if inferential else None,
                "adjusted_p_value": 0.02 if inferential else None,
                "adjustment_method": "holm" if inferential else None,
                "multiplicity_family": family,
                "sidedness": "two-sided" if inferential else None,
                "n_mice": 2,
                "n_sessions": 4,
                "n_trials": 100,
                "status": "pass",
            }
        )
    return pl.DataFrame(rows)


def test_prepare_behavior_figure_data_filters_and_cross_checks_inputs() -> None:
    prepared = dg.figure_behavior.prepare_behavior_figure_data(
        _mouse_blocks(),
        _mouse_gating(),
        _session_block_timing(),
        _attrition(),
        _statistics(),
    )

    assert prepared.trajectories.height == 6
    assert prepared.trajectories.get_column("session_cohort").unique().to_list() == [
        "technically_valid"
    ]
    assert prepared.timing_summary.select(
        "reward_block", "median_start_seconds", "median_stop_seconds"
    ).to_dicts() == [
        {
            "reward_block": "engaged_1",
            "median_start_seconds": 30.0,
            "median_stop_seconds": 630.0,
        },
        {
            "reward_block": "no_reward",
            "median_start_seconds": 630.0,
            "median_stop_seconds": 1230.0,
        },
        {
            "reward_block": "engaged_2",
            "median_start_seconds": 1230.0,
            "median_stop_seconds": 1830.0,
        },
    ]
    assert prepared.gating.get_column("subject_id").to_list() == ["m1", "m2"]
    assert prepared.gating_summary_statistic.get_column("estimate").item() == pytest.approx(0.6)


def test_prepare_behavior_figure_data_rejects_statistic_disagreement() -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("result_id") == "reversible_gating_mouse_mean")
        .then(0.7)
        .otherwise(pl.col("estimate"))
        .alias("estimate")
    )

    with pytest.raises(ValueError, match="estimate disagrees"):
        dg.figure_behavior.prepare_behavior_figure_data(
            _mouse_blocks(),
            _mouse_gating(),
            _session_block_timing(),
            _attrition(),
            statistics,
        )


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("contrast_id", "wrong_contrast", "unexpected contrast identities"),
        ("status", "failed", "unexpected contrast identities"),
        ("multiplicity_family", "wrong_family", "locked test families"),
        ("adjustment_method", None, "locked test families"),
    ],
)
def test_prepare_behavior_figure_data_rejects_invalid_manuscript_statistic_contract(
    column: str,
    invalid_value: str | None,
    message: str,
) -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("result_id") == "withdrawal_suppression_mouse_mean")
        .then(pl.lit(invalid_value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match=message):
        dg.figure_behavior.prepare_behavior_figure_data(
            _mouse_blocks(),
            _mouse_gating(),
            _session_block_timing(),
            _attrition(),
            statistics,
        )


def test_prepare_behavior_figure_data_rejects_incomplete_mouse_trajectory() -> None:
    mouse_blocks = _mouse_blocks().filter(
        ~(
            (pl.col("subject_id") == "m2")
            & (pl.col("behavior_block") == "no_reward_late")
            & (pl.col("session_cohort") == "technically_valid")
        )
    )

    with pytest.raises(ValueError, match="every primary behavior block"):
        dg.figure_behavior.prepare_behavior_figure_data(
            mouse_blocks,
            _mouse_gating(),
            _session_block_timing(),
            _attrition(),
            _statistics(),
        )


def test_prepare_behavior_figure_data_records_legitimate_denominator_mismatch() -> None:
    mouse_blocks = _mouse_blocks().with_columns(
        pl.when(
            (pl.col("subject_id") == "m2")
            & (pl.col("behavior_block") == "no_reward_late")
            & (pl.col("session_cohort") == "technically_valid")
        )
        .then(1)
        .otherwise(pl.col("n_sessions_contributing"))
        .alias("n_sessions_contributing"),
        pl.when(
            (pl.col("subject_id") == "m2")
            & (pl.col("behavior_block") == "no_reward_late")
            & (pl.col("session_cohort") == "technically_valid")
        )
        .then(0.4)
        .otherwise(pl.col("response_probability"))
        .alias("response_probability"),
    )

    prepared = dg.figure_behavior.prepare_behavior_figure_data(
        mouse_blocks,
        _mouse_gating(),
        _session_block_timing(),
        _attrition(),
        _statistics(),
    )

    assert prepared.gating.select("subject_id", "block_denominators_match_gating").to_dicts() == [
        {"subject_id": "m1", "block_denominators_match_gating": True},
        {"subject_id": "m2", "block_denominators_match_gating": False},
    ]


def test_deterministic_strip_offsets_are_centered_and_reproducible() -> None:
    first = dg.figure_behavior.deterministic_strip_offsets(5, width=0.2)
    second = dg.figure_behavior.deterministic_strip_offsets(5, width=0.2)

    np.testing.assert_array_equal(first, second)
    assert first.mean() == pytest.approx(0.0)
    np.testing.assert_allclose(np.sort(first), np.linspace(-0.1, 0.1, 5))
    assert not np.all(np.diff(first) >= 0)


@pytest.mark.parametrize("n_points", [0, -1, True, 1.5])
def test_deterministic_strip_offsets_rejects_invalid_counts(n_points) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dg.figure_behavior.deterministic_strip_offsets(n_points)
