"""Tests for the supplementary behavior-transition figure data contract."""

import polars as pl
import pytest

import dg.figure_behavior_transitions


def _time_trajectories() -> pl.DataFrame:
    rows = []
    for transition_id, transition_order, _ in dg.figure_behavior_transitions.TRANSITIONS:
        for anchor_type in ("real", "pseudo"):
            for condition in ("go", "catch"):
                for bin_index in dg.figure_behavior_transitions.EXPECTED_TIME_BINS:
                    start = float(bin_index * 120)
                    probability = 0.6 if condition == "go" else 0.1
                    rows.append(
                        {
                            "transition_id": transition_id,
                            "transition_order": transition_order,
                            "anchor_type": anchor_type,
                            "condition": condition,
                            "bin_index": bin_index,
                            "analysis_block": "pre" if bin_index < 0 else "post",
                            "relative_bin_start_seconds": start,
                            "relative_bin_stop_seconds": start + 120.0,
                            "relative_bin_center_seconds": start + 60.0,
                            "n_mice_response_contributing": 2,
                            "n_sessions_response_contributing": 3,
                            "n_trials_contributing": 20,
                            "response_probability": probability,
                            "response_probability_ci_low": probability - 0.05,
                            "response_probability_ci_high": probability + 0.05,
                            "aggregation": "equal_session_within_mouse_then_equal_mouse",
                        }
                    )
    return pl.DataFrame(rows)


def _mouse_effects() -> pl.DataFrame:
    rows = []
    values = {
        ("reward_withdrawal", "go"): (0.1, 0.3),
        ("reward_withdrawal", "go_minus_catch"): (0.0, 0.2),
        ("reward_restoration", "go"): (0.4, 0.6),
        ("reward_restoration", "go_minus_catch"): (0.3, 0.5),
    }
    for transition_id, transition_order, _ in dg.figure_behavior_transitions.TRANSITIONS:
        for condition, _, _ in dg.figure_behavior_transitions.CONDITIONS:
            for subject_id, value in zip(
                ("m1", "m2"), values[(transition_id, condition)], strict=True
            ):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "transition_id": transition_id,
                        "transition_order": transition_order,
                        "condition": condition,
                        "pseudo_controlled_step": value,
                        "n_sessions_pseudo_controlled_step": 2,
                        "n_controlled_window_trials": 30,
                        "within_mouse_aggregation": "equal_session_mean",
                    }
                )
    return pl.DataFrame(rows)


def _statistics() -> pl.DataFrame:
    estimates = {
        ("reward_withdrawal", "go"): 0.2,
        ("reward_withdrawal", "go_minus_catch"): 0.1,
        ("reward_restoration", "go"): 0.5,
        ("reward_restoration", "go_minus_catch"): 0.4,
    }
    rows = []
    for transition_id, _, _ in dg.figure_behavior_transitions.TRANSITIONS:
        for condition, _, _ in dg.figure_behavior_transitions.CONDITIONS:
            estimate = estimates[(transition_id, condition)]
            contrast_id = f"{transition_id}_{condition}_pseudo_controlled_step"
            rows.append(
                {
                    "analysis_id": "behavior_transition_qc",
                    "result_id": f"{contrast_id}_mouse_mean",
                    "contrast_id": contrast_id,
                    "estimate": estimate,
                    "ci_low": estimate - 0.1,
                    "ci_high": estimate + 0.1,
                    "confidence_level": 0.95,
                    "p_value": 0.01,
                    "adjusted_p_value": 0.02,
                    "adjustment_method": "holm",
                    "multiplicity_family": (
                        dg.figure_behavior_transitions.MULTIPLICITY_FAMILIES[condition]
                    ),
                    "n_mice": 2,
                    "n_sessions": 4,
                    "aggregation": "equal_mouse_mean",
                    "status": "pass",
                    "reason": None,
                }
            )
    rows.append(
        {
            **rows[0],
            "result_id": "descriptive_real_step_mouse_mean",
            "contrast_id": "reward_withdrawal_go_real_step",
        }
    )
    return pl.DataFrame(rows)


def test_prepare_selects_complete_real_trajectories_and_primary_effects() -> None:
    prepared = dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
        _time_trajectories(),
        _mouse_effects(),
        _statistics(),
    )

    assert prepared.trajectories.height == 24
    assert prepared.trajectories.get_column("anchor_type").unique().to_list() == ["real"]
    assert prepared.mouse_effects.height == 8
    assert prepared.statistics.select("transition_id", "condition", "estimate").to_dicts() == [
        {"transition_id": "reward_withdrawal", "condition": "go", "estimate": 0.2},
        {
            "transition_id": "reward_withdrawal",
            "condition": "go_minus_catch",
            "estimate": 0.1,
        },
        {"transition_id": "reward_restoration", "condition": "go", "estimate": 0.5},
        {
            "transition_id": "reward_restoration",
            "condition": "go_minus_catch",
            "estimate": 0.4,
        },
    ]


def test_prepare_rejects_missing_real_time_bin() -> None:
    trajectories = _time_trajectories().filter(
        ~(
            (pl.col("anchor_type") == "real")
            & (pl.col("transition_id") == "reward_withdrawal")
            & (pl.col("condition") == "go")
            & (pl.col("bin_index") == -3)
        )
    )

    with pytest.raises(ValueError, match="complete row set"):
        dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
            trajectories,
            _mouse_effects(),
            _statistics(),
        )


def test_prepare_rejects_mouse_mean_disagreement() -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("contrast_id") == "reward_withdrawal_go_pseudo_controlled_step")
        .then(0.25)
        .otherwise(pl.col("estimate"))
        .alias("estimate")
    )

    with pytest.raises(ValueError, match="do not reconstruct"):
        dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
            _time_trajectories(),
            _mouse_effects(),
            statistics,
        )


def test_prepare_rejects_failed_primary_contrast() -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("contrast_id") == "reward_restoration_go_pseudo_controlled_step")
        .then(pl.lit("not_estimable"))
        .otherwise(pl.col("status"))
        .alias("status")
    )

    with pytest.raises(ValueError, match="did not pass"):
        dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
            _time_trajectories(),
            _mouse_effects(),
            statistics,
        )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("analysis_id", "wrong_analysis"),
        ("result_id", "wrong_result"),
    ],
)
def test_prepare_rejects_incorrect_primary_statistic_identity(
    column: str,
    invalid_value: str,
) -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("contrast_id") == "reward_withdrawal_go_pseudo_controlled_step")
        .then(pl.lit(invalid_value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match="unexpected analysis/result identities"):
        dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
            _time_trajectories(),
            _mouse_effects(),
            statistics,
        )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("adjustment_method", "benjamini-hochberg"),
        ("multiplicity_family", "primary_go"),
        ("adjustment_method", None),
        ("multiplicity_family", None),
    ],
)
def test_prepare_rejects_incorrect_adjustment_contract(
    column: str,
    invalid_value: str | None,
) -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("contrast_id") == "reward_withdrawal_go_pseudo_controlled_step")
        .then(pl.lit(invalid_value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match="exact locked Holm families"):
        dg.figure_behavior_transitions.prepare_behavior_transition_figure_data(
            _time_trajectories(),
            _mouse_effects(),
            statistics,
        )
