"""Contracts for transition-resolved behavior tables.

The numeric records below are arithmetic fixtures for validating binning and
aggregation. They are not synthetic neuroscience observations and are never
written to publication result directories.
"""

import dataclasses

import polars as pl
import pytest

import dg.behavior_transitions


def _trial_fixture(session_id: str, mouse_id: str) -> pl.DataFrame:
    rows = []
    table_index = 0
    for block, block_start in (
        ("engaged_1", 0.0),
        ("no_reward", 100.0),
        ("engaged_2", 200.0),
    ):
        for within_block_index in range(20):
            start = block_start + within_block_index * 5.0
            condition = "go" if within_block_index % 2 == 0 else "catch"
            response = condition == "go" and block != "no_reward"
            rows.append(
                {
                    "_nwb_path": session_id,
                    "subject_id": mouse_id,
                    "_table_index": table_index,
                    "start_time": start,
                    "stop_time": start + 5.0,
                    "change_time": start + 2.5,
                    "reward_block": block,
                    "go": condition == "go",
                    "catch": condition == "catch",
                    "aborted": False,
                    "auto_rewarded": False,
                    "response_in_window": response,
                    "response_latency_from_licks": 0.3 if response else None,
                }
            )
            table_index += 1
    return pl.DataFrame(rows)


def _qc_fixture(session_id: str, mouse_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": [session_id],
            "subject_id": [mouse_id],
            "ecephys_session_id": [1],
            "recording_day": ["EPHYS_1"],
            "session_number": [1],
            "is_technically_valid": [True],
            "is_good_session": [True],
            "minimum_go_trials_pass": [True],
            "minimum_engaged_catch_trials_pass": [True],
            "engaged_1_response_pass": [True],
            "engaged_2_response_pass": [True],
            "no_reward_response_pass": [True],
            "engaged_1_dprime_pass": [True],
            "engaged_2_dprime_pass": [True],
            "reward_suppression_drop_pass": [True],
            "n_engaged_1_go_trials": [10],
            "n_no_reward_late_go_trials": [10],
            "n_engaged_2_early_go_trials": [10],
            "n_engaged_1_catch_trials": [10],
            "n_engaged_2_early_catch_trials": [10],
            "engaged_1_response_probability": [1.0],
            "engaged_2_early_response_probability": [1.0],
            "engaged_1_dprime": [2.0],
            "engaged_2_early_dprime": [2.0],
            "no_reward_late_response_probability": [0.0],
            "reward_suppression_drop": [1.0],
            "minimum_go_trials_threshold": [10],
            "minimum_engaged_catch_trials_threshold": [5],
            "minimum_engaged_response_rate_threshold": [0.5],
            "minimum_engaged_dprime_threshold": [1.0],
            "maximum_no_reward_response_rate_threshold": [0.2],
            "minimum_reward_suppression_drop_threshold": [0.3],
        }
    )


def _small_config() -> dg.behavior_transitions.TransitionConfig:
    return dg.behavior_transitions.TransitionConfig(
        time_window_seconds=20.0,
        time_bin_width_seconds=10.0,
        trial_window_count=4,
        trial_bin_width=2,
        effect_window_seconds=10.0,
        minimum_go_trials_per_effect_window=1,
        minimum_catch_trials_per_effect_window=1,
    )


def test_real_and_source_state_pseudo_boundaries_are_distinct() -> None:
    tables = dg.behavior_transitions.build_transition_tables(
        _trial_fixture("session-a", "mouse-a"),
        _qc_fixture("session-a", "mouse-a"),
        config=_small_config(),
    )

    assert tables.boundaries.select(
        "transition_id", "real_boundary_time", "pseudo_boundary_time", "pseudo_block"
    ).to_dicts() == [
        {
            "transition_id": "reward_withdrawal",
            "real_boundary_time": 100.0,
            "pseudo_boundary_time": 50.0,
            "pseudo_block": "engaged_1",
        },
        {
            "transition_id": "reward_restoration",
            "real_boundary_time": 200.0,
            "pseudo_boundary_time": 150.0,
            "pseudo_block": "no_reward",
        },
    ]
    assert tables.boundaries.get_column("boundary_status").to_list() == [
        "included",
        "included",
    ]
    assert tables.boundaries.get_column("latency_direction_multiplier").to_list() == [
        1.0,
        -1.0,
    ]


def test_complete_time_and_trial_grids_keep_go_catch_and_missing_latency() -> None:
    tables = dg.behavior_transitions.build_transition_tables(
        _trial_fixture("session-a", "mouse-a"),
        _qc_fixture("session-a", "mouse-a"),
        config=_small_config(),
    )

    expected_rows = 2 * 2 * 4 * 2
    assert tables.session_time_bins.height == expected_rows
    assert tables.session_trial_bins.height == expected_rows
    assert tables.mouse_time_bins.height == expected_rows
    assert tables.mouse_trial_bins.height == expected_rows
    assert tables.session_time_bins.get_column("bin_fully_covered").all()
    assert tables.session_trial_bins.get_column("bin_fully_covered").all()
    assert set(tables.session_time_bins.get_column("condition")) == {"go", "catch"}

    catch = tables.mouse_time_bins.filter(pl.col("condition") == "catch")
    assert catch.get_column("response_probability").drop_nulls().to_list() == [0.0] * catch.height
    assert catch.get_column("mean_session_median_response_latency_seconds").null_count() == (
        catch.height
    )
    assert catch.get_column("n_sessions_latency_missing").to_list() == [1] * catch.height


def test_transition_effects_are_signed_and_pseudo_controlled_separately() -> None:
    tables = dg.behavior_transitions.build_transition_tables(
        _trial_fixture("session-a", "mouse-a"),
        _qc_fixture("session-a", "mouse-a"),
        config=_small_config(),
    )

    go = tables.session_effects.filter(pl.col("condition") == "go").select(
        "transition_id", "real_step", "pseudo_step", "pseudo_controlled_step"
    )
    assert go.to_dicts() == [
        {
            "transition_id": "reward_withdrawal",
            "real_step": 1.0,
            "pseudo_step": 0.0,
            "pseudo_controlled_step": 1.0,
        },
        {
            "transition_id": "reward_restoration",
            "real_step": 1.0,
            "pseudo_step": 0.0,
            "pseudo_controlled_step": 1.0,
        },
    ]
    specificity = tables.session_effects.filter(pl.col("condition") == "go_minus_catch")
    assert specificity.get_column("pseudo_controlled_step").to_list() == [1.0, 1.0]
    assert tables.mouse_effects.filter(pl.col("condition") == "go").get_column(
        "pseudo_controlled_step"
    ).to_list() == [1.0, 1.0]


def test_latency_orientation_is_positive_for_nr_slowing_and_reward_speeding() -> None:
    trials = _trial_fixture("session-a", "mouse-a").with_columns(
        pl.when(pl.col("go"))
        .then(True)
        .otherwise(pl.col("response_in_window"))
        .alias("response_in_window"),
        pl.when(pl.col("go") & (pl.col("reward_block") == "no_reward"))
        .then(0.4)
        .when(pl.col("go"))
        .then(0.2)
        .otherwise(None)
        .alias("response_latency_from_licks"),
    )
    tables = dg.behavior_transitions.build_transition_tables(
        trials,
        _qc_fixture("session-a", "mouse-a"),
        config=_small_config(),
    )

    go = tables.session_effects.filter(pl.col("condition") == "go")
    assert go.get_column("real_latency_step_seconds").to_list() == pytest.approx([0.2, 0.2])
    assert go.get_column("pseudo_latency_step_seconds").to_list() == pytest.approx([0.0, 0.0])
    assert go.get_column("pseudo_controlled_latency_step_seconds").to_list() == pytest.approx(
        [0.2, 0.2]
    )


def test_effect_trial_minimum_preserves_low_support_sessions_as_missing() -> None:
    config = dataclasses.replace(
        _small_config(),
        minimum_go_trials_per_effect_window=3,
        minimum_catch_trials_per_effect_window=3,
    )
    tables = dg.behavior_transitions.build_transition_tables(
        _trial_fixture("session-a", "mouse-a"),
        _qc_fixture("session-a", "mouse-a"),
        config=config,
    )

    assert tables.session_effect_windows.height == 2 * 2 * 2 * 2
    assert set(tables.session_effect_windows.get_column("window_status")) == {
        "insufficient_condition_trials"
    }
    assert tables.session_effect_windows.get_column("response_probability").null_count() == (
        tables.session_effect_windows.height
    )
    assert tables.effect_window_count_summary.get_column("n_sessions_meeting_minimum").sum() == 0


def test_mouse_bins_weight_sessions_equally_and_report_missing_sessions() -> None:
    session_bins = pl.DataFrame(
        {
            "_nwb_path": ["session-a", "session-b"],
            "subject_id": ["mouse-a", "mouse-a"],
            "transition_id": ["reward_withdrawal"] * 2,
            "transition_order": [1, 1],
            "anchor_type": ["real", "real"],
            "condition": ["go", "go"],
            "bin_index": [-1, -1],
            "analysis_block": ["engaged_1", "engaged_1"],
            "relative_bin_start_seconds": [-10.0, -10.0],
            "relative_bin_stop_seconds": [0.0, 0.0],
            "relative_bin_center_seconds": [-5.0, -5.0],
            "included_in_transition_analysis": [True, True],
            "response_probability": [1.0, 0.0],
            "n_trials": [10, 1],
            "n_responses": [10, 0],
            "median_response_latency_seconds": [0.2, None],
            "n_latency_responses": [10, 0],
        }
    )

    result = dg.behavior_transitions.aggregate_mouse_bins(session_bins, bin_kind="time")

    assert result.get_column("response_probability").item() == 0.5
    assert result.get_column("n_trials_contributing").item() == 11
    assert result.get_column("n_sessions_response_contributing").item() == 2
    assert result.get_column("n_sessions_latency_contributing").item() == 1
    assert result.get_column("n_sessions_latency_missing").item() == 1


def test_d03_tables_expose_joint_pass_counts_and_metric_distributions() -> None:
    first = _qc_fixture("session-a", "mouse-a")
    second = _qc_fixture("session-b", "mouse-b").with_columns(
        pl.lit(False).alias("engaged_2_response_pass"),
        pl.lit(0.4).alias("engaged_2_early_response_probability"),
        pl.lit(False).alias("is_good_session"),
    )

    criterion, distributions = dg.behavior_transitions.build_d03_qc_tables(
        pl.concat([first, second])
    )

    e2 = criterion.filter(pl.col("criterion_id") == "engaged_2_response").row(0, named=True)
    assert e2["n_sessions_pass"] == 1
    assert e2["n_sessions_fail"] == 1
    assert e2["n_mice_with_any_session_pass"] == 1
    metric = distributions.filter(
        pl.col("metric_id") == "early_engaged_2_response_probability"
    ).row(0, named=True)
    assert metric["threshold_value"] == 0.5
    assert metric["median"] == pytest.approx(0.7)
    assert metric["n_sessions_meeting_threshold"] == 1
    overall = criterion.filter(pl.col("criterion_id") == "all_approved_d03_thresholds").row(
        0, named=True
    )
    assert overall["n_sessions_pass"] == 1
    assert overall["n_mice_with_any_session_pass"] == 1


def test_invalid_config_and_session_mapping_fail_fast() -> None:
    with pytest.raises(ValueError, match="divisible"):
        dg.behavior_transitions.TransitionConfig(
            time_window_seconds=21,
            time_bin_width_seconds=10,
        )

    with pytest.raises(ValueError, match="different session/mouse coverage"):
        dg.behavior_transitions.build_transition_tables(
            _trial_fixture("session-a", "mouse-a"),
            _qc_fixture("session-a", "mouse-b"),
            config=_small_config(),
        )


def test_trajectory_bootstrap_and_effect_statistics_are_mouse_primary() -> None:
    trials = pl.concat(
        [
            _trial_fixture("session-a", "mouse-a"),
            _trial_fixture("session-b", "mouse-b"),
        ]
    )
    qc = pl.concat(
        [
            _qc_fixture("session-a", "mouse-a"),
            _qc_fixture("session-b", "mouse-b"),
        ]
    )
    tables = dg.behavior_transitions.build_transition_tables(trials, qc, config=_small_config())

    trajectory = dg.behavior_transitions.summarize_mouse_trajectory(
        tables.mouse_time_bins,
        bin_kind="time",
        seed=1051,
        n_resamples=100,
    )
    assert trajectory.height == 2 * 2 * 4 * 2
    assert trajectory.get_column("n_mice_response_contributing").min() == 2
    assert trajectory.get_column("response_probability_ci_low").null_count() == 0

    statistics = dg.behavior_transitions.build_transition_statistics(
        tables.mouse_effects,
        dandiset_version="arithmetic-fixture",
        code_version="arithmetic-fixture",
        seed=1051,
        n_bootstrap_resamples=100,
        n_sign_flip_resamples=100,
        minimum_go_trials_per_effect_window=1,
        minimum_catch_trials_per_effect_window=1,
    )
    assert statistics.height == 30
    assert statistics.get_column("analysis_tier").unique().to_list() == ["exploratory"]
    tested = statistics.filter(pl.col("p_value").is_not_null())
    assert tested.height == 4
    assert set(tested.get_column("multiplicity_family")) == {
        "behavior_transition_primary_go",
        "behavior_transition_primary_specificity",
    }
    latency = statistics.filter(pl.col("result_id").str.contains("latency"))
    assert latency.get_column("p_value").null_count() == latency.height
    assert latency.get_column("inclusion_definition").str.contains("responder-only").all()
    sample_sizes = tables.effect_statistic_sample_sizes
    latency_counts = sample_sizes.filter(pl.col("estimand") == "responder_only_latency")
    assert (latency_counts.get_column("n_eligible_trials") >= 0).all()
    assert (latency_counts.get_column("n_finite_canonical_response_latencies") >= 0).all()


def test_technical_response_requires_a_finite_canonical_latency() -> None:
    trials = _trial_fixture("session-a", "mouse-a").with_columns(
        pl.when(pl.col("_table_index") == 0)
        .then(None)
        .otherwise(pl.col("response_latency_from_licks"))
        .alias("response_latency_from_licks")
    )

    with pytest.raises(ValueError, match="invalid response latency"):
        dg.behavior_transitions.build_transition_tables(
            trials,
            _qc_fixture("session-a", "mouse-a"),
            config=_small_config(),
        )
