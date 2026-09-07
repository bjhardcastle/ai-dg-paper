"""Quality-filter tests based on rows from DANDI:001051/0.260825.2232."""

import dataclasses
import math

import polars as pl
import pytest

import dg.quality

PUBLIC_SESSION_SOURCE = (
    "https://dandiarchive.s3.amazonaws.com/blobs/585/411/5854110b-2b31-4bbf-881e-3280f7b87b95"
)


def _published_trial_excerpt() -> pl.DataFrame:
    # Completed go trials spanning all three blocks in sub-604914, 2022-04-27.
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 9,
            "id": [130, 144, 170, 294, 298, 299, 450, 451, 452],
            "_table_index": [130, 144, 170, 294, 298, 299, 450, 451, 452],
            "start_time": [
                346.93608,
                396.47823,
                482.79968,
                927.97178,
                962.51788,
                971.52486,
                2427.27526,
                2435.54899,
                2443.80595,
            ],
            "stop_time": [
                355.19344,
                404.71809,
                491.05698,
                936.23045,
                970.77499,
                979.78367,
                2435.53333,
                2443.78935,
                2452.06323,
            ],
            "change_time": [
                351.46229,
                401.00426,
                487.32576,
                932.4977700000001,
                967.79384,
                977.5522100000001,
                2431.80226,
                2441.57612,
                2448.33174,
            ],
            "go": [True] * 9,
            "catch": [False] * 9,
            "aborted": [False] * 9,
            "auto_rewarded": [False] * 9,
            "hit": [True, True, True, True, False, False, True, True, True],
            "miss": [False] * 9,
            "false_alarm": [False] * 9,
            "correct_reject": [False] * 9,
            "no_reward_epoch": [
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        }
    )


def _published_behavior_excerpt() -> pl.DataFrame:
    # Two published catch trials from the same session. Both were false alarms;
    # together with the go rows they expose the indiscriminate-licker guard.
    catch_trials = pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 2,
            "id": [142, 478],
            "_table_index": [142, 478],
            "start_time": [382.9662, 2550.39593],
            "stop_time": [391.22377, 2558.65218],
            "change_time": [387.49361999999996, 2554.92147],
            "go": [False, False],
            "catch": [True, True],
            "aborted": [False, False],
            "auto_rewarded": [False, False],
            "hit": [False, False],
            "miss": [False, False],
            "false_alarm": [True, True],
            "correct_reject": [False, False],
            "no_reward_epoch": [False, False],
        }
    )
    return pl.concat([_published_trial_excerpt(), catch_trials]).sort("_table_index")


def _published_block_endpoint_excerpt() -> pl.DataFrame:
    # Actual E1/NR/E2 endpoint trials from sub-604914, 2022-04-27.
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 5,
            "_table_index": [292, 293, 449, 450, 797],
            "start_time": [923.46825, 924.96933, 2418.26772, 2427.27526, 3937.77099],
            "stop_time": [924.93612, 927.77182, 2426.52535, 2435.53333, 3946.02867],
            "no_reward_epoch": [False, True, True, False, False],
        }
    )


def _published_unit_excerpt() -> pl.DataFrame:
    # Complete spikes for NWB unit 219489 in [347, 357) and [2428, 2438).
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "id": pl.Series([219489], dtype=pl.Int32),
            "spike_times": [
                [
                    347.9157456367911,
                    348.56824274737454,
                    350.49373422082806,
                    351.6078959537145,
                    353.0411229403687,
                    356.11077601386194,
                    2428.9057305049464,
                    2431.103187440737,
                    2431.117420711042,
                    2431.215320277519,
                    2431.2200202567064,
                    2432.1301828929504,
                    2432.976845810387,
                    2433.5548099176876,
                    2435.200035965558,
                    2436.104265294743,
                    2437.19112714852,
                ]
            ],
        }
    )


def _published_unit_windows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "engaged_1_start": [347.0],
            "engaged_1_stop": [357.0],
            "engaged_2_start": [2428.0],
            "engaged_2_stop": [2438.0],
        }
    )


def _published_raw_lick_excerpt() -> pl.DataFrame:
    # Trial 298 has an in-window lick although its online hit event is false;
    # outcome events were not a state-independent response label in NR.
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE, PUBLIC_SESSION_SOURCE],
            "id": [294, 298],
            "change_time": [932.4977700000001, 967.79384],
            "lick_times": [
                [932.67598, 932.67598, 932.82576, 934.89424],
                [964.65235, 966.72075, 966.83738, 966.97091, 967.97229, 968.4391],
            ],
            "hit": [True, False],
            "no_reward_epoch": [True, True],
        }
    )


def test_raw_lick_response_is_state_independent_of_online_outcome() -> None:
    responses = dg.quality.add_trial_response_from_licks(_published_raw_lick_excerpt())

    assert responses.get_column("response_in_window").to_list() == [True, True]
    assert responses.get_column("n_response_window_licks").to_list() == [2, 2]
    assert responses.get_column("n_raw_response_window_licks").to_list() == [3, 2]
    assert responses.get_column("n_duplicate_response_window_licks").to_list() == [1, 0]
    assert responses.get_column("n_raw_lick_timestamps").to_list() == [4, 6]
    assert responses.get_column("n_unique_lick_timestamps").to_list() == [3, 6]
    assert responses.get_column("n_duplicate_lick_timestamps").to_list() == [1, 0]
    assert responses.get_column("response_latency_from_licks").to_list()[0] == pytest.approx(
        0.17820999999992182
    )
    discrepant = responses.filter(pl.col("id") == 298).row(0, named=True)
    assert discrepant["response_in_window"]
    assert not discrepant["hit"]


def test_raw_lick_response_uses_task_softwares_half_open_window() -> None:
    # Arithmetic boundary fixture, not a neuroscience observation.
    trials = pl.DataFrame(
        {
            "change_time": [0.0],
            "lick_times": [[0.15, 0.150001, 0.75, 0.750001]],
        }
    )

    response = dg.quality.add_trial_response_from_licks(trials).row(0, named=True)

    assert response["response_in_window"]
    assert response["response_latency_from_licks"] == pytest.approx(0.150001)
    assert response["n_response_window_licks"] == 2


@pytest.mark.parametrize(
    ("lick_times", "change_time", "status", "lick_times_valid", "event_time_valid"),
    [
        (None, 932.4977700000001, "missing_lick_times", False, True),
        ([932.67598, math.nan], 932.4977700000001, "invalid_lick_times", False, True),
        ([932.82576, 932.67598], 932.4977700000001, "unsorted_lick_times", False, True),
        ([932.67598], math.nan, "invalid_event_time", True, False),
    ],
)
def test_raw_lick_response_fails_closed(
    lick_times: list[float] | None,
    change_time: float,
    status: str,
    lick_times_valid: bool,
    event_time_valid: bool,
) -> None:
    trials = pl.DataFrame(
        {"change_time": [change_time], "lick_times": [lick_times]},
        schema={"change_time": pl.Float64, "lick_times": pl.List(pl.Float64)},
    )

    result = dg.quality.add_trial_response_from_licks(trials)

    assert result.get_column("response_in_window").item() is None
    assert result.get_column("lick_times_valid").item() is lick_times_valid
    assert result.get_column("response_event_time_valid").item() is event_time_valid
    assert result.get_column("lick_response_status").item() == status


def test_behavior_summary_can_use_raw_lick_response_column() -> None:
    trials = _published_behavior_excerpt().with_columns(
        pl.when(pl.col("id").is_in([298, 299]))
        .then(pl.lit(True))
        .otherwise(pl.col("hit") | pl.col("false_alarm"))
        .alias("response_in_window")
    )

    result = dg.quality.summarize_session_behavior(
        trials,
        no_reward_tail_seconds=None,
        final_engaged_exclusion_seconds=0,
        response_column="response_in_window",
    ).collect()

    assert result.get_column("no_reward_response_rate").item() == 1.0
    assert result.get_column("behavior_response_source").item() == "response_in_window"


def test_well_isolated_filter_uses_strict_project_thresholds() -> None:
    units = pl.DataFrame(
        {
            "id": [219487, 219488, 219489, 219492],
            "isi_violations": [
                4.042293115676158,
                0.5289306117865576,
                0.213029325911995,
                0.0268688756457586,
            ],
            "amplitude_cutoff": [0.5, 0.5, 0.0464355489869127, 0.0049734098824021],
            "quality": ["noise", "noise", "noise", "good"],
        }
    )

    filtered = dg.quality.filter_well_isolated_units(units)

    assert filtered.get_column("id").to_list() == [219492]


@pytest.mark.parametrize(
    ("isi", "amplitude", "reason"),
    [
        (None, 0.0464355489869127, "invalid_isi_violations"),
        (math.nan, 0.0464355489869127, "invalid_isi_violations"),
        (-0.1, 0.0464355489869127, "invalid_isi_violations"),
        (0.5, 0.0464355489869127, "isi_violations_above_threshold"),
        (0.213029325911995, None, "invalid_amplitude_cutoff"),
        (0.213029325911995, math.inf, "invalid_amplitude_cutoff"),
        (0.213029325911995, -0.1, "invalid_amplitude_cutoff"),
        (0.213029325911995, 0.1, "amplitude_cutoff_above_threshold"),
    ],
)
def test_isolation_filter_rejects_invalid_and_equal_threshold_values(
    isi: float | None,
    amplitude: float | None,
    reason: str,
) -> None:
    unit = pl.DataFrame(
        {
            "isi_violations": [isi],
            "amplitude_cutoff": [amplitude],
            "quality": ["good"],
        },
        schema={
            "isi_violations": pl.Float64,
            "amplitude_cutoff": pl.Float64,
            "quality": pl.String,
        },
    )

    audited = dg.quality.add_well_isolated_unit_flag(unit)

    assert not audited.get_column("well_isolated").item()
    assert reason in audited.get_column("unit_quality_exclusion_reasons").item()


@pytest.mark.parametrize(
    ("quality", "reason"),
    [(None, "invalid_quality"), ("noise", "quality_not_good")],
)
def test_isolation_filter_requires_author_good_label(
    quality: str | None,
    reason: str,
) -> None:
    unit = pl.DataFrame(
        {
            "isi_violations": [0.0268688756457586],
            "amplitude_cutoff": [0.0049734098824021],
            "quality": [quality],
        },
        schema_overrides={"quality": pl.String},
    )

    audited = dg.quality.add_well_isolated_unit_flag(unit)

    assert not audited.get_column("well_isolated").item()
    assert reason in audited.get_column("unit_quality_exclusion_reasons").item()


def test_reward_blocks_and_behavior_summary() -> None:
    trials = _published_trial_excerpt()

    labeled = dg.quality.label_reward_blocks(trials).collect()
    summary = dg.quality.summarize_session_behavior(
        _published_behavior_excerpt(),
        final_engaged_exclusion_seconds=0,
    ).collect()

    assert labeled.group_by("reward_block").len().sort("reward_block").to_dicts() == [
        {"reward_block": "engaged_1", "len": 3},
        {"reward_block": "engaged_2", "len": 3},
        {"reward_block": "no_reward", "len": 3},
    ]
    assert summary.get_column("engaged_1_response_rate").item() == 1.0
    # Only trial 294 has a valid hit/miss pair. Trials 298 and 299 have neither
    # outcome because online outcomes were disabled during NR.
    assert summary.get_column("no_reward_response_rate").item() == 1.0
    assert summary.get_column("n_invalid_behavior_labels").item() == 2
    assert summary.get_column("engaged_2_response_rate").item() == 1.0
    assert 0.4 < summary.get_column("engaged_1_dprime").item() < 0.5
    assert 0.4 < summary.get_column("engaged_2_dprime").item() < 0.5
    assert 0 < summary.get_column("engaged_1_response_rate_ci_low").item() < 1
    assert summary.get_column("engaged_1_response_rate_ci_high").item() == 1

    excerpt_thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=3,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0.9,
        minimum_engaged_dprime=0.4,
        maximum_no_reward_response_rate=0.34,
    )
    online_label_audit = dg.quality.add_good_session_flag(
        summary,
        thresholds=excerpt_thresholds,
    )
    assert not online_label_audit.get_column("is_good_session").item()
    assert (
        "invalid_behavior_labels"
        in online_label_audit.get_column("session_exclusion_reasons").item()
    )

    discriminating_thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=3,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0.9,
        minimum_engaged_dprime=1.0,
        maximum_no_reward_response_rate=0.34,
    )
    rejected = dg.quality.add_good_session_flag(
        summary,
        thresholds=discriminating_thresholds,
    )
    assert not rejected.get_column("is_good_session").item()
    assert "dprime_below_threshold" in rejected.get_column("session_exclusion_reasons").item()

    suppression_thresholds = dataclasses.replace(
        excerpt_thresholds,
        minimum_reward_suppression_drop=0.7,
    )
    suppression_rejected = dg.quality.add_good_session_flag(
        summary,
        thresholds=suppression_thresholds,
    )
    assert not suppression_rejected.get_column("reward_suppression_drop_pass").item()
    assert (
        "reward_suppression_drop_below_threshold"
        in suppression_rejected.get_column("session_exclusion_reasons").item()
    )


def test_unevaluable_session_metric_is_not_reported_as_a_threshold_failure() -> None:
    summary = dg.quality.summarize_session_behavior(
        _published_trial_excerpt(),
        final_engaged_exclusion_seconds=0,
    ).collect()
    thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=3,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0.9,
        minimum_engaged_dprime=0.4,
        maximum_no_reward_response_rate=0.34,
    )

    audited = dg.quality.add_good_session_flag(summary, thresholds=thresholds)
    reasons = audited.get_column("session_exclusion_reasons").item()

    assert not audited.get_column("is_good_session").item()
    assert "engaged_1_dprime_unevaluable" in reasons
    assert "engaged_1_dprime_below_threshold" not in reasons


def test_behavior_filter_rejects_nonfinite_change_event_times() -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.when(pl.col("id") == 130)
        .then(pl.lit(math.nan))
        .otherwise(pl.col("change_time"))
        .alias("change_time")
    )
    summary = dg.quality.summarize_session_behavior(
        trials,
        final_engaged_exclusion_seconds=0,
    ).collect()
    thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=2,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0,
        minimum_engaged_dprime=0,
        maximum_no_reward_response_rate=1,
    )

    audited = dg.quality.add_good_session_flag(summary, thresholds=thresholds)

    assert audited.get_column("n_invalid_behavior_event_times").item() == 1
    assert not audited.get_column("is_good_session").item()
    assert "invalid_behavior_event_times" in audited.get_column("session_exclusion_reasons").item()


@pytest.mark.parametrize(
    ("column", "trial_id"),
    [("aborted", 130), ("false_alarm", 142)],
)
def test_behavior_filter_rejects_unknown_task_or_outcome_labels(
    column: str,
    trial_id: int,
) -> None:
    trials = _published_behavior_excerpt().with_columns(
        pl.when(pl.col("id") == trial_id)
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col(column))
        .alias(column)
    )

    summary = dg.quality.summarize_session_behavior(
        trials,
        final_engaged_exclusion_seconds=0,
    ).collect()
    audited = dg.quality.add_good_session_flag(summary)

    # Two published NR trials already lack an online outcome; the mutation
    # introduces one additional invalid label.
    assert summary.get_column("n_invalid_behavior_labels").item() == 3
    assert not audited.get_column("behavior_labels_pass").item()
    assert "invalid_behavior_labels" in audited.get_column("session_exclusion_reasons").item()


def test_behavior_filter_rejects_change_time_outside_trial() -> None:
    trials = _published_behavior_excerpt().with_columns(
        pl.when(pl.col("id") == 130)
        .then(pl.col("stop_time") + 1)
        .otherwise(pl.col("change_time"))
        .alias("change_time")
    )

    summary = dg.quality.summarize_session_behavior(
        trials,
        final_engaged_exclusion_seconds=0,
    ).collect()
    audited = dg.quality.add_good_session_flag(summary)

    assert summary.get_column("n_behavior_event_times_outside_trial").item() == 1
    assert not audited.get_column("behavior_event_times_pass").item()
    assert "invalid_behavior_event_times" in audited.get_column("session_exclusion_reasons").item()


@pytest.mark.parametrize(
    ("column", "bad_value", "audit_column"),
    [
        ("start_time", None, "n_invalid_start_times"),
        ("start_time", math.inf, "n_invalid_start_times"),
        ("start_time", math.nan, "n_invalid_start_times"),
        ("stop_time", None, "n_invalid_stop_times"),
        ("stop_time", -math.inf, "n_invalid_stop_times"),
        ("_table_index", None, "n_invalid_table_indices"),
        ("_table_index", math.inf, "n_invalid_table_indices"),
    ],
)
def test_task_structure_rejects_missing_or_nonfinite_values(
    column: str,
    bad_value: float | None,
    audit_column: str,
) -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.when(pl.col("_table_index") == 130)
        .then(pl.lit(bad_value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    audit = dg.quality.summarize_task_structure(trials).collect()

    assert audit.get_column(audit_column).item() == 1
    assert not audit.get_column("task_structure_valid").item()


def test_task_structure_rejects_nonpositive_trial_duration() -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.when(pl.col("_table_index") == 130)
        .then(pl.col("start_time"))
        .otherwise(pl.col("stop_time"))
        .alias("stop_time")
    )

    audit = dg.quality.summarize_task_structure(trials).collect()

    assert audit.get_column("n_nonpositive_trial_durations").item() == 1
    assert not audit.get_column("task_structure_valid").item()


def test_task_structure_rejects_duplicate_table_indices() -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.when(pl.col("_table_index") == 144)
        .then(pl.lit(130))
        .otherwise(pl.col("_table_index"))
        .alias("_table_index")
    )

    audit = dg.quality.summarize_task_structure(trials).collect()

    assert audit.get_column("n_duplicate_table_indices").item() == 1
    assert not audit.get_column("task_structure_valid").item()


@pytest.mark.parametrize(
    "conflict_column",
    ["no_reward_epoch_conflict", "companion_reward_epoch_conflict"],
)
def test_task_structure_rejects_reward_epoch_conflicts_when_present(
    conflict_column: str,
) -> None:
    trials = _published_trial_excerpt().with_columns(
        (pl.col("_table_index") == 130).alias(conflict_column)
    )

    audit = dg.quality.summarize_task_structure(trials).collect()

    assert audit.get_column("n_reward_epoch_conflicts").item() == 1
    assert not audit.get_column("task_structure_valid").item()


def test_irrelevant_companion_conflict_does_not_override_nwb_authority() -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.lit("nwb").alias("no_reward_epoch_source"),
        (pl.col("_table_index") == 130).alias("companion_reward_epoch_conflict"),
        pl.lit(False).alias("no_reward_epoch_conflict"),
    )

    audit = dg.quality.summarize_task_structure(trials).collect()

    assert audit.get_column("n_reward_epoch_conflicts").item() == 0
    assert audit.get_column("task_structure_valid").item()


def test_invalid_task_structure_yields_auditable_null_engaged_windows() -> None:
    trials = _published_trial_excerpt().with_columns(
        pl.when(pl.col("_table_index") == 298)
        .then(pl.lit(False))
        .otherwise(pl.col("no_reward_epoch"))
        .alias("no_reward_epoch")
    )

    windows = dg.quality.get_engaged_block_windows(
        trials,
        final_engaged_exclusion_seconds=0,
    ).collect()

    assert not windows.get_column("task_structure_valid").item()
    assert not windows.get_column("engaged_window_valid").item()
    assert windows.get_column("engaged_window_status").item() == "invalid_task_structure"
    assert windows.select(
        "engaged_1_start",
        "engaged_1_stop",
        "engaged_2_start",
        "engaged_2_stop",
    ).null_count().row(0) == (1, 1, 1, 1)


def test_final_600_seconds_are_removed_from_second_unit_qc_window() -> None:
    windows = dg.quality.get_engaged_block_windows(
        _published_block_endpoint_excerpt(),
        final_engaged_exclusion_seconds=600.0,
    ).collect()

    assert windows.get_column("task_structure_valid").item()
    assert windows.get_column("engaged_window_valid").item()
    assert windows.get_column("engaged_window_status").item() == "pass"
    assert windows.get_column("engaged_2_excluded_tail_seconds").item() == 600.0
    assert windows.get_column("engaged_2_start").item() == 2427.27526
    assert windows.get_column("engaged_2_recorded_stop").item() == 3946.02867
    assert windows.get_column("engaged_2_stop").item() == pytest.approx(3346.02867)


def test_engaged_rate_stability_uses_sorted_spike_search() -> None:
    units = _published_unit_excerpt()
    windows = _published_unit_windows()
    thresholds = dg.quality.CoarseUnitStabilityThresholds(
        minimum_spikes_per_block=5,
        minimum_engaged_rate_ratio=0.5,
    )

    stability = dg.quality.summarize_coarse_engaged_rate_stability(
        units,
        windows,
        thresholds=thresholds,
    )

    assert stability.get_column("engaged_1_spike_count").item() == 6
    assert stability.get_column("engaged_2_spike_count").item() == 11
    assert stability.get_column("engaged_rate_ratio").item() == 6 / 11
    assert stability.get_column("coarse_engaged_rate_consistent").item()
    assert (
        dg.quality.filter_coarse_engaged_rate_consistent_units(
            units,
            stability,
        ).height
        == 1
    )


@pytest.mark.parametrize("case", ["nonfinite", "unsorted"])
def test_engaged_rate_stability_rejects_invalid_spike_arrays(case: str) -> None:
    units = _published_unit_excerpt()
    spikes = units.get_column("spike_times").to_list()[0]
    if case == "nonfinite":
        spikes[0] = math.inf
        message = "finite values"
    else:
        spikes[0], spikes[1] = spikes[1], spikes[0]
        message = "sorted"
    units = units.with_columns(pl.Series("spike_times", [spikes]))

    with pytest.raises(ValueError, match=message):
        dg.quality.summarize_coarse_engaged_rate_stability(
            units,
            _published_unit_windows(),
        )


def test_engaged_rate_stability_rejects_duplicate_unit_keys() -> None:
    units = pl.concat([_published_unit_excerpt(), _published_unit_excerpt()])

    with pytest.raises(ValueError, match="unique"):
        dg.quality.summarize_coarse_engaged_rate_stability(
            units,
            _published_unit_windows(),
        )


@pytest.mark.parametrize(
    ("case", "expected_status"),
    [
        ("invalid_window", "invalid_window"),
        ("missing_window", "missing_window"),
        ("missing_spike_times", "missing_spike_times"),
    ],
)
def test_unevaluable_stability_has_null_metrics(
    case: str,
    expected_status: str,
) -> None:
    units = _published_unit_excerpt()
    windows = _published_unit_windows()
    if case == "invalid_window":
        windows = windows.with_columns(pl.col("engaged_1_start").alias("engaged_1_stop"))
    elif case == "missing_window":
        windows = windows.clear()
    else:
        units = units.with_columns(pl.lit(None, dtype=pl.List(pl.Float64)).alias("spike_times"))

    stability = dg.quality.summarize_coarse_engaged_rate_stability(units, windows)

    assert stability.get_column("coarse_engaged_rate_status").item() == expected_status
    assert not stability.get_column("coarse_engaged_rate_consistent").item()
    metric_columns = [
        "engaged_1_spike_count",
        "engaged_2_spike_count",
        "engaged_1_rate_hz",
        "engaged_2_rate_hz",
        "engaged_rate_ratio",
    ]
    assert stability.select(metric_columns).null_count().row(0) == (1, 1, 1, 1, 1)
    assert stability.schema["engaged_1_spike_count"] == pl.Int64
    assert stability.schema["engaged_1_rate_hz"] == pl.Float64
