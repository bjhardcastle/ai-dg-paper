"""Behavior-table tests using published DANDI:001051 trial rows."""

import polars as pl
import pytest

import dg.behavior
import dg.quality

PUBLIC_SESSION_SOURCE = (
    "https://dandiarchive.s3.amazonaws.com/blobs/585/411/5854110b-2b31-4bbf-881e-3280f7b87b95"
)


def _published_trial_excerpt() -> pl.DataFrame:
    # Completed go trials spanning E1, NR, and E2 in sub-604914, 2022-04-27.
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 9,
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
            "lick_times": [
                [351.64013, 351.64013, 351.77339, 352.07336, 353.92522, 354.05833, 355.77649],
                [
                    401.19952,
                    401.19952,
                    401.61485,
                    401.69822,
                    401.83166,
                    401.94842,
                    402.06506,
                    402.19857,
                    402.31542,
                    402.43241,
                    402.56563,
                    402.69903,
                    403.3329,
                    403.46631,
                    404.0835,
                    404.26695,
                    404.38379,
                ],
                [487.50374, 487.50374],
                [932.67598, 932.67598, 932.82576, 934.89424, 935.01089, 935.17768],
                [964.65235, 966.72075, 966.83738, 966.97091, 967.97229, 967.97229, 968.4391],
                [
                    973.5768,
                    977.17949,
                    977.73026,
                    977.73026,
                    977.84696,
                    979.39798,
                    979.54831,
                ],
                [2431.98125, 2431.98125],
                [2441.75443, 2441.75443],
                [2448.51079, 2448.51079],
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
    # Published catch trials expose the engaged-block discrimination criterion.
    catch_trials = pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 2,
            "_table_index": [142, 478],
            "start_time": [382.9662, 2550.39593],
            "stop_time": [391.22377, 2558.65218],
            "change_time": [387.49361999999996, 2554.92147],
            "lick_times": [
                [387.6702, 387.6702, 387.85329, 387.97002],
                [2555.09882, 2555.09882, 2556.85015, 2558.53499],
            ],
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


def _inventory() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "subject_id": ["604914"],
        }
    )


def _canonical_published_trials() -> pl.DataFrame:
    return dg.behavior.add_response_in_window(_published_trial_excerpt())


def _canonical_published_behavior() -> pl.DataFrame:
    return dg.behavior.add_response_in_window(_published_behavior_excerpt())


def test_response_in_window_is_derived_from_published_lick_times() -> None:
    trials = _canonical_published_behavior()

    assert trials.get_column("response_in_window").all()
    # These NR trials expose why the author hit label cannot define responses.
    assert trials.filter(pl.col("_table_index").is_in([298, 299])).select(
        "hit", "response_in_window"
    ).to_dicts() == [
        {"hit": False, "response_in_window": True},
        {"hit": False, "response_in_window": True},
    ]


def test_session_summary_separates_technical_and_threshold_selection() -> None:
    thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=3,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0.9,
        minimum_engaged_dprime=1.0,
        maximum_no_reward_response_rate=0.34,
    )

    sessions = dg.behavior.summarize_behavior_sessions(
        _canonical_published_behavior(),
        _inventory(),
        thresholds=thresholds,
        late_engaged_2_seconds=0,
    )

    assert sessions.get_column("is_technically_valid").item()
    assert not sessions.get_column("is_good_session").item()
    assert sessions.get_column("technical_exclusion_reasons").item() == ""
    assert "dprime_below_threshold" in sessions.get_column("session_exclusion_reasons").item()
    assert sessions.get_column("engaged_1_response_probability").item() == 1.0
    assert sessions.get_column("no_reward_late_response_probability").item() == 1.0
    assert sessions.get_column("engaged_2_early_response_probability").item() == 1.0
    assert sessions.get_column("reversible_gating_estimate").item() == 0.0
    assert sessions.get_column("n_engaged_1_response_label_disagreements").item() == 0
    assert sessions.get_column("n_no_reward_response_label_disagreements").item() == 0
    assert sessions.get_column("n_no_reward_response_label_comparisons").item() == 1
    assert sessions.get_column("n_no_reward_online_outcome_absent").item() == 2
    assert sessions.get_column("no_reward_response_label_agreement_rate").item() == 1.0
    assert sessions.get_column("n_engaged_2_response_label_disagreements").item() == 0
    assert sessions.get_column("behavior_response_source").item() == "response_in_window"


def test_late_e2_is_retained_but_excluded_from_gating_estimate() -> None:
    sessions = dg.behavior.summarize_behavior_sessions(
        _canonical_published_behavior(),
        _inventory(),
        late_engaged_2_seconds=115,
    )

    assert sessions.get_column("n_engaged_2_early_go_trials").item() == 2
    assert sessions.get_column("n_engaged_2_late_go_trials").item() == 1
    assert sessions.get_column("engaged_2_early_response_probability").item() == 1.0
    assert sessions.get_column("engaged_2_late_response_probability").item() == 1.0
    assert sessions.get_column("n_engaged_2_late_catch_trials").item() == 1
    assert sessions.get_column("engaged_2_late_false_alarm_probability").item() == 1.0
    assert sessions.get_column("engaged_2_late_dprime").item() == 0.0
    assert sessions.get_column("reversible_gating_estimate").item() == 0.0

    changed_late = sessions.with_columns(pl.lit(0.0).alias("engaged_2_late_response_probability"))
    assert changed_late.get_column("reversible_gating_estimate").item() == 0.0


def test_build_tables_returns_tidy_blocks_and_attrition() -> None:
    thresholds = dg.quality.SessionQualityThresholds(
        minimum_go_trials_per_block=3,
        minimum_catch_trials_per_engaged_block=1,
        minimum_engaged_response_rate=0.9,
        minimum_engaged_dprime=0.4,
        maximum_no_reward_response_rate=0.34,
    )
    tables = dg.behavior.build_behavior_tables(
        _canonical_published_behavior(),
        _inventory(),
        thresholds=thresholds,
        late_engaged_2_seconds=0,
    )

    assert tables.session_blocks.select("behavior_block").to_series().to_list() == [
        "engaged_1",
        "no_reward_late",
        "engaged_2_early",
        "engaged_2_late",
    ]
    assert tables.session_blocks.filter(pl.col("is_primary_gating_block")).height == 3
    primary_blocks = tables.session_blocks.filter(pl.col("is_primary_gating_block"))
    assert primary_blocks.get_column("false_alarm_probability").to_list() == [1.0, None, 1.0]
    assert primary_blocks.get_column("n_catch_trials").to_list() == [1, 0, 1]
    assert primary_blocks.get_column("dprime").to_list()[0] == pytest.approx(
        tables.sessions.get_column("engaged_1_dprime").item()
    )
    assert tables.mouse_blocks.get_column("session_cohort").unique().sort().to_list() == [
        "technically_valid",
        "threshold_selected",
    ]
    assert (
        tables.mouse_gating.filter(pl.col("session_cohort") == "technically_valid")
        .get_column("reversible_gating_estimate")
        .item()
        == 0.0
    )
    assert tables.session_attrition.select("attrition_stage", "n_sessions").to_dicts() == [
        {"attrition_stage": "inventory", "n_sessions": 1},
        {"attrition_stage": "technically_valid", "n_sessions": 1},
        {"attrition_stage": "reversible_gating_estimable", "n_sessions": 1},
        {"attrition_stage": "threshold_selected", "n_sessions": 0},
    ]


def test_mouse_block_aggregation_weights_sessions_not_trials() -> None:
    # Arithmetic-only fixture: values are not presented as neuroscience data.
    session_blocks = pl.DataFrame(
        {
            "_nwb_path": ["session-a", "session-b"],
            "subject_id": ["mouse-a", "mouse-a"],
            "behavior_block": ["engaged_1", "engaged_1"],
            "block_order": [1, 1],
            "response_probability": [1.0, 0.0],
            "n_responses": [9, 0],
            "n_go_trials": [9, 1],
            "false_alarm_probability": [0.0, 1.0],
            "n_false_alarms": [0, 1],
            "n_catch_trials": [1, 1],
            "dprime": [1.5, -0.5],
            "is_technically_valid": [True, True],
            "is_good_session": [True, False],
        }
    )

    mouse_blocks = dg.behavior.aggregate_mouse_blocks(session_blocks)
    technical = mouse_blocks.filter(pl.col("session_cohort") == "technically_valid")
    selected = mouse_blocks.filter(pl.col("session_cohort") == "threshold_selected")

    assert technical.get_column("response_probability").item() == 0.5
    assert technical.get_column("n_sessions_contributing").item() == 2
    assert technical.get_column("n_responses").item() == 9
    assert technical.get_column("n_go_trials").item() == 10
    assert technical.get_column("false_alarm_probability").item() == 0.5
    assert technical.get_column("dprime").item() == 0.5
    assert technical.get_column("n_sessions_false_alarm_contributing").item() == 2
    assert technical.get_column("n_sessions_dprime_contributing").item() == 2
    assert selected.get_column("response_probability").item() == 1.0
    assert selected.get_column("n_sessions_contributing").item() == 1


def test_missing_columns_and_duplicate_inventory_sources_fail_fast() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        dg.behavior.summarize_behavior_sessions(
            _canonical_published_trials().drop("response_in_window"),
            _inventory(),
        )

    duplicate_inventory = pl.concat([_inventory(), _inventory()])
    with pytest.raises(ValueError, match="unique"):
        dg.behavior.summarize_behavior_sessions(
            _canonical_published_trials(),
            duplicate_inventory,
        )


def test_inventory_session_without_trials_is_retained_as_technical_attrition() -> None:
    inventory = pl.concat(
        [
            _inventory(),
            pl.DataFrame(
                {
                    "_nwb_path": ["sub-607660/sub-607660_ses-20220607T212534.nwb"],
                    "subject_id": ["607660"],
                }
            ),
        ]
    )
    sessions = dg.behavior.summarize_behavior_sessions(
        _canonical_published_trials(),
        inventory,
        late_engaged_2_seconds=5,
    )
    missing = sessions.filter(pl.col("subject_id") == "607660")

    assert missing.height == 1
    assert not missing.get_column("is_technically_valid").item()
    assert not missing.get_column("is_good_session").item()
    assert missing.get_column("technical_exclusion_reasons").item() == ("missing_behavior_trials")
    assert missing.get_column("reversible_gating_status").item() == "invalid_technical"
