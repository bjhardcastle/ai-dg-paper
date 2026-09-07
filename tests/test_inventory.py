"""Tests for Milestone 0 unit, stimulus, and event-clock inventories."""

import polars as pl
import pytest

import dg.data
import dg.inventory


def _session_inventory(*sources: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": list(sources),
            "dandiset_id": ["001051"] * len(sources),
            "dandiset_version": ["0.260825.2232"] * len(sources),
            "asset_id": [f"asset-{index}" for index in range(len(sources))],
            "path": [f"sub-{index}/session-{index}.nwb" for index in range(len(sources))],
            "subject_id": [f"mouse-{index}" for index in range(len(sources))],
            "ecephys_session_id": [1000 + index for index in range(len(sources))],
            "recording_day": ["EPHYS_1"] * len(sources),
            "session_number": [1] * len(sources),
            "date_of_acquisition": ["2024-01-01"] * len(sources),
            "companion_unit_count": [1] * len(sources),
            "novel_image_id": ["im104_r-1.0"] * len(sources),
        }
    )


def _electrodes(source: str) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "id": [10],
            "location": ["VISp"],
            "x": [8000.0],
            "y": [2000.0],
            "z": [6000.0],
            "probe_id": [70],
            "probe_channel_number": [10],
            "probe_horizontal_position": [20],
            "probe_vertical_position": [300],
            "valid_data": [True],
            "_nwb_path": [source],
            "_table_path": [dg.data.ELECTRODES_PATH],
            "_table_index": [10],
        }
    ).lazy()


def test_unit_inventory_projects_out_arrays_and_retains_qc_anatomy(monkeypatch) -> None:
    source = "session-a"
    sessions = _session_inventory(source)
    scanned_columns: list[object] = []
    units = pl.DataFrame(
        {
            "id": [41],
            "peak_channel_id": [10],
            "isi_violations": [0.2],
            "amplitude_cutoff": [0.05],
            "quality": ["good"],
            "firing_rate": [3.5],
            "presence_ratio": [0.99],
            "snr": [2.0],
            "spike_times": [[1.0, 2.0]],
            "waveform_mean": [[0.0, 1.0]],
            "_nwb_path": [source],
            "_table_path": [dg.data.UNITS_PATH],
            "_table_index": [0],
        }
    )

    def scan_table(sources, table_path, *, columns=None, **kwargs):
        scanned_columns.append(columns)
        if table_path == dg.data.UNITS_PATH:
            return units.lazy()
        assert table_path == dg.data.ELECTRODES_PATH
        return _electrodes(source)

    monkeypatch.setattr(dg.data, "scan_nwb_table", scan_table)

    result = dg.inventory.scan_unit_inventory(sessions).collect()

    assert scanned_columns == [None, None]
    assert "spike_times" not in result.columns
    assert "waveform_mean" not in result.columns
    assert "firing_rate" not in result.columns
    assert result.get_column("unit_key").item() == "asset-0:41"
    assert result.get_column("structure_acronym").item() == "VISp"
    assert result.get_column("probe_id").item() == 70
    assert result.get_column("well_isolated").item()
    assert result.get_column("spike_arrays_intentionally_not_loaded").item()
    assert result.get_column("waveform_duration").null_count() == 1
    dg.inventory.validate_unit_inventory(result, sessions)


def test_unit_reconciliation_reports_companion_discrepancy_without_exclusion() -> None:
    sessions = _session_inventory("session-a").with_columns(pl.lit(2).alias("companion_unit_count"))
    units = pl.DataFrame(
        {
            "unit_key": ["asset-0:41"],
            "unit_id": [41],
            "unit_table_path": [dg.data.UNITS_PATH],
            "unit_table_index": [0],
            "_nwb_path": ["session-a"],
            "asset_id": ["asset-0"],
            "subject_id": ["mouse-0"],
            "well_isolated": [True],
            "unit_quality_exclusion_reasons": [""],
            "peak_channel_id": [10],
            "electrode_table_index": [10],
            "structure_acronym": ["VISp"],
            "probe_id": [70],
        }
    )

    result = dg.inventory.summarize_unit_session_reconciliation(units, sessions)

    assert result.get_column("n_nwb_units").item() == 1
    assert result.get_column("nwb_minus_companion_units").item() == -1
    assert not result.get_column("companion_unit_count_agrees").item()


def test_unit_inventory_fails_when_a_frozen_session_is_missing() -> None:
    sessions = _session_inventory("session-a", "session-b")
    units = pl.DataFrame(
        {
            "unit_key": ["asset-0:41"],
            "unit_id": [41],
            "unit_table_path": [dg.data.UNITS_PATH],
            "unit_table_index": [0],
            "_nwb_path": ["session-a"],
            "asset_id": ["asset-0"],
            "subject_id": ["mouse-0"],
            "well_isolated": [True],
            "unit_quality_exclusion_reasons": [""],
            "peak_channel_id": [10],
            "electrode_table_index": [10],
        }
    )

    with pytest.raises(ValueError, match="source coverage"):
        dg.inventory.validate_unit_inventory(units, sessions)


def test_unit_inventory_fails_on_unmatched_peak_electrode() -> None:
    sessions = _session_inventory("session-a")
    units = pl.DataFrame(
        {
            "unit_key": ["asset-0:41"],
            "unit_id": [41],
            "unit_table_path": [dg.data.UNITS_PATH],
            "unit_table_index": [0],
            "_nwb_path": ["session-a"],
            "asset_id": ["asset-0"],
            "subject_id": ["mouse-0"],
            "well_isolated": [True],
            "unit_quality_exclusion_reasons": [""],
            "peak_channel_id": [10],
            "electrode_table_index": [None],
        }
    )

    with pytest.raises(ValueError, match="matched peak electrode"):
        dg.inventory.validate_unit_inventory(units, sessions)


def _schema_audit(source: str, table_path: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": [source],
            "n_image_presentation_candidates": [1],
            "image_presentation_paths": [table_path],
            "audit_error": [None],
        }
    )


def _presentations(source: str, table_path: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [0, 1],
            "start_time": [1.0, 1.25],
            "stop_time": [1.25, 1.5],
            "start_frame": [60, 75],
            "image_name": ["im001_r-1.0", "im001_r-0.7"],
            "is_change": [False, True],
            "active": [True, True],
            "is_image_novel": [False, False],
            "flashes_since_change": [1.0, 2.0],
            "stimulus_block": [0, 0],
            "rewarded": [False, True],
            "omitted": [False, False],
            "_nwb_path": [source, source],
            "_table_path": [table_path, table_path],
            "_table_index": [0, 1],
        }
    )


def test_task_stimulus_inventory_uses_discovered_path_and_raw_labels(monkeypatch) -> None:
    source = "session-a"
    table_path = "/intervals/task_presentations"
    sessions = _session_inventory(source)
    audit = _schema_audit(source, table_path)
    raw = _presentations(source, table_path)

    def scan_table(sources, observed_path, *, columns=None, **kwargs):
        assert sources == [source]
        assert observed_path == table_path
        assert columns is None
        return raw.lazy()

    monkeypatch.setattr(dg.data, "scan_nwb_table", scan_table)

    presentations = dg.inventory.scan_task_presentations(sessions, audit).collect()
    dg.inventory.validate_task_presentations(presentations, sessions)
    inventory = dg.inventory.summarize_stimulus_inventory(presentations)
    coverage = dg.inventory.summarize_stimulus_coverage(inventory)

    assert presentations.get_column("image_id").to_list() == ["im001", "im001"]
    assert presentations.get_column("image_relative_contrast").to_list() == [1.0, 0.7]
    assert presentations.get_column("start_frame").to_list() == [60, 75]
    assert presentations.get_column("presentation_row_key").to_list() == [
        "asset-0:0",
        "asset-0:1",
    ]
    assert presentations.get_column("within_session_identity_exposure_index").to_list() == [1, 2]
    assert presentations.get_column("stimulus_name").null_count() == 2
    assert inventory.get_column("n_presentations").sum() == 2
    assert coverage.get_column("n_presentations").sum() == 2
    assert coverage.get_column("analysis_scope").unique().to_list() == [
        "coverage_only_no_neural_or_behavioral_outcomes"
    ]
    assert (
        coverage.get_column("interpretation_scope")
        .item(0)
        .startswith("active_and_rewarded_are_presentation_labels")
    )
    with pytest.raises(ValueError, match="presentation IDs must be unique"):
        dg.inventory.validate_task_presentations(
            presentations.with_columns(pl.lit(0).alias("id")),
            sessions,
        )


def test_task_presentation_path_resolution_fails_closed_on_ambiguous_schema() -> None:
    sessions = _session_inventory("session-a")
    audit = _schema_audit("session-a", "/intervals/a;/intervals/b").with_columns(
        pl.lit(2).alias("n_image_presentation_candidates")
    )

    with pytest.raises(ValueError, match="exactly one"):
        dg.inventory.resolve_task_presentation_paths(sessions, audit)


def _trials(source: str = "session-a") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [0, 1, 2],
            "start_time": [0.0, 1.0, 2.0],
            "stop_time": [1.0, 2.0, 3.0],
            "change_time": [0.5, None, 2.5],
            "_nwb_path": [source] * 3,
            "_table_path": [dg.data.TRIALS_PATH] * 3,
            "_table_index": [0, 1, 2],
        }
    )


def test_event_clock_counts_missing_events_but_accepts_valid_intervals() -> None:
    trials = dg.inventory.summarize_interval_clock(
        _trials(),
        table_kind="trials",
        event_time_column="change_time",
    )
    presentations = dg.inventory.summarize_interval_clock(
        _presentations("session-a", "/intervals/task"),
        table_kind="task_presentations",
    )
    sessions = _session_inventory("session-a")

    combined = dg.inventory.combine_event_clock_audits(trials, presentations, sessions)

    assert trials.get_column("n_missing_event_times").item() == 1
    assert trials.get_column("n_event_times_outside_interval").item() == 0
    assert trials.get_column("scalar_interval_clock_pass").item()
    assert combined.get_column("trial_presentation_clock_ranges_overlap").item()
    assert combined.get_column("scalar_interval_clock_screen_pass").item()
    dg.inventory.validate_event_clock_audit(combined, sessions)
    dictionary = dg.inventory.build_event_dictionary(combined)
    assert set(dictionary.get_column("event_name")) == {
        "trial_interval",
        "trial_change_anchor",
        "trial_change_frame_anchor",
        "task_image_presentation_interval",
        "task_presentation_start_frame",
        "raw_trial_lick_timestamps",
    }
    assert dictionary.get_column("scalar_interval_clock_screen_status").unique().to_list() == [
        "pass"
    ]
    lick_row = dictionary.filter(pl.col("event_name") == "raw_trial_lick_timestamps")
    assert lick_row.get_column("field_validation_status").item() == (
        "not_scanned_by_scalar_clock_audit"
    )


def test_event_clock_treats_nan_optional_event_as_reported_missingness() -> None:
    trials = _trials().with_columns(pl.Series("change_time", [0.5, float("nan"), 2.5]))

    result = dg.inventory.summarize_interval_clock(
        trials,
        table_kind="trials",
        event_time_column="change_time",
    )

    assert result.get_column("n_missing_event_times").item() == 1
    assert result.get_column("n_nonfinite_event_times").item() == 0
    assert result.get_column("scalar_interval_clock_pass").item()


def test_event_clock_fails_on_nonmonotonic_and_out_of_interval_times() -> None:
    invalid = _trials().with_columns(
        pl.Series("start_time", [0.0, 2.0, 1.0]),
        pl.Series("stop_time", [1.0, 3.0, 2.0]),
        pl.Series("change_time", [0.5, 3.5, 1.5]),
    )

    result = dg.inventory.summarize_interval_clock(
        invalid,
        table_kind="trials",
        event_time_column="change_time",
    )

    assert result.get_column("n_decreasing_start_times").item() == 1
    assert result.get_column("n_event_times_outside_interval").item() == 1
    assert not result.get_column("scalar_interval_clock_pass").item()


def test_scalar_interval_clock_fails_on_duplicate_start_times() -> None:
    duplicate_start = _trials().with_columns(
        pl.Series("start_time", [0.0, 1.0, 1.0]),
        pl.Series("stop_time", [1.0, 2.0, 2.5]),
        pl.Series("change_time", [0.5, 1.5, 1.5]),
    )

    result = dg.inventory.summarize_interval_clock(
        duplicate_start,
        table_kind="trials",
        event_time_column="change_time",
    )

    assert result.get_column("n_rows_with_duplicate_start_times").item() == 2
    assert not result.get_column("scalar_interval_clock_pass").item()


def test_unit_anatomy_coverage_retains_missing_region_denominator() -> None:
    units = pl.DataFrame(
        {
            "structure_acronym": ["VISp", None],
            "subject_id": ["mouse-a", "mouse-a"],
            "_nwb_path": ["session-a", "session-a"],
            "probe_id": [1, 1],
            "unit_key": ["a:1", "a:2"],
            "well_isolated": [True, False],
        }
    )

    coverage = dg.inventory.summarize_unit_anatomy_coverage(units)

    assert set(coverage.get_column("structure_acronym")) == {"VISp", "[missing]"}
    assert coverage.get_column("n_units").sum() == 2
    assert coverage.get_column("n_well_isolated_units").sum() == 1

    by_session = dg.inventory.summarize_unit_session_anatomy_coverage(units)
    assert by_session.select("subject_id", "_nwb_path").unique().height == 1
    assert by_session.get_column("n_units").sum() == 2
    assert by_session.get_column("analysis_scope").unique().to_list() == [
        "coverage_only_no_neural_activity"
    ]


def _audited_trials(source: str = "session-a") -> pl.DataFrame:
    change_names = [
        "im115_r-0.7",
        "im115_r-1.0",
        "im115_r-0.7",
        "im115_r-1.0",
        "im115_r-0.7",
        "im115_r-1.0",
        None,
    ]
    initial_names = [
        "im115_r-1.0",
        "im115_r-0.7",
        "im115_r-1.0",
        "im115_r-0.7",
        "im115_r-1.0",
        "im115_r-0.7",
        None,
    ]
    return pl.DataFrame(
        {
            "_nwb_path": [source] * 7,
            "_table_path": [dg.data.TRIALS_PATH] * 7,
            "_table_index": list(range(7)),
            "id": list(range(7)),
            "trial_row_key": [f"asset-0:{index}" for index in range(7)],
            "asset_id": ["asset-0"] * 7,
            "subject_id": ["mouse-0"] * 7,
            "ecephys_session_id": [1000] * 7,
            "recording_day": ["EPHYS_1"] * 7,
            "start_time": [float(index) for index in range(7)],
            "stop_time": [index + 0.9 for index in range(7)],
            "change_time": [index + 0.5 for index in range(6)] + [float("nan")],
            "change_frame": [60.0 * (index + 1) for index in range(6)] + [float("nan")],
            "initial_image_name": initial_names,
            "change_image_name": change_names,
            "initial_image_id": ["im115"] * 6 + [None],
            "change_image_id": ["im115"] * 6 + [None],
            "initial_image_relative_contrast": [1.0, 0.7, 1.0, 0.7, 1.0, 0.7, None],
            "change_image_relative_contrast": [0.7, 1.0, 0.7, 1.0, 0.7, 1.0, None],
            "stimulus_token_changed": [True] * 6 + [None],
            "identity_changed": [False] * 6 + [None],
            "contrast_changed": [True] * 6 + [None],
            "aborted": [False] * 6 + [True],
            "go": [True] * 5 + [False, False],
            "catch": [False] * 7,
            "auto_rewarded": [False] * 5 + [True, False],
            "is_change": [True] * 6 + [False],
            "is_sham_change": [False] * 7,
            "nwb_no_reward_epoch": [False, False, True, True, False, False, False],
            "companion_no_reward_epoch": [False, False, True, True, False, False, False],
            "companion_reward_epoch_conflict": [False] * 7,
            "no_reward_epoch": [False, False, True, True, False, False, False],
            "no_reward_epoch_conflict": [False] * 7,
            "no_reward_epoch_source": ["nwb"] * 7,
            "reward_block": [
                "engaged_1",
                "engaged_1",
                "no_reward",
                "no_reward",
                "engaged_2",
                "engaged_2",
                "engaged_2",
            ],
            "reward_block_source": ["audited_trial_no_reward_epoch_not_presentation_labels"] * 7,
        }
    )


def _frame_matched_presentations(trials: pl.DataFrame) -> pl.DataFrame:
    anchors = trials.filter(~pl.col("aborted"))
    return pl.DataFrame(
        {
            "_nwb_path": anchors.get_column("_nwb_path"),
            "_table_path": ["/intervals/task"] * anchors.height,
            "_table_index": list(range(anchors.height)),
            "presentation_row_key": [f"asset-0:p{index}" for index in range(anchors.height)],
            "id": list(range(anchors.height)),
            "start_time": [value + 0.023 for value in anchors.get_column("change_time")],
            "stop_time": [value + 0.273 for value in anchors.get_column("change_time")],
            "start_frame": anchors.get_column("change_frame").cast(pl.Int64),
            "image_name": anchors.get_column("change_image_name"),
            "image_id": anchors.get_column("change_image_id"),
            "image_relative_contrast": anchors.get_column("change_image_relative_contrast"),
            "is_change": anchors.get_column("stimulus_token_changed"),
            "is_image_novel": [False] * anchors.height,
            "omitted": [False] * anchors.height,
        }
    )


def test_trial_inventory_accepts_aborted_missing_anchors_and_rejects_partial() -> None:
    trials = _audited_trials()
    structure = dg.inventory.validate_trial_inventory(trials, _session_inventory("session-a"))

    assert structure.get_column("task_structure_valid").item()
    assert structure.get_column("trial_label_consistency_pass").item()

    partial = trials.with_columns(
        pl.when(pl.col("id") == 6)
        .then(pl.lit(7.0))
        .otherwise(pl.col("change_time"))
        .alias("change_time")
    )
    with pytest.raises(ValueError, match="partial time/frame"):
        dg.inventory.validate_trial_inventory(partial, _session_inventory("session-a"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("catch", True, "task-type label consistency"),
        ("is_change", False, "task-type label consistency"),
        ("is_sham_change", True, "task-type label consistency"),
    ],
)
def test_trial_inventory_rejects_inconsistent_task_type_labels(column, value, message) -> None:
    invalid = _audited_trials().with_columns(
        pl.when(pl.col("id") == 0).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
    )

    with pytest.raises(ValueError, match=message):
        dg.inventory.validate_trial_inventory(invalid, _session_inventory("session-a"))


def test_trial_inventory_rejects_go_auto_rewarded_overlap() -> None:
    invalid = _audited_trials().with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit(True))
        .otherwise(pl.col("auto_rewarded"))
        .alias("auto_rewarded")
    )

    with pytest.raises(ValueError, match="task-type label consistency"):
        dg.inventory.validate_trial_inventory(invalid, _session_inventory("session-a"))


def test_trial_inventory_allows_catch_auto_rewarded_combination() -> None:
    valid = _audited_trials().with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.col("initial_image_name"))
        .otherwise(pl.col("change_image_name"))
        .alias("change_image_name"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(1.0))
        .otherwise(pl.col("change_image_relative_contrast"))
        .alias("change_image_relative_contrast"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(False))
        .otherwise(pl.col("stimulus_token_changed"))
        .alias("stimulus_token_changed"),
        pl.when(pl.col("id") == 0).then(pl.lit(False)).otherwise(pl.col("go")).alias("go"),
        pl.when(pl.col("id") == 0).then(pl.lit(True)).otherwise(pl.col("catch")).alias("catch"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(True))
        .otherwise(pl.col("auto_rewarded"))
        .alias("auto_rewarded"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(False))
        .otherwise(pl.col("is_change"))
        .alias("is_change"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(True))
        .otherwise(pl.col("is_sham_change"))
        .alias("is_sham_change"),
    )

    structure = dg.inventory.validate_trial_inventory(valid, _session_inventory("session-a"))

    assert structure.get_column("trial_label_consistency_pass").item()
    assert structure.get_column("n_catch_auto_rewarded_combinations_descriptive").item() == 1


def test_trial_inventory_rejects_reused_change_frame_anchor() -> None:
    invalid = _audited_trials().with_columns(
        pl.when(pl.col("id") == 2)
        .then(pl.lit(60.0))
        .otherwise(pl.col("change_frame"))
        .alias("change_frame")
    )

    with pytest.raises(ValueError, match="change_frame must be unique"):
        dg.inventory.validate_trial_inventory(invalid, _session_inventory("session-a"))


def test_trial_inventory_rejects_duplicate_dynamic_table_id() -> None:
    invalid = _audited_trials().with_columns(
        pl.when(pl.col("id") == 1).then(pl.lit(0)).otherwise(pl.col("id")).alias("id")
    )

    with pytest.raises(ValueError, match="trial IDs must be unique"):
        dg.inventory.validate_trial_inventory(invalid, _session_inventory("session-a"))


def test_trial_inventory_scan_projects_frames_and_audited_reward_state(monkeypatch) -> None:
    expected = _audited_trials()
    raw = expected.with_columns(
        pl.col("nwb_no_reward_epoch").alias("no_reward_epoch"),
        pl.lit(False).alias("omitted_reward"),
    ).select(
        *dg.inventory.TRIAL_CLOCK_COLUMNS,
        "_nwb_path",
        "_table_path",
        "_table_index",
    )
    companion = expected.select(
        pl.col("ecephys_session_id").alias("session_id"),
        pl.col("id").alias("trials_id"),
        pl.col("companion_no_reward_epoch"),
        pl.lit(1).alias("n_companion_reward_epoch_values"),
        pl.lit(0).alias("companion_csv_first_row"),
        pl.lit(0).alias("companion_csv_last_row"),
        pl.lit(1).alias("n_companion_csv_rows"),
        pl.lit("companion.csv").alias("companion_source"),
        pl.lit(False).alias("companion_reward_epoch_conflict"),
    )

    def scan_trials(sources, *, columns, **kwargs):
        assert sources == ["session-a"]
        assert tuple(columns) == dg.inventory.TRIAL_CLOCK_COLUMNS
        return raw.lazy()

    monkeypatch.setattr(dg.data, "scan_trials", scan_trials)

    result = dg.inventory.scan_trial_inventory(
        _session_inventory("session-a"),
        companion,
    ).collect()

    assert result.get_column("change_frame").dtype == pl.Float64
    assert (
        result.get_column("reward_block").to_list() == expected.get_column("reward_block").to_list()
    )
    assert result.get_column("reward_block_source").unique().to_list() == [
        "audited_trial_no_reward_epoch_not_presentation_labels"
    ]
    assert "lick_times" not in result.columns
    dg.inventory.validate_trial_inventory(result, _session_inventory("session-a"))


@pytest.mark.parametrize("frame_offset", [0, 91_200, 91_201])
def test_frame_alignment_infers_session_local_integer_frame_origin(frame_offset: int) -> None:
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials).with_columns(
        (pl.col("start_frame") + frame_offset).alias("start_frame")
    )

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)
    summary = dg.inventory.validate_trial_presentation_alignment(
        alignment,
        trials,
        presentations,
        _session_inventory("session-a"),
    )

    assert alignment.get_column("trial_presentation_alignment_pass").all()
    assert alignment.get_column("session_frame_offset").unique().to_list() == [frame_offset]
    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [1]
    assert alignment.get_column("frame_offset_inference_pass").all()
    assert alignment.get_column("trial_identity_changed").eq(False).all()
    assert alignment.get_column("trial_stimulus_token_changed").all()
    assert alignment.get_column("trial_change_minus_presentation_start_seconds").to_list() == (
        pytest.approx([-0.023] * 6)
    )
    assert summary.get_column("n_alignment_pass").item() == 6
    assert summary.get_column("session_frame_offset").item() == frame_offset
    assert summary.get_column("n_session_frame_offset_candidates").item() == 1
    assert summary.get_column("frame_offset_inference_pass").item()


def test_frame_alignment_infers_offsets_independently_per_session() -> None:
    trials_a = _audited_trials("session-a")
    trials_b = _audited_trials("session-b").with_columns(
        pl.lit("mouse-1").alias("subject_id"),
        pl.lit(1001, dtype=pl.Int64).alias("ecephys_session_id"),
        pl.concat_str(pl.lit("asset-1:"), pl.col("id")).alias("trial_row_key"),
    )
    presentations_a = _frame_matched_presentations(trials_a).with_columns(
        (pl.col("start_frame") + 91_200).alias("start_frame")
    )
    presentations_b = _frame_matched_presentations(trials_b).with_columns(
        (pl.col("start_frame") + 91_201).alias("start_frame"),
        pl.concat_str(pl.lit("asset-1:p"), pl.col("id")).alias("presentation_row_key"),
    )

    alignment = dg.inventory.align_trials_to_presentations(
        pl.concat([trials_a, trials_b]),
        pl.concat([presentations_a, presentations_b]),
    )
    summary = dg.inventory.validate_trial_presentation_alignment(
        alignment,
        pl.concat([trials_a, trials_b]),
        pl.concat([presentations_a, presentations_b]),
        _session_inventory("session-a", "session-b"),
    )

    assert dict(summary.select("_nwb_path", "session_frame_offset").iter_rows()) == {
        "session-a": 91_200,
        "session-b": 91_201,
    }


def test_frame_alignment_fails_closed_on_ambiguous_frame_origin() -> None:
    trials = _audited_trials()
    base = _frame_matched_presentations(trials)
    shifted = base.with_columns(
        (pl.col("start_frame") + 100_000).alias("start_frame"),
        (pl.col("_table_index") + base.height).alias("_table_index"),
        (pl.col("id") + base.height).alias("id"),
        pl.concat_str(pl.lit("asset-0:shifted-p"), pl.col("id")).alias("presentation_row_key"),
    )
    presentations = pl.concat([base, shifted])

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [2]
    assert alignment.get_column("session_frame_offset").null_count() == alignment.height
    assert not alignment.get_column("frame_offset_inference_pass").any()
    assert alignment.get_column("stimulus_token_agrees").null_count() == alignment.height
    with pytest.raises(ValueError, match="frame-offset inference"):
        dg.inventory.validate_trial_presentation_alignment(
            alignment,
            trials,
            presentations,
            _session_inventory("session-a"),
        )


def test_frame_alignment_fails_closed_when_no_constant_frame_origin_exists() -> None:
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials).with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.col("start_frame") + 1)
        .otherwise(pl.col("start_frame"))
        .alias("start_frame")
    )

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [0]
    assert alignment.get_column("session_frame_offset").null_count() == alignment.height
    assert alignment.get_column("stimulus_token_agrees").null_count() == alignment.height
    assert (
        alignment.get_column("alignment_failure_reasons")
        .str.contains("stimulus_token_mismatch")
        .not_()
        .all()
    )
    with pytest.raises(ValueError, match="frame-offset inference"):
        dg.inventory.validate_trial_presentation_alignment(
            alignment,
            trials,
            presentations,
            _session_inventory("session-a"),
        )


def test_frame_alignment_rejects_nonintegral_trial_anchor() -> None:
    trials = _audited_trials().with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit(60.5))
        .otherwise(pl.col("change_frame"))
        .alias("change_frame")
    )

    with pytest.raises(ValueError, match="must be integral"):
        dg.inventory.align_trials_to_presentations(
            trials,
            _frame_matched_presentations(_audited_trials()),
        )


def test_frame_alignment_checks_trial_time_membership_after_frame_inference() -> None:
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials).with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit(0.0))
        .otherwise(pl.col("start_time"))
        .alias("start_time")
    )

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [1]
    assert alignment.get_column("frame_offset_inference_pass").all()
    assert not alignment.get_column("presentation_start_within_trial_interval").item(0)
    assert "presentation_start_outside_trial_interval" in alignment.get_column(
        "alignment_failure_reasons"
    ).item(0)
    with pytest.raises(ValueError, match="alignment failed"):
        dg.inventory.validate_trial_presentation_alignment(
            alignment,
            trials,
            presentations,
            _session_inventory("session-a"),
        )


def test_frame_alignment_excludes_omitted_rows_from_offset_inference() -> None:
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials).with_columns(
        pl.when(pl.col("id") == 0).then(pl.lit(True)).otherwise(pl.col("omitted")).alias("omitted")
    )

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [0]
    with pytest.raises(ValueError, match="frame-offset inference"):
        dg.inventory.validate_trial_presentation_alignment(
            alignment,
            trials,
            presentations,
            _session_inventory("session-a"),
        )


def test_frame_alignment_allows_catch_with_unchanged_presentation_flag() -> None:
    trials = _audited_trials().with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.col("initial_image_name"))
        .otherwise(pl.col("change_image_name"))
        .alias("change_image_name"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(1.0))
        .otherwise(pl.col("change_image_relative_contrast"))
        .alias("change_image_relative_contrast"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(False))
        .otherwise(pl.col("stimulus_token_changed"))
        .alias("stimulus_token_changed"),
        pl.when(pl.col("id") == 0).then(pl.lit(False)).otherwise(pl.col("go")).alias("go"),
        pl.when(pl.col("id") == 0).then(pl.lit(True)).otherwise(pl.col("catch")).alias("catch"),
        pl.when(pl.col("id") == 0)
        .then(pl.lit(True))
        .otherwise(pl.col("is_sham_change"))
        .alias("is_sham_change"),
    )
    presentations = _frame_matched_presentations(trials)

    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    catch = alignment.filter(pl.col("trial_catch"))
    assert not catch.get_column("presentation_is_change").item()
    assert catch.get_column("trial_is_sham_change").item()
    assert catch.get_column("trial_presentation_alignment_pass").item()


def test_frame_alignment_reports_duplicate_frame_and_image_mismatch() -> None:
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials)
    duplicate = pl.concat([presentations, presentations.head(1)], how="vertical")

    alignment = dg.inventory.align_trials_to_presentations(trials, duplicate)

    assert alignment.get_column("n_session_frame_offset_candidates").unique().to_list() == [0]
    assert alignment.get_column("stimulus_token_agrees").null_count() == alignment.height
    assert not alignment.get_column("trial_presentation_alignment_pass").all()
    with pytest.raises(ValueError, match="start_frame must be unique"):
        dg.inventory.validate_trial_presentation_alignment(
            alignment,
            trials,
            duplicate,
            _session_inventory("session-a"),
        )

    mismatched = presentations.with_columns(
        pl.when(pl.col("id") == 0)
        .then(pl.lit("im999"))
        .otherwise(pl.col("image_id"))
        .alias("image_id")
    )
    mismatch_alignment = dg.inventory.align_trials_to_presentations(trials, mismatched)
    assert "image_identity_mismatch" in mismatch_alignment.get_column(
        "alignment_failure_reasons"
    ).item(0)


def _semantic_presentations(source: str = "session-a") -> pl.DataFrame:
    image_names = [
        "im115_r-0.7",
        "im115_r-1.0",
        "im104_r-1.0",
    ]
    return pl.DataFrame(
        {
            "_nwb_path": [source] * 3,
            "subject_id": ["mouse-0"] * 3,
            "ecephys_session_id": [1000] * 3,
            "recording_day": ["EPHYS_1"] * 3,
            "image_name": image_names,
            "image_id": ["im115", "im115", "im104"],
            "image_relative_contrast": [0.7, 1.0, 1.0],
            "is_image_novel": [False, False, True],
            "omitted": [False] * 3,
            "metadata_novel_image_id": ["im104"] * 3,
            "metadata_novel_relative_contrast": [1.0] * 3,
            "is_metadata_designated_novel_identity": [False, False, True],
            "within_session_identity_exposure_index": [1, 2, 1],
        }
    )


def test_stimulus_semantics_rejects_novel_metadata_disagreement() -> None:
    trials = _audited_trials()
    alignment = dg.inventory.align_trials_to_presentations(
        trials,
        _frame_matched_presentations(trials),
    )
    valid = dg.inventory.summarize_stimulus_semantics(
        _semantic_presentations(),
        alignment,
    )
    dg.inventory.validate_stimulus_semantics(valid, _session_inventory("session-a"))
    assert valid.get_column("stimulus_semantics_internal_consistency_pass").item()

    invalid_presentations = _semantic_presentations().with_columns(
        pl.when(pl.col("image_id") == "im104")
        .then(pl.lit(False))
        .otherwise(pl.col("is_image_novel"))
        .alias("is_image_novel")
    )
    invalid = dg.inventory.summarize_stimulus_semantics(invalid_presentations, alignment)
    with pytest.raises(ValueError, match="internal consistency"):
        dg.inventory.validate_stimulus_semantics(invalid, _session_inventory("session-a"))


def test_contrast_estimability_uses_audited_trial_reward_blocks() -> None:
    sessions = _session_inventory("session-a")
    trials = _audited_trials()
    presentations = _frame_matched_presentations(trials)
    alignment = dg.inventory.align_trials_to_presentations(trials, presentations)

    estimability = dg.inventory.summarize_stimulus_estimability(alignment, sessions)
    dg.inventory.validate_stimulus_estimability(estimability, sessions)
    aliases = dg.inventory.build_stimulus_design_aliases(
        _semantic_presentations(),
        estimability,
    )
    dg.inventory.validate_stimulus_design_aliases(aliases)

    assert estimability.get_column("both_im115_contrasts_present").all()
    assert estimability.get_column("reward_state_source").unique().to_list() == [
        "audited_trial_reward_block_at_frame_aligned_anchor"
    ]
    assert (
        aliases.filter(pl.col("estimand") == "state_x_contrast_within_im115")
        .get_column("rank_status")
        .item()
        == "estimable"
    )
    assert (
        aliases.filter(pl.col("estimand") == "pure_novelty_main_effect")
        .get_column("rank_status")
        .item()
        == "unidentifiable"
    )


def test_stimulus_alias_statuses_are_conditional_on_observed_design_evidence() -> None:
    sessions = _session_inventory("session-a")
    trials = _audited_trials()
    alignment = dg.inventory.align_trials_to_presentations(
        trials, _frame_matched_presentations(trials)
    )
    estimability = dg.inventory.summarize_stimulus_estimability(alignment, sessions)
    presentations = _semantic_presentations()
    familiar_same_identity = presentations.filter(pl.col("image_id") == "im104").with_columns(
        pl.lit(False).alias("is_metadata_designated_novel_identity")
    )
    identity_overlap = dg.inventory.build_stimulus_design_aliases(
        pl.concat([presentations, familiar_same_identity]),
        estimability,
    )
    dg.inventory.validate_stimulus_design_aliases(identity_overlap)
    assert (
        identity_overlap.filter(pl.col("estimand") == "pure_novelty_main_effect")
        .get_column("rank_status")
        .item()
        == "not_established_from_inventory"
    )

    low_contrast_novel = presentations.with_columns(
        pl.when(pl.col("is_metadata_designated_novel_identity"))
        .then(pl.lit(0.7))
        .otherwise(pl.col("image_relative_contrast"))
        .alias("image_relative_contrast")
    )
    contrast_overlap = dg.inventory.build_stimulus_design_aliases(
        low_contrast_novel,
        estimability,
    )
    dg.inventory.validate_stimulus_design_aliases(contrast_overlap)
    assert (
        contrast_overlap.filter(pl.col("estimand") == "novelty_x_contrast")
        .get_column("rank_status")
        .item()
        == "not_established_from_inventory"
    )
