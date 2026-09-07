"""Focused tests for the behavior orchestration script.

The small numeric frames are arithmetic contract fixtures, not synthetic
neuroscience observations and not estimates from DANDI:001051.
"""

import argparse
import datetime
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

import dg.behavior
import dg.data
import dg.statistics

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "01_analyze_behavior.py"
SPEC = importlib.util.spec_from_file_location("dg_analyze_behavior_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ANALYZE_BEHAVIOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE_BEHAVIOR
SPEC.loader.exec_module(ANALYZE_BEHAVIOR)


def test_completed_audit_requires_task_parameter_pass(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "audit_run.json"
    record = {
        "run_status": "complete",
        "authoritative": True,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "asset_inventory_status": "pass",
        "schema_audit_status": "pass",
        "task_parameters_audit_status": "skipped_by_request",
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="task-parameter audit has not passed"):
        ANALYZE_BEHAVIOR._read_completed_audit(path)

    record["task_parameters_audit_status"] = "pass"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert ANALYZE_BEHAVIOR._read_completed_audit(path) == record


def test_completed_audit_rejects_stale_or_in_progress_marker(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "audit_run.json"
    record = {
        "run_status": "in_progress",
        "authoritative": False,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "asset_inventory_status": "pass",
        "schema_audit_status": "pass",
        "task_parameters_audit_status": "pass",
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="completed authoritative"):
        ANALYZE_BEHAVIOR._read_completed_audit(path)


def test_completed_audit_binds_exact_behavior_inputs(tmp_path: pathlib.Path) -> None:
    results_root = tmp_path / "results"
    manifests = results_root / "manifests"
    manifests.mkdir(parents=True)
    inventory = manifests / "session_inventory.parquet"
    task_parameters = manifests / "task_parameters_audit.csv"
    inventory.write_bytes(b"content-addressed inventory")
    task_parameters.write_bytes(b"content-addressed task parameters")

    def artifact_record(path: pathlib.Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(results_root)),
            "sha256": ANALYZE_BEHAVIOR._sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    audit_path = manifests / "audit_run.json"
    record = {
        "manifest_schema_version": ANALYZE_BEHAVIOR.M0_MANIFEST_SCHEMA_VERSION,
        "run_status": "complete",
        "authoritative": True,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "asset_inventory_status": "pass",
        "schema_audit_status": "pass",
        "task_parameters_audit_status": "pass",
        "artifacts": {
            "session_inventory": artifact_record(inventory),
            "task_parameters_audit": artifact_record(task_parameters),
        },
    }
    audit_path.write_text(json.dumps(record), encoding="utf-8")
    artifact_paths = {
        "session_inventory": inventory,
        "task_parameters_audit": task_parameters,
    }

    assert (
        ANALYZE_BEHAVIOR._read_completed_audit(
            audit_path,
            artifact_paths=artifact_paths,
        )
        == record
    )

    inventory.write_bytes(b"changed after M0")
    with pytest.raises(RuntimeError, match="content differs"):
        ANALYZE_BEHAVIOR._read_completed_audit(
            audit_path,
            artifact_paths=artifact_paths,
        )


def test_behavior_main_replaces_stale_success_with_failed_marker(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    manifest_path = results_root / "manifests" / "behavior_analysis_run.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"analysis_status":"pass"}', encoding="utf-8")
    arguments = argparse.Namespace(
        results_root=results_root,
        analysis_lock=tmp_path / "lock.json",
        seed=1051,
        bootstrap_resamples=10,
        sign_flip_resamples=10,
        reuse_behavior_trials=False,
    )
    monkeypatch.setattr(ANALYZE_BEHAVIOR, "parse_arguments", lambda: arguments)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(ANALYZE_BEHAVIOR, "_run_analysis", interrupt)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        ANALYZE_BEHAVIOR.main()

    marker = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert marker["analysis_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "KeyboardInterrupt"


def test_task_parameter_audit_requires_exact_inventory_coverage(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "task_parameters_audit.csv"
    audit = pl.DataFrame(
        {
            "asset_id": ["asset-a", "asset-b"],
            "path": ["a.nwb", "b.nwb"],
            "response_window_start_seconds": [0.15, 0.15],
            "response_window_stop_seconds": [0.75, 0.75],
            "audit_error": [None, None],
        }
    )
    audit.write_csv(path)
    inventory = pl.DataFrame({"asset_id": ["asset-b", "asset-a"]})

    digest = ANALYZE_BEHAVIOR._read_validated_task_parameters_audit(path, inventory)

    assert len(digest) == 64
    with pytest.raises(RuntimeError, match="asset coverage differs"):
        ANALYZE_BEHAVIOR._read_validated_task_parameters_audit(
            path,
            pl.DataFrame({"asset_id": ["asset-a"]}),
        )


def _behavior_tables_fixture() -> dg.behavior.BehaviorTables:
    mouse_gating = pl.DataFrame(
        {
            "subject_id": ["fixture_mouse_1", "fixture_mouse_2"],
            "reversible_gating_estimate": [0.4, 0.2],
            "withdrawal_suppression_estimate": [0.5, 0.3],
            "restoration_recovery_estimate": [0.3, 0.1],
            "engaged_response_drift_estimate": [-0.2, -0.2],
            "reversible_specificity_gating_estimate": [0.3, 0.1],
            "withdrawal_specificity_suppression_estimate": [0.4, 0.2],
            "restoration_specificity_recovery_estimate": [0.2, 0.0],
            "engaged_specificity_drift_estimate": [-0.2, -0.2],
            "n_sessions_contributing": [2, 1],
            "n_sessions_specificity_contributing": [2, 1],
            "n_sessions_inventory": [2, 1],
            "n_sessions_in_cohort": [2, 1],
            "session_cohort": ["technically_valid", "technically_valid"],
            "aggregation_method": ["equal_session_mean", "equal_session_mean"],
        }
    )
    block_rows = []
    for block_order, block in enumerate(dg.behavior.BEHAVIOR_BLOCKS, start=1):
        for mouse_index, subject_id in enumerate(("fixture_mouse_1", "fixture_mouse_2")):
            block_rows.append(
                {
                    "subject_id": subject_id,
                    "behavior_block": block,
                    "block_order": block_order,
                    "response_probability": 0.8 - 0.2 * (block_order == 2) - 0.1 * mouse_index,
                    "n_sessions_response_contributing": 1,
                    "n_responses": 8,
                    "n_go_trials": 10,
                    "false_alarm_probability": 0.1 + 0.05 * mouse_index,
                    "n_sessions_false_alarm_contributing": 1,
                    "n_false_alarms": 1,
                    "n_catch_trials": 10,
                    "dprime": 1.5 - 0.1 * mouse_index,
                    "n_sessions_dprime_contributing": 1,
                    "n_sessions_inventory": 1,
                    "n_sessions_in_cohort": 1,
                    "session_cohort": "technically_valid",
                    "aggregation_method": "equal_session_mean",
                    "n_sessions_contributing": 1,
                }
            )
    sessions = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session_1", "fixture_session_2", "fixture_session_3"],
            "subject_id": ["fixture_mouse_1", "fixture_mouse_1", "fixture_mouse_2"],
            "is_technically_valid": [True, True, True],
            "reversible_gating_estimable": [True, True, True],
            "n_engaged_1_go_trials": [2, 3, 5],
            "n_engaged_1_catch_trials": [29, 31, 37],
            "n_no_reward_late_go_trials": [7, 11, 13],
            "n_no_reward_late_catch_trials": [41, 43, 47],
            "n_engaged_2_early_go_trials": [17, 19, 23],
            "n_engaged_2_early_catch_trials": [53, 59, 61],
            "reversible_specificity_gating_estimate": [0.3, 0.25, 0.1],
            "engaged_1_response_label_agreement_rate": [0.9, 0.8, 0.95],
            "engaged_2_response_label_agreement_rate": [0.9, 0.85, 0.95],
            "n_engaged_1_response_label_comparisons": [10, 10, 10],
            "n_engaged_2_response_label_comparisons": [10, 10, 10],
        }
    )
    attrition = pl.DataFrame(
        {
            "stage_order": [0, 1, 2, 3],
            "attrition_stage": [
                "inventory",
                "technically_valid",
                "reversible_gating_estimable",
                "threshold_selected",
            ],
            "criterion_type": ["inventory", "technical", "estimability", "behavior_threshold"],
            "n_sessions": [3, 3, 3, 2],
            "n_mice": [2, 2, 2, 2],
            "n_sessions_removed_at_stage": [0, 0, 0, 1],
            "n_sessions_removed_from_inventory": [0, 0, 0, 1],
            "fraction_of_inventory": [1.0, 1.0, 1.0, 2 / 3],
        }
    )
    return dg.behavior.BehaviorTables(
        sessions=sessions,
        session_blocks=pl.DataFrame(),
        mouse_blocks=pl.DataFrame(block_rows),
        mouse_gating=mouse_gating,
        session_attrition=attrition,
    )


def _session_qc_fixture() -> pl.DataFrame:
    values = {
        "_nwb_path": ["fixture_session"],
        "subject_id": ["fixture_mouse"],
        "ecephys_session_id": [1],
        "recording_day": ["EPHYS_1"],
        "is_technically_valid": [True],
        "is_good_session": [False],
    }
    for block in ("engaged_1", "no_reward", "engaged_2"):
        values.update(
            {
                f"n_{block}_response_label_trials": [10],
                f"n_{block}_response_label_missing": [1],
                f"n_{block}_online_outcome_absent": [1],
                f"n_{block}_online_outcome_conflicts": [0],
                f"n_{block}_canonical_response_missing": [0],
                f"n_{block}_response_label_comparisons": [9],
                f"n_{block}_response_label_disagreements": [2],
                f"{block}_response_label_agreement_rate": [7 / 9],
            }
        )
    return pl.DataFrame(values)


def test_behavior_statistics_use_the_technical_mouse_cohort() -> None:
    result = ANALYZE_BEHAVIOR.build_behavior_statistics(
        _behavior_tables_fixture(),
        pending_decisions=["D01", "D02", "D03"],
        code_version="arithmetic-test-fixture",
        seed=1051,
        bootstrap_resamples=200,
        sign_flip_resamples=200,
    )

    assert result.schema == dg.statistics.STATISTICS_SCHEMA
    assert result.height == 26
    symmetric_summary = result.filter(
        pl.col("result_id") == ANALYZE_BEHAVIOR.SYMMETRIC_SUMMARY_RESULT_ID
    ).row(0, named=True)
    assert symmetric_summary["estimate"] == pytest.approx(0.3)
    assert symmetric_summary["n_mice"] == 2
    assert symmetric_summary["n_sessions"] == 3
    assert symmetric_summary["n_trials"] == 100
    assert symmetric_summary["p_value"] is None
    assert symmetric_summary["adjusted_p_value"] is None
    assert symmetric_summary["status"] == "fragile"
    assert symmetric_summary["reason"] == ("analysis-lock decisions remain proposed: D01, D02, D03")
    withdrawal = result.filter(pl.col("result_id") == "withdrawal_suppression_mouse_mean").row(
        0, named=True
    )
    restoration = result.filter(pl.col("result_id") == "restoration_recovery_mouse_mean").row(
        0, named=True
    )
    drift = result.filter(pl.col("result_id") == "engaged_response_drift_mouse_mean").row(
        0, named=True
    )
    assert withdrawal["estimate"] == pytest.approx(0.4)
    assert restoration["estimate"] == pytest.approx(0.2)
    assert withdrawal["n_trials"] == 41
    assert restoration["n_trials"] == 90
    assert drift["n_trials"] == 69
    assert withdrawal["multiplicity_family"] == "figure_1_primary_behavior"
    assert restoration["multiplicity_family"] == "figure_1_primary_behavior"
    assert withdrawal["adjusted_p_value"] == pytest.approx(1.0)
    assert restoration["adjusted_p_value"] == pytest.approx(1.0)
    assert symmetric_summary["inclusion_definition"] == ANALYZE_BEHAVIOR.INCLUSION_DEFINITION
    assert result.filter(pl.col("result_id").str.contains("no_reward_raw_lick")).is_empty()

    specificity_counts = {
        result_id: result.filter(pl.col("result_id") == result_id).item(0, "n_trials")
        for result_id in (
            "reversible_specificity_gating_mouse_mean",
            "withdrawal_specificity_suppression_mouse_mean",
            "restoration_specificity_recovery_mouse_mean",
            "engaged_specificity_drift_mouse_mean",
        )
    }
    assert specificity_counts == {
        "reversible_specificity_gating_mouse_mean": 501,
        "withdrawal_specificity_suppression_mouse_mean": 269,
        "restoration_specificity_recovery_mouse_mean": 394,
        "engaged_specificity_drift_mouse_mean": 339,
    }


def test_response_label_audit_limits_interpretation_to_reward_available_blocks() -> None:
    audit = ANALYZE_BEHAVIOR.build_response_label_audit(_session_qc_fixture())

    assert audit.get_column("reward_block").to_list() == [
        "engaged_1",
        "no_reward",
        "engaged_2",
    ]
    assert audit.get_column("online_labels_interpretable").to_list() == [True, False, True]
    no_reward = audit.filter(pl.col("reward_block") == "no_reward").row(0, named=True)
    assert no_reward["interpretation_scope"] == ("audit_only_online_outcomes_not_state_independent")
    assert no_reward["n_label_disagreements"] == 2
    assert no_reward["n_online_outcome_absent"] == 1


def test_session_block_timing_uses_observed_trial_boundaries() -> None:
    trials = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"] * 6,
            "reward_block": [
                "engaged_1",
                "engaged_1",
                "no_reward",
                "no_reward",
                "engaged_2",
                "engaged_2",
            ],
            "start_time": [10.0, 12.0, 20.0, 23.0, 30.0, 34.0],
            "stop_time": [11.0, 13.0, 21.0, 24.0, 31.0, 35.0],
            "_table_index": [0, 1, 2, 3, 4, 5],
        }
    )

    timing = ANALYZE_BEHAVIOR.build_session_block_timing(trials, _session_qc_fixture())

    assert timing.select(
        "reward_block",
        "block_start_from_session_seconds",
        "block_stop_from_session_seconds",
        "n_trials",
    ).to_dicts() == [
        {
            "reward_block": "engaged_1",
            "block_start_from_session_seconds": 0.0,
            "block_stop_from_session_seconds": 3.0,
            "n_trials": 2,
        },
        {
            "reward_block": "no_reward",
            "block_start_from_session_seconds": 10.0,
            "block_stop_from_session_seconds": 14.0,
            "n_trials": 2,
        },
        {
            "reward_block": "engaged_2",
            "block_start_from_session_seconds": 20.0,
            "block_stop_from_session_seconds": 25.0,
            "n_trials": 2,
        },
    ]


def test_session_metadata_join_accepts_long_form_block_rows() -> None:
    inventory = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"],
            "subject_id": ["fixture_mouse"],
            "ecephys_session_id": [1],
        }
    )
    block_rows = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session", "fixture_session"],
            "subject_id": ["fixture_mouse", "fixture_mouse"],
            "behavior_block": ["engaged_1", "no_reward_late"],
        }
    )

    joined = ANALYZE_BEHAVIOR._attach_session_metadata(
        block_rows,
        inventory,
        join_validation="1:m",
    )

    assert joined.height == 2
    assert joined.get_column("ecephys_session_id").to_list() == [1, 1]


def test_behavior_trial_checkpoint_requires_exact_inventory_coverage(
    tmp_path: pathlib.Path,
) -> None:
    # Structural checkpoint fixture, not a neuroscience observation.
    row = {column: None for column in dg.data.BEHAVIOR_TRIAL_COLUMNS}
    row.update(
        {
            "id": 1,
            "_nwb_path": "fixture_session",
            "_table_index": 0,
            "dandiset_version": dg.data.DANDISET_VERSION,
        }
    )
    path = tmp_path / "behavior_trials.parquet"
    pl.DataFrame([row], infer_schema_length=None).write_parquet(path)
    inventory = pl.DataFrame({"_nwb_path": ["fixture_session"]})

    result = ANALYZE_BEHAVIOR._read_behavior_trials_checkpoint(path, inventory)

    assert result.height == 1
    with pytest.raises(RuntimeError, match="coverage differs"):
        ANALYZE_BEHAVIOR._read_behavior_trials_checkpoint(
            path,
            pl.DataFrame({"_nwb_path": ["different_session"]}),
        )


def _trusted_checkpoint_fixture(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Write small structural files and a direct-run trust manifest."""

    behavior_trials_path = tmp_path / "behavior_trials.parquet"
    pl.DataFrame({"_nwb_path": ["fixture_session"], "_table_index": [0]}).write_parquet(
        behavior_trials_path
    )
    session_inventory_path = tmp_path / "session_inventory.parquet"
    pl.DataFrame({"_nwb_path": ["fixture_session"]}).write_parquet(session_inventory_path)
    companion_trials_path = tmp_path / "master_stim_trials_table.csv"
    companion_trials_path.write_text("ecephys_session_id,stimulus_presentations_id\n1,0\n")
    companion_record = ANALYZE_BEHAVIOR._file_provenance_record(companion_trials_path)
    checkpoint_path = tmp_path / ANALYZE_BEHAVIOR.BEHAVIOR_TRIALS_CHECKPOINT_FILENAME
    ANALYZE_BEHAVIOR._write_behavior_trials_checkpoint_manifest(
        checkpoint_path,
        behavior_trials_path=behavior_trials_path,
        session_inventory_path=session_inventory_path,
        companion_trials_path=companion_trials_path,
        companion_provenance={
            "sha256": companion_record["sha256"],
            "content_size_bytes": companion_record["size_bytes"],
            "source_url": "https://example.invalid/pinned-companion.csv",
            "repository_commit": "fixture-commit",
        },
        source_provenance=ANALYZE_BEHAVIOR._analysis_source_provenance(),
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        trial_input_mode="remote_nwb",
    )
    return {
        "checkpoint": checkpoint_path,
        "behavior_trials": behavior_trials_path,
        "session_inventory": session_inventory_path,
        "companion_trials": companion_trials_path,
    }


def test_trusted_behavior_checkpoint_validates_without_refreshing(
    tmp_path: pathlib.Path,
) -> None:
    paths = _trusted_checkpoint_fixture(tmp_path)
    before = paths["checkpoint"].read_bytes()

    checkpoint, manifest_record = (
        ANALYZE_BEHAVIOR._read_validated_behavior_trials_checkpoint_manifest(
            paths["checkpoint"],
            behavior_trials_path=paths["behavior_trials"],
            session_inventory_path=paths["session_inventory"],
            companion_trials_path=paths["companion_trials"],
        )
    )

    assert checkpoint["trial_input_mode"] == "remote_nwb"
    assert checkpoint["dandiset_version"] == dg.data.DANDISET_VERSION
    assert checkpoint["behavior_trials"]["sha256"] == ANALYZE_BEHAVIOR._sha256_file(
        paths["behavior_trials"]
    )
    assert manifest_record["sha256"] == ANALYZE_BEHAVIOR._sha256_file(paths["checkpoint"])
    assert paths["checkpoint"].read_bytes() == before


def test_trusted_behavior_checkpoint_is_required_for_reuse(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        ANALYZE_BEHAVIOR._read_validated_behavior_trials_checkpoint_manifest(
            tmp_path / "missing.json",
            behavior_trials_path=tmp_path / "behavior_trials.parquet",
            session_inventory_path=tmp_path / "session_inventory.parquet",
            companion_trials_path=tmp_path / "companion.csv",
        )


@pytest.mark.parametrize(
    ("changed_file", "error_label"),
    [
        ("behavior_trials", "behavior_trials"),
        ("session_inventory", "session_inventory"),
        ("companion_trials", "companion_trials"),
    ],
)
def test_trusted_behavior_checkpoint_rejects_changed_data_files(
    tmp_path: pathlib.Path,
    changed_file: str,
    error_label: str,
) -> None:
    paths = _trusted_checkpoint_fixture(tmp_path)
    paths[changed_file].write_bytes(paths[changed_file].read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match=error_label):
        ANALYZE_BEHAVIOR._read_validated_behavior_trials_checkpoint_manifest(
            paths["checkpoint"],
            behavior_trials_path=paths["behavior_trials"],
            session_inventory_path=paths["session_inventory"],
            companion_trials_path=paths["companion_trials"],
        )


def test_trusted_behavior_checkpoint_rejects_wrong_dandiset_version(
    tmp_path: pathlib.Path,
) -> None:
    paths = _trusted_checkpoint_fixture(tmp_path)
    checkpoint = json.loads(paths["checkpoint"].read_text())
    checkpoint["dandiset_version"] = "different-version"
    paths["checkpoint"].write_text(json.dumps(checkpoint))

    with pytest.raises(RuntimeError, match="Dandiset version"):
        ANALYZE_BEHAVIOR._read_validated_behavior_trials_checkpoint_manifest(
            paths["checkpoint"],
            behavior_trials_path=paths["behavior_trials"],
            session_inventory_path=paths["session_inventory"],
            companion_trials_path=paths["companion_trials"],
        )


def test_trusted_behavior_checkpoint_cannot_be_written_from_reuse(
    tmp_path: pathlib.Path,
) -> None:
    paths = _trusted_checkpoint_fixture(tmp_path)
    companion_record = ANALYZE_BEHAVIOR._file_provenance_record(paths["companion_trials"])

    with pytest.raises(RuntimeError, match="direct remote-NWB run"):
        ANALYZE_BEHAVIOR._write_behavior_trials_checkpoint_manifest(
            tmp_path / "reuse-checkpoint.json",
            behavior_trials_path=paths["behavior_trials"],
            session_inventory_path=paths["session_inventory"],
            companion_trials_path=paths["companion_trials"],
            companion_provenance={
                "sha256": companion_record["sha256"],
                "content_size_bytes": companion_record["size_bytes"],
            },
            source_provenance=ANALYZE_BEHAVIOR._analysis_source_provenance(),
            created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
            trial_input_mode="validated_existing_behavior_trials",
        )
    assert not (tmp_path / "reuse-checkpoint.json").exists()


def test_reward_epoch_audit_distinguishes_nwb_and_companion_sources() -> None:
    trials = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"] * 3,
            "no_reward_epoch": [False, True, True],
            "nwb_no_reward_epoch": [False, None, None],
            "companion_no_reward_epoch": [False, True, True],
            "companion_reward_epoch_conflict": [False, False, False],
            "no_reward_epoch_conflict": [False, False, False],
            "no_reward_epoch_source": ["nwb", "companion", "companion"],
        }
    )
    inventory = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"],
            "subject_id": ["fixture_mouse"],
            "ecephys_session_id": [1],
            "recording_day": ["EPHYS_1"],
        }
    )

    audit = ANALYZE_BEHAVIOR.build_reward_epoch_audit(trials, inventory).row(0, named=True)

    assert audit["n_nwb_source_trials"] == 1
    assert audit["n_companion_source_trials"] == 2
    assert audit["n_unresolved_trials"] == 0
    assert audit["n_effective_reward_epoch_conflicts"] == 0
    assert audit["reward_epoch_audit_status"] == "pass"


def test_lick_timestamp_audit_reports_duplicates_by_session_and_block() -> None:
    trials = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"] * 3,
            "reward_block": ["engaged_1", "engaged_1", "no_reward"],
            "lick_times_valid": [True, True, False],
            "response_event_time_valid": [True, True, False],
            "n_raw_lick_timestamps": [4, 2, None],
            "n_unique_lick_timestamps": [3, 2, None],
            "n_duplicate_lick_timestamps": [1, 0, None],
            "n_raw_response_window_licks": [3, 0, None],
            "n_unique_response_window_licks": [2, 0, None],
            "n_duplicate_response_window_licks": [1, 0, None],
            "n_response_window_licks": [2, 0, None],
            "response_in_window": [True, False, None],
        }
    )
    inventory = pl.DataFrame(
        {
            "_nwb_path": ["fixture_session"],
            "subject_id": ["fixture_mouse"],
            "ecephys_session_id": [1],
            "recording_day": ["EPHYS_1"],
        }
    )

    audit = ANALYZE_BEHAVIOR.build_lick_timestamp_audit(trials, inventory)
    session = audit.filter(pl.col("audit_scope") == "session").row(0, named=True)
    engaged = audit.filter(pl.col("reward_block") == "engaged_1").row(0, named=True)

    assert audit.height == 3
    assert session["n_trials"] == 3
    assert session["n_trials_with_invalid_lick_vectors"] == 1
    assert session["n_trials_without_valid_response_event"] == 1
    assert session["n_raw_lick_timestamps"] == 6
    assert session["n_unique_lick_timestamps"] == 5
    assert session["n_duplicate_lick_timestamps"] == 1
    assert session["n_trials_with_duplicate_lick_timestamps"] == 1
    assert session["n_raw_response_window_licks"] == 3
    assert session["n_unique_response_window_licks"] == 2
    assert session["n_response_window_licks"] == 2
    assert session["canonical_response_count_is_deduplicated"]
    assert session["lick_timestamp_audit_status"] == "invalid_vectors_reported"
    assert engaged["lick_timestamp_audit_status"] == "duplicates_reported"
