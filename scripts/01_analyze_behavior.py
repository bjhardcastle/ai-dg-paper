# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Build the real-data behavior tables and exploratory Figure 1 statistics."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.audit  # noqa: E402
import dg.behavior  # noqa: E402
import dg.data  # noqa: E402
import dg.gates  # noqa: E402
import dg.quality  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "figure_1_behavior"
ANALYSIS_TIER = "exploratory"
BEHAVIOR_DECISIONS = ("D01", "D02", "D03")
SYMMETRIC_SUMMARY_RESULT_ID = "reversible_gating_mouse_mean"
BEHAVIOR_TRIALS_CHECKPOINT_FILENAME = "behavior_trials_checkpoint.json"
BEHAVIOR_TRIALS_CHECKPOINT_SCHEMA_VERSION = 1
M0_MANIFEST_SCHEMA_VERSION = 3
LOCAL_ANALYSIS_MODULE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/audit.py",
    "src/dg/behavior.py",
    "src/dg/data.py",
    "src/dg/gates.py",
    "src/dg/quality.py",
    "src/dg/statistics.py",
)
DEFAULT_SEED = 1051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
DERIVED_TRIAL_COLUMNS = (
    "physical_image_change",
    "reward_block",
    "response_in_window",
    "response_latency_from_licks",
    "n_response_window_licks",
    "n_raw_response_window_licks",
    "n_unique_response_window_licks",
    "n_duplicate_response_window_licks",
    "n_raw_lick_timestamps",
    "n_unique_lick_timestamps",
    "n_duplicate_lick_timestamps",
    "lick_times_valid",
    "response_event_time_valid",
    "lick_response_status",
    "response_window_start_seconds",
    "response_window_stop_seconds",
)
INCLUSION_DEFINITION = (
    "technically valid sessions with complete estimable E1, late-NR, and early-E2 "
    "primary blocks; no behavioral-threshold selection"
)
BLOCK_INCLUSION_DEFINITION = (
    "all technically valid sessions with an estimable block metric; "
    "no behavioral-threshold selection"
)
SPECIFICITY_INCLUSION_DEFINITION = (
    "technically valid sessions with complete estimable E1, late-NR, and early-E2 "
    "change and catch response probabilities; no behavioral-threshold selection"
)
ATTRITION_INCLUSION_DEFINITION = "all sessions in the immutable DANDI session inventory"
AGGREGATION = "equal-session mean within mouse; equal-mouse mean"


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments, with repository-relative safe defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Artifact root produced by 00_freeze_and_audit.py",
    )
    parser.add_argument(
        "--analysis-lock",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
        help="Human-reviewed analysis-lock path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--sign-flip-resamples",
        type=int,
        default=DEFAULT_SIGN_FLIP_RESAMPLES,
    )
    parser.add_argument(
        "--reuse-behavior-trials",
        action="store_true",
        help=(
            "Reuse an existing tables/behavior_trials.parquet only when its trusted "
            "direct-run checkpoint matches the file, inventory, and companion table; "
            "all derived response fields are recomputed."
        ),
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Publish a fail-closed run marker around the behavior analysis."""

    arguments = parse_arguments()
    _validate_analysis_arguments(arguments)
    started_at = datetime.datetime.now(datetime.UTC)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    run_manifest_path = result_root / "manifests" / "behavior_analysis_run.json"
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "authoritative": False,
        "script": str(pathlib.Path(__file__).relative_to(REPOSITORY_ROOT)),
        "started_at_utc": started_at.isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
    }
    dg.artifacts.write_json(in_progress, run_manifest_path)
    try:
        _run_analysis(arguments, started_at=started_at, result_root=result_root)
    except BaseException as error:
        failed = {
            **in_progress,
            "analysis_status": "failed",
            "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(failed, run_manifest_path)
        raise


def _run_analysis(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    result_root: pathlib.Path,
) -> None:
    """Run the complete behavior vertical slice against frozen DANDI inputs."""

    manifests = result_root / "manifests"
    tables_directory = result_root / "tables"
    inventory_path = manifests / "session_inventory.parquet"
    task_parameters_audit_path = manifests / "task_parameters_audit.csv"
    companion_path = result_root / "cache" / "master_stim_trials_table.csv"
    audit_run_path = manifests / "audit_run.json"

    lock = dg.gates.read_analysis_lock(arguments.analysis_lock)
    lock_text = arguments.analysis_lock.read_text(encoding="utf-8")
    if json.loads(lock_text) != lock:
        raise RuntimeError("analysis lock changed while it was being read")
    _validate_locked_dandiset(lock)
    pending_decisions = dg.gates.unapproved_decisions(lock, BEHAVIOR_DECISIONS)
    audit_run = _read_completed_audit(
        audit_run_path,
        artifact_paths={
            "session_inventory": inventory_path,
            "task_parameters_audit": task_parameters_audit_path,
        },
    )
    audit_run_manifest_record = _file_provenance_record(audit_run_path)
    inventory_sha256 = _sha256_file(inventory_path)
    inventory = _read_session_inventory(inventory_path)
    if _sha256_file(inventory_path) != inventory_sha256:
        raise RuntimeError("session inventory changed while it was being read")
    task_parameters_audit_sha256 = _read_validated_task_parameters_audit(
        task_parameters_audit_path,
        inventory,
    )
    companion_provenance = dg.audit.cache_companion_trials(companion_path)
    companion_provenance_record = companion_provenance.row(0, named=True)
    code_version = _code_version(REPOSITORY_ROOT)
    source_provenance = _analysis_source_provenance()

    behavior_trials_path = tables_directory / "behavior_trials.parquet"
    checkpoint_path = manifests / BEHAVIOR_TRIALS_CHECKPOINT_FILENAME
    checkpoint: dict[str, Any] | None = None
    checkpoint_manifest_record: dict[str, Any] | None = None
    reused_behavior_trials_sha256 = None
    if arguments.reuse_behavior_trials:
        checkpoint, checkpoint_manifest_record = (
            _read_validated_behavior_trials_checkpoint_manifest(
                checkpoint_path,
                behavior_trials_path=behavior_trials_path,
                session_inventory_path=inventory_path,
                companion_trials_path=companion_path,
            )
        )
        reused_behavior_trials_sha256 = checkpoint["behavior_trials"]["sha256"]
        canonical_trials = _read_behavior_trials_checkpoint(
            behavior_trials_path,
            inventory,
        )
        replaceable = [
            column for column in DERIVED_TRIAL_COLUMNS if column in canonical_trials.columns
        ]
        canonical_trials = canonical_trials.drop(*replaceable)
    else:
        sources = inventory.get_column("_nwb_path").to_list()
        raw_trials = dg.data.scan_trials(
            sources,
            columns=dg.data.BEHAVIOR_TRIAL_COLUMNS,
        )
        metadata_columns = ["_nwb_path", "ecephys_session_id"]
        metadata_columns.extend(
            column
            for column in ("ecephys_session_id_valid", "session_identifier_conflict")
            if column in inventory.columns
        )
        companion_epochs = dg.data.scan_companion_trial_reward_epochs(companion_path)
        canonical_trials = dg.data.add_no_reward_epoch_from_companion(
            raw_trials,
            inventory.select(*metadata_columns),
            companion_epochs,
        )
    canonical_trials = dg.data.add_physical_image_change_flag(canonical_trials)
    canonical_trials = dg.quality.add_trial_response_from_licks(canonical_trials)
    canonical_trials = dg.quality.label_reward_blocks(canonical_trials)
    canonical_trials = canonical_trials.collect()
    if not arguments.reuse_behavior_trials:
        canonical_trials = _attach_trial_metadata(canonical_trials, inventory)

    behavior_tables = dg.behavior.build_behavior_tables(
        canonical_trials,
        inventory.select("_nwb_path", "subject_id"),
    )
    session_qc = _attach_session_metadata(behavior_tables.sessions, inventory)
    session_blocks = _attach_session_metadata(
        behavior_tables.session_blocks,
        inventory,
        join_validation="1:m",
    )
    block_timing = build_session_block_timing(canonical_trials, session_qc)
    reward_epoch_audit = build_reward_epoch_audit(canonical_trials, inventory)
    response_audit = build_response_label_audit(session_qc)
    lick_timestamp_audit = build_lick_timestamp_audit(canonical_trials, inventory)
    behavior_statistics = build_behavior_statistics(
        behavior_tables,
        pending_decisions=pending_decisions,
        code_version=code_version,
        seed=arguments.seed,
        bootstrap_resamples=arguments.bootstrap_resamples,
        sign_flip_resamples=arguments.sign_flip_resamples,
    )

    if _file_provenance_record(audit_run_path) != audit_run_manifest_record:
        raise RuntimeError("M0 audit manifest changed during the behavior analysis")
    if _sha256_file(inventory_path) != inventory_sha256:
        raise RuntimeError("session inventory changed during the behavior analysis")
    if _sha256_file(task_parameters_audit_path) != task_parameters_audit_sha256:
        raise RuntimeError("task-parameter audit changed during the behavior analysis")

    outputs = {
        "behavior_trials": behavior_trials_path,
        "session_qc": tables_directory / "session_qc.parquet",
        "session_attrition": tables_directory / "session_attrition.csv",
        "session_behavior_blocks": tables_directory / "session_behavior_blocks.csv",
        "session_block_timing": tables_directory / "session_block_timing.csv",
        "mouse_behavior_blocks": tables_directory / "mouse_behavior_blocks.csv",
        "mouse_gating": tables_directory / "mouse_gating.csv",
        "reward_epoch_audit": tables_directory / "reward_epoch_audit.csv",
        "behavior_response_audit": tables_directory / "behavior_response_audit.csv",
        "lick_timestamp_audit": tables_directory / "lick_timestamp_audit.csv",
        "behavior_statistics": tables_directory / "behavior_statistics.csv",
    }
    artifact_frames = {
        "behavior_trials": canonical_trials,
        "session_qc": session_qc,
        "session_attrition": behavior_tables.session_attrition,
        "session_behavior_blocks": session_blocks,
        "session_block_timing": block_timing,
        "mouse_behavior_blocks": behavior_tables.mouse_blocks,
        "mouse_gating": behavior_tables.mouse_gating,
        "reward_epoch_audit": reward_epoch_audit,
        "behavior_response_audit": response_audit,
        "lick_timestamp_audit": lick_timestamp_audit,
        "behavior_statistics": behavior_statistics,
    }
    if _analysis_source_provenance() != source_provenance:
        raise RuntimeError("local behavior-analysis source changed during the run")
    for name, frame in artifact_frames.items():
        dg.artifacts.write_frame(frame, outputs[name])
    lock_snapshot = manifests / "behavior_analysis_lock.yaml"
    dg.artifacts.write_text(lock_text, lock_snapshot)
    if _analysis_source_provenance() != source_provenance:
        raise RuntimeError("local behavior-analysis source changed while outputs were written")
    output_records = {name: _file_provenance_record(path) for name, path in outputs.items()}

    if arguments.reuse_behavior_trials:
        current_checkpoint_manifest_record = _file_provenance_record(checkpoint_path)
        if current_checkpoint_manifest_record != checkpoint_manifest_record:
            raise RuntimeError("behavior-trials checkpoint manifest changed during the reuse run")
        checkpoint_status = "validated_existing_not_refreshed"
    else:
        checkpoint, checkpoint_manifest_record = _write_behavior_trials_checkpoint_manifest(
            checkpoint_path,
            behavior_trials_path=behavior_trials_path,
            session_inventory_path=inventory_path,
            companion_trials_path=companion_path,
            companion_provenance=companion_provenance_record,
            source_provenance=source_provenance,
            created_at=datetime.datetime.now(datetime.UTC),
            trial_input_mode="remote_nwb",
        )
        checkpoint_status = "created_from_remote_nwb"
    if checkpoint is None or checkpoint_manifest_record is None:
        raise RuntimeError("behavior-trials checkpoint state was not established")
    checkpoint_reference = {
        **checkpoint_manifest_record,
        "status": checkpoint_status,
        "trusted_behavior_trials_sha256": checkpoint["behavior_trials"]["sha256"],
        "output_matches_trusted_checkpoint": (
            output_records["behavior_trials"]["sha256"] == checkpoint["behavior_trials"]["sha256"]
        ),
    }

    completed_at = datetime.datetime.now(datetime.UTC)
    technical_sessions = session_qc.filter(pl.col("is_technically_valid"))
    estimable_mice = behavior_tables.mouse_gating.filter(
        (pl.col("session_cohort") == "technically_valid")
        & pl.col("reversible_gating_estimate").is_not_null()
        & pl.col("reversible_gating_estimate").is_finite()
    )
    session_lick_audit = lick_timestamp_audit.filter(pl.col("audit_scope") == "session")
    run_record = {
        "script": str(pathlib.Path(__file__).relative_to(REPOSITORY_ROOT)),
        "generator": source_provenance["generator"],
        "local_sources": source_provenance["local_sources"],
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "fragile" if pending_decisions else "pass",
        "run_status": "complete",
        "authoritative": checkpoint_reference["output_matches_trusted_checkpoint"],
        "authoritative_definition": (
            "completed output exactly matches the trusted direct remote-NWB trial checkpoint"
        ),
        "pending_behavior_decisions": pending_decisions,
        "analysis_lock_path": str(arguments.analysis_lock.resolve()),
        "analysis_lock_snapshot": _display_path(lock_snapshot),
        "analysis_lock_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "dandiset_doi": dg.data.DANDISET_DOI,
        "code_version": code_version,
        "input_audit_run": audit_run,
        "input_audit_run_manifest": audit_run_manifest_record,
        "session_inventory_path": str(inventory_path.resolve()),
        "session_inventory_sha256": inventory_sha256,
        "task_parameters_audit_path": _display_path(task_parameters_audit_path),
        "task_parameters_audit_sha256": task_parameters_audit_sha256,
        "trial_input_mode": (
            "validated_existing_behavior_trials"
            if arguments.reuse_behavior_trials
            else "remote_nwb"
        ),
        "reused_behavior_trials_sha256": reused_behavior_trials_sha256,
        "behavior_trials_checkpoint": checkpoint_reference,
        "companion_trials_provenance": companion_provenance_record,
        "response_definition": {
            "source": "raw trial lick_times",
            "event_time_column": "change_time",
            "window_start_seconds": dg.quality.DEFAULT_RESPONSE_WINDOW_START_SECONDS,
            "window_stop_seconds": dg.quality.DEFAULT_RESPONSE_WINDOW_STOP_SECONDS,
            "window_interval": "(start, stop]",
            "count_semantics": "exact-deduplicated lick timestamps",
            "online_outcomes": "agreement audit only; interpretable in reward-available blocks",
        },
        "lick_timestamp_audit_summary": {
            "n_raw_lick_timestamps": int(
                session_lick_audit.get_column("n_raw_lick_timestamps").sum() or 0
            ),
            "n_unique_lick_timestamps": int(
                session_lick_audit.get_column("n_unique_lick_timestamps").sum() or 0
            ),
            "n_duplicate_lick_timestamps": int(
                session_lick_audit.get_column("n_duplicate_lick_timestamps").sum() or 0
            ),
            "n_raw_response_window_licks": int(
                session_lick_audit.get_column("n_raw_response_window_licks").sum() or 0
            ),
            "n_unique_response_window_licks": int(
                session_lick_audit.get_column("n_unique_response_window_licks").sum() or 0
            ),
            "n_duplicate_response_window_licks": int(
                session_lick_audit.get_column("n_duplicate_response_window_licks").sum() or 0
            ),
            "n_trials_with_duplicate_lick_timestamps": int(
                session_lick_audit.get_column("n_trials_with_duplicate_lick_timestamps").sum() or 0
            ),
            "n_trials_with_invalid_lick_vectors": int(
                session_lick_audit.get_column("n_trials_with_invalid_lick_vectors").sum() or 0
            ),
            "n_trials_without_valid_response_event": int(
                session_lick_audit.get_column("n_trials_without_valid_response_event").sum() or 0
            ),
            "n_sessions_with_duplicate_lick_timestamps": session_lick_audit.filter(
                pl.col("n_duplicate_lick_timestamps") > 0
            ).height,
            "n_sessions_with_invalid_lick_vectors": session_lick_audit.filter(
                pl.col("n_trials_with_invalid_lick_vectors") > 0
            ).height,
        },
        "inference_cohort": "technically_valid_complete_primary_blocks",
        "primary_test_result_ids": [
            "withdrawal_suppression_mouse_mean",
            "restoration_recovery_mouse_mean",
        ],
        "restoration_claim_rule": (
            "requires positive withdrawal and restoration estimates with both tests interpreted "
            "in the shared Holm family; the symmetric reversible score is summary-only"
        ),
        "threshold_selected_cohort_role": "QC characterization only",
        "session_quality_thresholds": dataclasses.asdict(
            dg.quality.DEFAULT_SESSION_QUALITY_THRESHOLDS
        ),
        "late_no_reward_seconds": 600.0,
        "late_engaged_2_exclusion_seconds": 600.0,
        "aggregation": AGGREGATION,
        "seed": arguments.seed,
        "bootstrap_resamples": arguments.bootstrap_resamples,
        "sign_flip_resamples": arguments.sign_flip_resamples,
        "n_inventory_sessions": inventory.height,
        "n_inventory_mice": inventory.get_column("subject_id").n_unique(),
        "n_trials": canonical_trials.height,
        "n_technically_valid_sessions": technical_sessions.height,
        "n_technically_valid_mice": technical_sessions.get_column("subject_id").n_unique(),
        "n_estimable_sessions": session_qc.filter(
            pl.col("is_technically_valid") & pl.col("reversible_gating_estimable")
        ).height,
        "n_estimable_mice": estimable_mice.height,
        "outputs": output_records,
    }
    dg.artifacts.write_json(run_record, manifests / "behavior_analysis_run.json")
    print(
        "Wrote behavior artifacts for "
        f"{technical_sessions.height}/{inventory.height} technically valid sessions "
        f"and {estimable_mice.height} estimable mice to {tables_directory}"
    )


def build_session_block_timing(
    trials: pl.DataFrame,
    session_qc: pl.DataFrame,
) -> pl.DataFrame:
    """Return actual E1/NR/E2 boundaries for the Figure 1 schematic."""

    _require_columns(
        trials,
        ("_nwb_path", "reward_block", "start_time", "stop_time", "_table_index"),
        frame_name="trials",
    )
    _require_columns(
        session_qc,
        ("_nwb_path", "subject_id", "is_technically_valid", "is_good_session"),
        frame_name="session_qc",
    )
    session_metadata_columns = [
        "_nwb_path",
        "subject_id",
        "is_technically_valid",
        "is_good_session",
    ]
    session_metadata_columns.extend(
        column
        for column in ("ecephys_session_id", "recording_day", "session_number")
        if column in session_qc.columns
    )
    session_metadata = session_qc.select(*session_metadata_columns)
    _validate_unique_non_null(session_metadata, "_nwb_path", frame_name="session_qc")

    session_start = trials.group_by("_nwb_path").agg(
        pl.col("start_time").min().alias("session_first_trial_start_time")
    )
    timing = (
        trials.filter(pl.col("reward_block").is_in(["engaged_1", "no_reward", "engaged_2"]))
        .group_by("_nwb_path", "reward_block")
        .agg(
            pl.col("start_time").min().alias("block_start_time"),
            pl.col("stop_time").max().alias("block_stop_time"),
            pl.col("_table_index").min().alias("first_trial_table_index"),
            pl.col("_table_index").max().alias("last_trial_table_index"),
            pl.len().cast(pl.Int64).alias("n_trials"),
        )
        .join(session_start, on="_nwb_path", how="left", validate="m:1")
        .join(session_metadata, on="_nwb_path", how="left", validate="m:1")
        .with_columns(
            pl.col("reward_block")
            .replace_strict(
                {"engaged_1": 1, "no_reward": 2, "engaged_2": 3},
                return_dtype=pl.Int8,
            )
            .alias("block_order"),
            (pl.col("block_stop_time") - pl.col("block_start_time")).alias(
                "block_duration_seconds"
            ),
            (pl.col("block_start_time") - pl.col("session_first_trial_start_time")).alias(
                "block_start_from_session_seconds"
            ),
            (pl.col("block_stop_time") - pl.col("session_first_trial_start_time")).alias(
                "block_stop_from_session_seconds"
            ),
        )
        .sort("subject_id", "_nwb_path", "block_order")
    )
    return timing.select(
        "_nwb_path",
        "subject_id",
        *(
            column
            for column in ("ecephys_session_id", "recording_day", "session_number")
            if column in timing.columns
        ),
        "reward_block",
        "block_order",
        "block_start_time",
        "block_stop_time",
        "block_duration_seconds",
        "block_start_from_session_seconds",
        "block_stop_from_session_seconds",
        "first_trial_table_index",
        "last_trial_table_index",
        "n_trials",
        "is_technically_valid",
        "is_good_session",
    )


def build_reward_epoch_audit(
    trials: pl.DataFrame,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Summarize NWB/companion reward-label coverage and conflicts per session."""

    required = (
        "_nwb_path",
        "no_reward_epoch",
        "nwb_no_reward_epoch",
        "companion_no_reward_epoch",
        "companion_reward_epoch_conflict",
        "no_reward_epoch_conflict",
        "no_reward_epoch_source",
    )
    _require_columns(trials, required, frame_name="trials")
    _require_columns(inventory, ("_nwb_path", "subject_id"), frame_name="inventory")
    metadata_columns = ["_nwb_path", "subject_id"]
    metadata_columns.extend(
        column for column in ("ecephys_session_id", "recording_day") if column in inventory.columns
    )
    effective_companion_conflict = pl.col("companion_reward_epoch_conflict").fill_null(False) & (
        pl.col("no_reward_epoch_source") == "companion"
    )
    audit = trials.group_by("_nwb_path").agg(
        pl.len().cast(pl.Int64).alias("n_trials"),
        pl.col("no_reward_epoch").is_null().sum().cast(pl.Int64).alias("n_unresolved_trials"),
        (pl.col("no_reward_epoch_source") == "nwb")
        .sum()
        .cast(pl.Int64)
        .alias("n_nwb_source_trials"),
        (pl.col("no_reward_epoch_source") == "companion")
        .sum()
        .cast(pl.Int64)
        .alias("n_companion_source_trials"),
        (pl.col("no_reward_epoch_source") == "missing")
        .sum()
        .cast(pl.Int64)
        .alias("n_missing_source_trials"),
        pl.col("no_reward_epoch").fill_null(False).sum().cast(pl.Int64).alias("n_no_reward_trials"),
        pl.col("no_reward_epoch_conflict")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("n_nwb_companion_disagreements"),
        pl.col("companion_reward_epoch_conflict")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("n_companion_internal_conflict_trials"),
        (pl.col("no_reward_epoch_conflict").fill_null(False) | effective_companion_conflict)
        .sum()
        .cast(pl.Int64)
        .alias("n_effective_reward_epoch_conflicts"),
    )
    return (
        inventory.select(*metadata_columns)
        .join(audit, on="_nwb_path", how="left", validate="1:1")
        .with_columns(
            pl.when(pl.col("n_trials").is_null())
            .then(pl.lit("missing_trials"))
            .when(pl.col("n_unresolved_trials") > 0)
            .then(pl.lit("unresolved"))
            .when(pl.col("n_effective_reward_epoch_conflicts") > 0)
            .then(pl.lit("conflict"))
            .otherwise(pl.lit("pass"))
            .alias("reward_epoch_audit_status")
        )
        .sort("subject_id", "_nwb_path")
    )


def build_response_label_audit(session_qc: pl.DataFrame) -> pl.DataFrame:
    """Reshape online-label agreement, marking NR outcomes as audit-only."""

    identity = ["_nwb_path", "subject_id", "is_technically_valid", "is_good_session"]
    identity.extend(
        column for column in ("ecephys_session_id", "recording_day") if column in session_qc.columns
    )
    blocks = []
    for order, block in enumerate(("engaged_1", "no_reward", "engaged_2"), start=1):
        columns = {
            "n_label_trials": f"n_{block}_response_label_trials",
            "n_unavailable_label_comparisons": f"n_{block}_response_label_missing",
            "n_online_outcome_absent": f"n_{block}_online_outcome_absent",
            "n_online_outcome_conflicts": f"n_{block}_online_outcome_conflicts",
            "n_canonical_response_missing": f"n_{block}_canonical_response_missing",
            "n_label_comparisons": f"n_{block}_response_label_comparisons",
            "n_label_disagreements": f"n_{block}_response_label_disagreements",
            "agreement_rate": f"{block}_response_label_agreement_rate",
        }
        _require_columns(
            session_qc,
            (*identity, *columns.values()),
            frame_name="session_qc",
        )
        online_labels_interpretable = block != "no_reward"
        blocks.append(
            session_qc.select(
                *identity,
                pl.lit(block).alias("reward_block"),
                pl.lit(order, dtype=pl.Int8).alias("block_order"),
                pl.lit(online_labels_interpretable).alias("online_labels_interpretable"),
                pl.lit(
                    "reward_available_validation"
                    if online_labels_interpretable
                    else "audit_only_online_outcomes_not_state_independent"
                ).alias("interpretation_scope"),
                *(pl.col(source).alias(destination) for destination, source in columns.items()),
            )
        )
    return pl.concat(blocks, how="vertical").sort("subject_id", "_nwb_path", "block_order")


def build_lick_timestamp_audit(
    trials: pl.DataFrame,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Report raw, exact-deduplicated, and duplicate lick counts by session/block."""

    count_columns = (
        "n_raw_lick_timestamps",
        "n_unique_lick_timestamps",
        "n_duplicate_lick_timestamps",
        "n_raw_response_window_licks",
        "n_unique_response_window_licks",
        "n_duplicate_response_window_licks",
        "n_response_window_licks",
    )
    _require_columns(
        trials,
        (
            "_nwb_path",
            "reward_block",
            "lick_times_valid",
            "response_event_time_valid",
            *count_columns,
        ),
        frame_name="trials",
    )
    _require_columns(inventory, ("_nwb_path", "subject_id"), frame_name="inventory")
    metadata_columns = ["_nwb_path", "subject_id"]
    metadata_columns.extend(
        column for column in ("ecephys_session_id", "recording_day") if column in inventory.columns
    )

    aggregations = (
        pl.len().cast(pl.Int64).alias("n_trials"),
        pl.col("lick_times_valid")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_with_valid_lick_vectors"),
        (~pl.col("lick_times_valid").fill_null(False))
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_with_invalid_lick_vectors"),
        pl.col("response_event_time_valid")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_with_valid_response_event"),
        (~pl.col("response_event_time_valid").fill_null(False))
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_without_valid_response_event"),
        (pl.col("n_duplicate_lick_timestamps").fill_null(0) > 0)
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_with_duplicate_lick_timestamps"),
        (pl.col("n_duplicate_response_window_licks").fill_null(0) > 0)
        .sum()
        .cast(pl.Int64)
        .alias("n_trials_with_duplicate_response_window_licks"),
        *(
            pl.col(column).fill_null(0).sum().cast(pl.Int64).alias(column)
            for column in count_columns
        ),
    )
    by_block = (
        trials.group_by("_nwb_path", "reward_block")
        .agg(*aggregations)
        .with_columns(
            pl.lit("session_block").alias("audit_scope"),
            pl.col("reward_block")
            .replace_strict(
                {"engaged_1": 1, "no_reward": 2, "engaged_2": 3, "transition": 4},
                default=5,
                return_dtype=pl.Int8,
            )
            .alias("block_order"),
        )
    )
    by_session = (
        trials.group_by("_nwb_path")
        .agg(*aggregations)
        .with_columns(
            pl.lit("all").alias("reward_block"),
            pl.lit("session").alias("audit_scope"),
            pl.lit(0, dtype=pl.Int8).alias("block_order"),
        )
    )
    audit = pl.concat([by_session, by_block], how="diagonal_relaxed").with_columns(
        (pl.col("n_response_window_licks") == pl.col("n_unique_response_window_licks")).alias(
            "canonical_response_count_is_deduplicated"
        ),
        (
            pl.col("n_raw_lick_timestamps") - pl.col("n_unique_lick_timestamps")
            == pl.col("n_duplicate_lick_timestamps")
        ).alias("total_duplicate_count_consistent"),
        (
            pl.col("n_raw_response_window_licks") - pl.col("n_unique_response_window_licks")
            == pl.col("n_duplicate_response_window_licks")
        ).alias("response_window_duplicate_count_consistent"),
        pl.when(pl.col("n_trials") > 0)
        .then(pl.col("n_trials_with_duplicate_lick_timestamps") / pl.col("n_trials"))
        .otherwise(None)
        .alias("fraction_trials_with_duplicate_lick_timestamps"),
    )
    audit = audit.with_columns(
        pl.when(
            ~pl.col("canonical_response_count_is_deduplicated")
            | ~pl.col("total_duplicate_count_consistent")
            | ~pl.col("response_window_duplicate_count_consistent")
        )
        .then(pl.lit("count_inconsistency"))
        .when(pl.col("n_trials_with_invalid_lick_vectors") > 0)
        .then(pl.lit("invalid_vectors_reported"))
        .when(pl.col("n_duplicate_lick_timestamps") > 0)
        .then(pl.lit("duplicates_reported"))
        .otherwise(pl.lit("pass"))
        .alias("lick_timestamp_audit_status")
    )
    joined = audit.join(
        inventory.select(*metadata_columns),
        on="_nwb_path",
        how="left",
        validate="m:1",
    )
    optional_metadata = [
        column for column in ("ecephys_session_id", "recording_day") if column in joined.columns
    ]
    return joined.select(
        "_nwb_path",
        "subject_id",
        *optional_metadata,
        "audit_scope",
        "reward_block",
        "block_order",
        pl.exclude(
            "_nwb_path",
            "subject_id",
            "ecephys_session_id",
            "recording_day",
            "audit_scope",
            "reward_block",
            "block_order",
        ),
    ).sort("subject_id", "_nwb_path", "block_order")


def _sum_trial_counts(sessions: pl.DataFrame, columns: tuple[str, ...]) -> int:
    """Sum the selected per-session trial counts, treating absent counts as zero."""

    total = sessions.select(
        pl.sum_horizontal(*(pl.col(column).fill_null(0) for column in columns))
        .sum()
        .cast(pl.Int64)
        .alias("n_trials")
    ).item()
    return int(total or 0)


def build_behavior_statistics(
    behavior_tables: dg.behavior.BehaviorTables,
    *,
    pending_decisions: list[str],
    code_version: str,
    seed: int = DEFAULT_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    sign_flip_resamples: int = DEFAULT_SIGN_FLIP_RESAMPLES,
) -> pl.DataFrame:
    """Build the canonical mouse-level exploratory statistics table."""

    rows: list[dict[str, Any]] = []
    fragile_reason = (
        "analysis-lock decisions remain proposed: " + ", ".join(pending_decisions)
        if pending_decisions
        else None
    )
    default_status = "fragile" if pending_decisions else "pass"

    mouse_gating = behavior_tables.mouse_gating.filter(
        (pl.col("session_cohort") == "technically_valid")
        & pl.col("reversible_gating_estimate").is_not_null()
        & pl.col("reversible_gating_estimate").is_finite()
        & (pl.col("n_sessions_contributing") > 0)
    )
    primary_sessions = behavior_tables.sessions.filter(
        pl.col("is_technically_valid") & pl.col("reversible_gating_estimable")
    )
    response_trial_columns = {
        "reversible_gating_estimate": (
            "n_engaged_1_go_trials",
            "n_no_reward_late_go_trials",
            "n_engaged_2_early_go_trials",
        ),
        "withdrawal_suppression_estimate": (
            "n_engaged_1_go_trials",
            "n_no_reward_late_go_trials",
        ),
        "restoration_recovery_estimate": (
            "n_no_reward_late_go_trials",
            "n_engaged_2_early_go_trials",
        ),
        "engaged_response_drift_estimate": (
            "n_engaged_1_go_trials",
            "n_engaged_2_early_go_trials",
        ),
    }
    response_trial_totals = {
        metric: _sum_trial_counts(primary_sessions, columns)
        for metric, columns in response_trial_columns.items()
    }
    symmetric_summary = _mouse_metric_statistics(
        mouse_gating,
        mouse_column="subject_id",
        value_column="reversible_gating_estimate",
        n_sessions_column="n_sessions_contributing",
        n_trials=response_trial_totals["reversible_gating_estimate"],
        result_id=SYMMETRIC_SUMMARY_RESULT_ID,
        contrast_id="0.5*engaged_1-no_reward_late+0.5*engaged_2_early",
        hypothesis=(
            "Summarize the symmetric contrast between mean reward-available change-response "
            "probability and late no-reward response probability."
        ),
        scale="response-probability difference",
        code_version=code_version,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
        default_status=default_status,
        fragile_reason=fragile_reason,
    )
    rows.append(symmetric_summary)

    response_contrasts = (
        (
            "withdrawal_suppression_estimate",
            "withdrawal_suppression_mouse_mean",
            "engaged_1-no_reward_late",
            "Change-response probability decreases after reward withdrawal.",
            "figure_1_primary_behavior",
        ),
        (
            "restoration_recovery_estimate",
            "restoration_recovery_mouse_mean",
            "engaged_2_early-no_reward_late",
            "Change-response probability increases after reward restoration.",
            "figure_1_primary_behavior",
        ),
        (
            "engaged_response_drift_estimate",
            "engaged_response_drift_mouse_mean",
            "engaged_2_early-engaged_1",
            "Quantify monotonic response drift between reward-available blocks.",
            "figure_1_response_drift_sensitivity",
        ),
    )
    for value_column, result_id, contrast_id, hypothesis, family in response_contrasts:
        contrast_mice = mouse_gating.filter(
            pl.col(value_column).is_not_null() & pl.col(value_column).is_finite()
        )
        rows.append(
            _mouse_metric_statistics(
                contrast_mice,
                mouse_column="subject_id",
                value_column=value_column,
                n_sessions_column="n_sessions_contributing",
                n_trials=response_trial_totals[value_column],
                result_id=result_id,
                contrast_id=contrast_id,
                hypothesis=hypothesis,
                scale="response-probability difference",
                code_version=code_version,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
                default_status=default_status,
                fragile_reason=fragile_reason,
                run_sign_flip=True,
                sign_flip_resamples=sign_flip_resamples,
                multiplicity_family=family,
            )
        )

    specificity_sessions = behavior_tables.sessions.filter(
        pl.col("is_technically_valid")
        & pl.col("reversible_specificity_gating_estimate").is_not_null()
    )
    specificity_trial_columns = {
        "reversible_specificity_gating_estimate": (
            "n_engaged_1_go_trials",
            "n_engaged_1_catch_trials",
            "n_no_reward_late_go_trials",
            "n_no_reward_late_catch_trials",
            "n_engaged_2_early_go_trials",
            "n_engaged_2_early_catch_trials",
        ),
        "withdrawal_specificity_suppression_estimate": (
            "n_engaged_1_go_trials",
            "n_engaged_1_catch_trials",
            "n_no_reward_late_go_trials",
            "n_no_reward_late_catch_trials",
        ),
        "restoration_specificity_recovery_estimate": (
            "n_no_reward_late_go_trials",
            "n_no_reward_late_catch_trials",
            "n_engaged_2_early_go_trials",
            "n_engaged_2_early_catch_trials",
        ),
        "engaged_specificity_drift_estimate": (
            "n_engaged_1_go_trials",
            "n_engaged_1_catch_trials",
            "n_engaged_2_early_go_trials",
            "n_engaged_2_early_catch_trials",
        ),
    }
    specificity_trial_totals = {
        metric: _sum_trial_counts(specificity_sessions, columns)
        for metric, columns in specificity_trial_columns.items()
    }
    specificity_definitions = (
        (
            "reversible_specificity_gating_estimate",
            "reversible_specificity_gating_mouse_mean",
            "0.5*engaged_1_specificity-no_reward_late_specificity+0.5*engaged_2_early_specificity",
            "Summarize reversible change-versus-catch response specificity.",
            False,
            None,
        ),
        (
            "withdrawal_specificity_suppression_estimate",
            "withdrawal_specificity_suppression_mouse_mean",
            "engaged_1_specificity-no_reward_late_specificity",
            "Change-versus-catch response specificity decreases after reward withdrawal.",
            True,
            "figure_1_specificity_behavior",
        ),
        (
            "restoration_specificity_recovery_estimate",
            "restoration_specificity_recovery_mouse_mean",
            "engaged_2_early_specificity-no_reward_late_specificity",
            "Change-versus-catch response specificity increases after reward restoration.",
            True,
            "figure_1_specificity_behavior",
        ),
        (
            "engaged_specificity_drift_estimate",
            "engaged_specificity_drift_mouse_mean",
            "engaged_2_early_specificity-engaged_1_specificity",
            "Quantify specificity drift between reward-available blocks.",
            True,
            "figure_1_specificity_drift_sensitivity",
        ),
    )
    for (
        value_column,
        result_id,
        contrast_id,
        hypothesis,
        run_test,
        family,
    ) in specificity_definitions:
        specificity_mice = mouse_gating.filter(
            (pl.col("n_sessions_specificity_contributing") > 0)
            & pl.col(value_column).is_not_null()
            & pl.col(value_column).is_finite()
        )
        rows.append(
            _mouse_metric_statistics(
                specificity_mice,
                mouse_column="subject_id",
                value_column=value_column,
                n_sessions_column="n_sessions_specificity_contributing",
                n_trials=specificity_trial_totals[value_column],
                result_id=result_id,
                contrast_id=contrast_id,
                hypothesis=hypothesis,
                scale="change-minus-catch probability difference",
                code_version=code_version,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
                default_status=default_status,
                fragile_reason=fragile_reason,
                run_sign_flip=run_test,
                sign_flip_resamples=sign_flip_resamples,
                multiplicity_family=family,
                inclusion_definition=SPECIFICITY_INCLUSION_DEFINITION,
            )
        )

    mouse_blocks = behavior_tables.mouse_blocks.filter(
        pl.col("session_cohort") == "technically_valid"
    )
    metric_definitions = (
        (
            "response_probability",
            "n_sessions_response_contributing",
            "n_go_trials",
            "change-response probability",
        ),
        (
            "false_alarm_probability",
            "n_sessions_false_alarm_contributing",
            "n_catch_trials",
            "catch-response probability",
        ),
        (
            "dprime",
            "n_sessions_dprime_contributing",
            None,
            "loglinear d-prime",
        ),
    )
    for block in dg.behavior.BEHAVIOR_BLOCKS:
        block_rows = mouse_blocks.filter(pl.col("behavior_block") == block)
        for metric, n_sessions_column, n_trials_column, scale in metric_definitions:
            estimable = block_rows.filter(
                pl.col(metric).is_not_null()
                & pl.col(metric).is_finite()
                & (pl.col(n_sessions_column) > 0)
            )
            n_trials = (
                int(estimable.get_column(n_trials_column).sum() or 0)
                if n_trials_column is not None
                else int(
                    estimable.select(
                        (pl.col("n_go_trials") + pl.col("n_catch_trials")).sum()
                    ).item()
                    or 0
                )
            )
            rows.append(
                _mouse_metric_statistics(
                    estimable,
                    mouse_column="subject_id",
                    value_column=metric,
                    n_sessions_column=n_sessions_column,
                    n_trials=n_trials,
                    result_id=f"{block}_{metric}_mouse_mean",
                    contrast_id=f"{block}_{metric}",
                    hypothesis=f"Describe mouse-level {scale} during {block}.",
                    scale=scale,
                    code_version=code_version,
                    seed=seed,
                    bootstrap_resamples=bootstrap_resamples,
                    default_status=default_status,
                    fragile_reason=fragile_reason,
                    inclusion_definition=BLOCK_INCLUSION_DEFINITION,
                )
            )

    sessions = behavior_tables.sessions.filter(pl.col("is_technically_valid"))
    for block in ("engaged_1", "engaged_2"):
        value_column = f"{block}_response_label_agreement_rate"
        trials_column = f"n_{block}_response_label_comparisons"
        per_mouse = (
            sessions.filter(pl.col(value_column).is_not_null() & pl.col(value_column).is_finite())
            .group_by("subject_id")
            .agg(
                pl.col(value_column).mean(),
                pl.len().cast(pl.Int64).alias("n_sessions_contributing"),
                pl.col(trials_column).sum().cast(pl.Int64).alias("n_trials"),
            )
        )
        rows.append(
            _mouse_metric_statistics(
                per_mouse,
                mouse_column="subject_id",
                value_column=value_column,
                n_sessions_column="n_sessions_contributing",
                n_trials=int(per_mouse.get_column("n_trials").sum()) if per_mouse.height else 0,
                result_id=f"{block}_raw_lick_online_label_agreement",
                contrast_id=f"{block}_response_label_agreement",
                hypothesis=(
                    "Raw-lick responses agree with online outcome labels in the "
                    f"reward-available {block} block."
                ),
                scale="agreement probability",
                code_version=code_version,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
                default_status=default_status,
                fragile_reason=fragile_reason,
                inclusion_definition=BLOCK_INCLUSION_DEFINITION,
            )
        )

    for attrition in behavior_tables.session_attrition.iter_rows(named=True):
        rows.append(
            _base_statistics_row(
                result_id=f"session_attrition_{attrition['attrition_stage']}",
                contrast_id=f"attrition_{attrition['attrition_stage']}",
                hypothesis=(
                    "Describe cumulative session retention at the "
                    f"{attrition['attrition_stage']} stage."
                ),
                code_version=code_version,
                status=default_status,
                reason=fragile_reason,
                estimate=attrition["fraction_of_inventory"],
                scale="fraction of session inventory",
                n_mice=attrition["n_mice"],
                n_sessions=attrition["n_sessions"],
                aggregation="cumulative session attrition",
                inclusion_definition=ATTRITION_INCLUSION_DEFINITION,
            )
        )

    statistics = dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    return dg.statistics.add_holm_adjustment(statistics)


def _mouse_metric_statistics(
    mice: pl.DataFrame,
    *,
    mouse_column: str,
    value_column: str,
    n_sessions_column: str,
    n_trials: int,
    result_id: str,
    contrast_id: str,
    hypothesis: str,
    scale: str,
    code_version: str,
    seed: int,
    bootstrap_resamples: int,
    default_status: str,
    fragile_reason: str | None,
    run_sign_flip: bool = False,
    sign_flip_resamples: int = DEFAULT_SIGN_FLIP_RESAMPLES,
    multiplicity_family: str | None = None,
    inclusion_definition: str = INCLUSION_DEFINITION,
) -> dict[str, Any]:
    """Return one canonical row for a complete one-row-per-mouse metric."""

    if mice.height < 2:
        return _base_statistics_row(
            result_id=result_id,
            contrast_id=contrast_id,
            hypothesis=hypothesis,
            code_version=code_version,
            status="not_estimable",
            reason=f"mouse-level interval requires at least two mice; observed {mice.height}",
            estimate=None,
            scale=scale,
            n_mice=mice.height,
            n_sessions=(int(mice.get_column(n_sessions_column).sum()) if mice.height else 0),
            n_trials=n_trials,
            inclusion_definition=inclusion_definition,
        )
    bootstrap = dg.statistics.bootstrap_mouse_mean(
        mice,
        seed=seed,
        mouse_column=mouse_column,
        value_column=value_column,
        n_resamples=bootstrap_resamples,
    )
    row = _base_statistics_row(
        result_id=result_id,
        contrast_id=contrast_id,
        hypothesis=hypothesis,
        code_version=code_version,
        status=default_status,
        reason=fragile_reason,
        seed=seed,
        estimate=bootstrap.estimate,
        scale=scale,
        ci_low=bootstrap.ci_low,
        ci_high=bootstrap.ci_high,
        confidence_level=bootstrap.confidence_level,
        ci_method="percentile bootstrap over mice",
        n_mice=bootstrap.n_mice,
        n_sessions=int(mice.get_column(n_sessions_column).sum()),
        n_trials=n_trials,
        bootstrap_id=f"seed={seed};n_resamples={bootstrap.n_resamples}",
        inclusion_definition=inclusion_definition,
    )
    if run_sign_flip:
        sign_flip = dg.statistics.two_sided_sign_flip_test(
            mice,
            seed=seed,
            mouse_column=mouse_column,
            value_column=value_column,
            n_resamples=sign_flip_resamples,
        )
        row.update(
            {
                "test_statistic": sign_flip.statistic,
                "test_method": f"{sign_flip.method};n_permutations={sign_flip.n_permutations}",
                "p_value": sign_flip.p_value,
                "multiplicity_family": multiplicity_family,
                "sidedness": "two-sided",
            }
        )
    return row


def _base_statistics_row(
    *,
    result_id: str,
    contrast_id: str,
    hypothesis: str,
    code_version: str,
    status: str,
    estimate: float | None,
    scale: str,
    aggregation: str = AGGREGATION,
    inclusion_definition: str = INCLUSION_DEFINITION,
    reason: str | None = None,
    **values: Any,
) -> dict[str, Any]:
    return {
        "analysis_id": ANALYSIS_ID,
        "result_id": result_id,
        "contrast_id": contrast_id,
        "hypothesis": hypothesis,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "code_version": code_version,
        "analysis_tier": ANALYSIS_TIER,
        "inclusion_definition": inclusion_definition,
        "missingness_stratum": "complete technically valid behavior",
        "estimate": estimate,
        "scale": scale,
        "aggregation": aggregation,
        "status": status,
        "reason": reason,
        **values,
    }


def _read_completed_audit(
    path: pathlib.Path,
    *,
    artifact_paths: dict[str, pathlib.Path] | None = None,
) -> dict[str, Any]:
    """Read a passing M0 marker and optionally verify exact bound artifacts."""

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read completed M0 audit {path}: {error}") from error
    if record.get("run_status") != "complete" or record.get("authoritative") is not True:
        raise RuntimeError("M0 audit is not a completed authoritative run")
    if record.get("dandiset_version") != dg.data.DANDISET_VERSION:
        raise RuntimeError("M0 audit Dandiset version does not match the analysis code")
    if record.get("asset_inventory_status") != "pass":
        raise RuntimeError("M0 asset-inventory audit has not passed")
    if record.get("schema_audit_status") != "pass":
        raise RuntimeError("M0 full NWB schema audit has not passed")
    if record.get("task_parameters_audit_status") != "pass":
        raise RuntimeError("M0 task-parameter audit has not passed")
    if artifact_paths is not None:
        if record.get("manifest_schema_version") != M0_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                "M0 audit manifest schema does not match the behavior-analysis contract"
            )
        artifact_records = record.get("artifacts")
        if not isinstance(artifact_records, dict):
            raise RuntimeError("M0 audit manifest lacks content-addressed artifacts")
        result_root = path.resolve().parent.parent
        for artifact_id, artifact_path in artifact_paths.items():
            expected = artifact_records.get(artifact_id)
            if not isinstance(expected, dict):
                raise RuntimeError(f"M0 audit lacks required artifact {artifact_id!r}")
            relative_path = expected.get("path")
            if not isinstance(relative_path, str) or not relative_path:
                raise RuntimeError(f"M0 artifact {artifact_id!r} lacks a path")
            bound_path = (result_root / relative_path).resolve()
            if not bound_path.is_relative_to(result_root):
                raise RuntimeError(f"M0 artifact {artifact_id!r} escapes the results root")
            if bound_path != artifact_path.resolve():
                raise RuntimeError(
                    f"M0 artifact {artifact_id!r} path differs from the behavior input"
                )
            observed = _file_provenance_record(artifact_path)
            if (
                expected.get("sha256") != observed["sha256"]
                or expected.get("size_bytes") != observed["size_bytes"]
            ):
                raise RuntimeError(f"M0 artifact {artifact_id!r} content differs from its manifest")
    return record


def _read_validated_task_parameters_audit(
    path: pathlib.Path,
    inventory: pl.DataFrame,
) -> str:
    """Validate the audited response window and exact inventory coverage."""

    try:
        audit = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"could not read M0 task-parameter audit {path}: {error}") from error
    try:
        dg.audit.validate_task_parameters_audit(audit)
    except ValueError as error:
        raise RuntimeError(f"M0 task-parameter audit is invalid: {error}") from error
    _require_columns(inventory, ("asset_id",), frame_name="session_inventory")
    expected_assets = inventory.select("asset_id").sort("asset_id")
    observed_assets = audit.select("asset_id").sort("asset_id")
    if not expected_assets.equals(observed_assets):
        raise RuntimeError("M0 task-parameter audit asset coverage differs from session inventory")
    return _sha256_file(path)


def _validate_analysis_arguments(arguments: argparse.Namespace) -> None:
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")
    if arguments.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    if arguments.sign_flip_resamples < 1:
        raise ValueError("--sign-flip-resamples must be positive")


def _validate_locked_dandiset(lock: dict[str, Any]) -> None:
    expected = f"DANDI:{dg.data.DANDISET_ID}/{dg.data.DANDISET_VERSION}"
    observed = lock["decisions"]["D01"].get("value")
    if observed != expected:
        message = (
            "analysis lock D01 does not match the immutable data target: "
            f"{observed!r} != {expected!r}"
        )
        raise RuntimeError(message)


def _read_session_inventory(path: pathlib.Path) -> pl.DataFrame:
    try:
        inventory = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"could not read session inventory {path}: {error}") from error
    required = ("_nwb_path", "subject_id", "ecephys_session_id", "dandiset_version")
    _require_columns(inventory, required, frame_name="session_inventory")
    _validate_unique_non_null(inventory, "_nwb_path", frame_name="session_inventory")
    _validate_unique_non_null(inventory, "ecephys_session_id", frame_name="session_inventory")
    if inventory.get_column("subject_id").null_count():
        raise RuntimeError("session_inventory.subject_id contains nulls")
    versions = inventory.get_column("dandiset_version").unique().to_list()
    if versions != [dg.data.DANDISET_VERSION]:
        raise RuntimeError(f"session inventory has unexpected Dandiset versions: {versions}")
    return inventory.sort("_nwb_path")


def _analysis_source_provenance() -> dict[str, Any]:
    """Return exact hashes for the orchestration script and local modules it uses."""

    return {
        "generator": _file_provenance_record(pathlib.Path(__file__).resolve()),
        "local_sources": [
            _file_provenance_record(REPOSITORY_ROOT / relative_path)
            for relative_path in LOCAL_ANALYSIS_MODULE_PATHS
        ],
    }


def _file_provenance_record(path: pathlib.Path) -> dict[str, Any]:
    """Hash one stable file and return a portable provenance record."""

    resolved = path.resolve()
    try:
        before = resolved.stat()
        digest = _sha256_file(resolved)
        after = resolved.stat()
    except OSError as error:
        raise RuntimeError(f"could not fingerprint provenance file {resolved}: {error}") from error
    identity_before = (before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"provenance file changed while it was hashed: {resolved}")
    return {
        "path": _display_path(resolved),
        "sha256": digest,
        "size_bytes": after.st_size,
    }


def _validate_file_provenance_record(record: Any, *, label: str) -> dict[str, Any]:
    """Validate the shape of a file provenance record from JSON."""

    if not isinstance(record, dict):
        raise RuntimeError(f"behavior-trials checkpoint {label} record must be an object")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"behavior-trials checkpoint {label} path is missing")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"behavior-trials checkpoint {label} SHA-256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"behavior-trials checkpoint {label} size is invalid")
    return record


def _require_checkpoint_file_match(
    recorded: Any,
    current_path: pathlib.Path,
    *,
    label: str,
) -> None:
    """Fail unless a checkpoint file record matches the current file exactly."""

    expected = _validate_file_provenance_record(recorded, label=label)
    observed = _file_provenance_record(current_path)
    for field in ("path", "size_bytes", "sha256"):
        if expected[field] != observed[field]:
            raise RuntimeError(
                f"behavior-trials checkpoint {label} {field} does not match the current file"
            )


def _write_behavior_trials_checkpoint_manifest(
    destination: pathlib.Path,
    *,
    behavior_trials_path: pathlib.Path,
    session_inventory_path: pathlib.Path,
    companion_trials_path: pathlib.Path,
    companion_provenance: dict[str, Any],
    source_provenance: dict[str, Any],
    created_at: datetime.datetime,
    trial_input_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the trusted trial checkpoint, only after a direct remote-NWB run."""

    if trial_input_mode != "remote_nwb":
        raise RuntimeError("trusted behavior-trials checkpoints require a direct remote-NWB run")
    generator = _validate_file_provenance_record(
        source_provenance.get("generator"),
        label="generator",
    )
    local_sources = source_provenance.get("local_sources")
    if not isinstance(local_sources, list) or not local_sources:
        raise RuntimeError("behavior-trials checkpoint local_sources must be a non-empty list")
    validated_local_sources = [
        _validate_file_provenance_record(record, label=f"local_sources[{index}]")
        for index, record in enumerate(local_sources)
    ]
    companion_record = _file_provenance_record(companion_trials_path)
    if companion_provenance.get("sha256") != companion_record["sha256"]:
        raise RuntimeError("companion-trials provenance SHA-256 differs from the cached file")
    if companion_provenance.get("content_size_bytes") != companion_record["size_bytes"]:
        raise RuntimeError("companion-trials provenance size differs from the cached file")
    companion_record.update(
        {
            key: companion_provenance.get(key)
            for key in ("source_url", "repository_commit")
            if companion_provenance.get(key) is not None
        }
    )
    checkpoint = {
        "schema_version": BEHAVIOR_TRIALS_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_kind": "direct_remote_nwb_behavior_trials",
        "created_at_utc": created_at.isoformat(),
        "trial_input_mode": trial_input_mode,
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "behavior_trials": _file_provenance_record(behavior_trials_path),
        "session_inventory": _file_provenance_record(session_inventory_path),
        "companion_trials": companion_record,
        "generator": dict(generator),
        "local_sources": [dict(record) for record in validated_local_sources],
    }
    dg.artifacts.write_json(checkpoint, destination)
    return checkpoint, _file_provenance_record(destination)


def _read_validated_behavior_trials_checkpoint_manifest(
    checkpoint_path: pathlib.Path,
    *,
    behavior_trials_path: pathlib.Path,
    session_inventory_path: pathlib.Path,
    companion_trials_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a trusted checkpoint and match all current data inputs exactly."""

    if not checkpoint_path.is_file():
        raise RuntimeError(f"trusted behavior-trials checkpoint is missing: {checkpoint_path}")
    try:
        checkpoint_manifest_record = _file_provenance_record(checkpoint_path)
        payload = checkpoint_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != checkpoint_manifest_record["sha256"]:
            raise RuntimeError("behavior-trials checkpoint manifest changed while it was read")
        checkpoint = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"could not read trusted behavior-trials checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(checkpoint, dict):
        raise RuntimeError("behavior-trials checkpoint manifest must contain a JSON object")
    if checkpoint.get("schema_version") != BEHAVIOR_TRIALS_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("behavior-trials checkpoint schema version is unsupported")
    if checkpoint.get("checkpoint_kind") != "direct_remote_nwb_behavior_trials":
        raise RuntimeError("behavior-trials checkpoint was not produced by a direct remote-NWB run")
    if checkpoint.get("trial_input_mode") != "remote_nwb":
        raise RuntimeError("behavior-trials checkpoint has an invalid trial input mode")
    if checkpoint.get("dandiset_id") != dg.data.DANDISET_ID:
        raise RuntimeError("behavior-trials checkpoint Dandiset ID does not match analysis code")
    if checkpoint.get("dandiset_version") != dg.data.DANDISET_VERSION:
        raise RuntimeError(
            "behavior-trials checkpoint Dandiset version does not match analysis code"
        )
    _validate_file_provenance_record(checkpoint.get("generator"), label="generator")
    local_sources = checkpoint.get("local_sources")
    if not isinstance(local_sources, list) or not local_sources:
        raise RuntimeError("behavior-trials checkpoint local_sources must be a non-empty list")
    for index, record in enumerate(local_sources):
        _validate_file_provenance_record(record, label=f"local_sources[{index}]")
    _require_checkpoint_file_match(
        checkpoint.get("behavior_trials"),
        behavior_trials_path,
        label="behavior_trials",
    )
    _require_checkpoint_file_match(
        checkpoint.get("session_inventory"),
        session_inventory_path,
        label="session_inventory",
    )
    _require_checkpoint_file_match(
        checkpoint.get("companion_trials"),
        companion_trials_path,
        label="companion_trials",
    )
    return checkpoint, checkpoint_manifest_record


def _read_behavior_trials_checkpoint(
    path: pathlib.Path,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Validate and return a previously written full behavior-trial table."""

    try:
        trials = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"could not read behavior trial checkpoint {path}: {error}") from error
    _require_columns(
        trials,
        ("_nwb_path", "_table_index", *dg.data.BEHAVIOR_TRIAL_COLUMNS),
        frame_name="behavior_trials checkpoint",
    )
    if trials.select("_nwb_path", "_table_index").n_unique() != trials.height:
        raise RuntimeError("behavior trial checkpoint has duplicate source/table-index keys")
    expected_sources = inventory.select("_nwb_path").unique().sort("_nwb_path")
    observed_sources = trials.select("_nwb_path").unique().sort("_nwb_path")
    if not expected_sources.equals(observed_sources):
        raise RuntimeError("behavior trial checkpoint session coverage differs from inventory")
    if "dandiset_version" in trials.columns:
        versions = trials.get_column("dandiset_version").drop_nulls().unique().to_list()
        if versions != [dg.data.DANDISET_VERSION]:
            raise RuntimeError(
                f"behavior trial checkpoint has unexpected Dandiset versions: {versions}"
            )
    return trials.sort("_nwb_path", "_table_index")


def _attach_trial_metadata(trials: pl.DataFrame, inventory: pl.DataFrame) -> pl.DataFrame:
    """Join immutable session identifiers without duplicating reward-join fields."""

    columns = [
        column
        for column in (
            "_nwb_path",
            "subject_id",
            "asset_id",
            "dandiset_id",
            "dandiset_version",
            "behavior_session_id",
            "recording_day",
            "session_number",
            "sex",
            "genotype",
            "project_code",
            "novel_image_id",
        )
        if column in inventory.columns and column not in trials.columns
    ]
    if "_nwb_path" not in columns:
        columns.insert(0, "_nwb_path")
    return trials.join(inventory.select(*columns), on="_nwb_path", how="left", validate="m:1").sort(
        "_nwb_path", "_table_index"
    )


def _attach_session_metadata(
    summary: pl.DataFrame,
    inventory: pl.DataFrame,
    *,
    join_validation: str = "1:1",
) -> pl.DataFrame:
    """Put immutable inventory fields beside session-keyed summary rows."""

    summary_without_mouse = summary.drop("subject_id")
    return inventory.join(
        summary_without_mouse,
        on="_nwb_path",
        how="left",
        validate=join_validation,
    )


def _require_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def _validate_unique_non_null(
    frame: pl.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> None:
    if frame.get_column(column).null_count():
        raise RuntimeError(f"{frame_name}.{column} contains nulls")
    if frame.get_column(column).n_unique() != frame.height:
        raise RuntimeError(f"{frame_name}.{column} must be unique")


def _code_version(repository: pathlib.Path) -> str:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{revision}+dirty" if dirty else revision


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
