# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Freeze data provenance and audit schemas, coverage, and scalar event clocks."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import pathlib
import signal
import sys

import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.audit  # noqa: E402
import dg.data  # noqa: E402
import dg.gates  # noqa: E402
import dg.inventory  # noqa: E402
import dg.m0_integrity  # noqa: E402
import dg.task_presentations  # noqa: E402

EXPECTED_ASSET_COUNTS = {
    "probe_lfp_nwb": 567,
    "session_nwb": 99,
}
EXPECTED_SESSION_SUBJECTS = 27
AUDIT_MANIFEST_SCHEMA_VERSION = 3
EXPECTED_EYE_ABSENT_SESSION_KEYS = (
    ("614608", 1187475832),
    ("623784", 1202438738),
)
LOCAL_SOURCE_PATHS = (
    "scripts/00_freeze_and_audit.py",
    "src/dg/__init__.py",
    "src/dg/artifacts.py",
    "src/dg/audit.py",
    "src/dg/data.py",
    "src/dg/gates.py",
    "src/dg/inventory.py",
    "src/dg/m0_integrity.py",
    "src/dg/quality.py",
    "src/dg/task_presentations.py",
    "config/analysis_lock.yaml",
    "pyproject.toml",
    "uv.lock",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Artifact root (default: repository results/)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent NWB metadata reads for schema and task-parameter audits",
    )
    parser.add_argument(
        "--presentation-workers",
        type=int,
        default=dg.task_presentations.DEFAULT_MAX_WORKERS,
        help="Concurrent isolated task-presentation session workers (default: 4; maximum: 8)",
    )
    parser.add_argument(
        "--presentation-timeout-seconds",
        type=float,
        default=dg.task_presentations.DEFAULT_TIMEOUT_SECONDS,
        help="Wall-clock limit for each isolated presentation attempt (default: 300)",
    )
    parser.add_argument(
        "--presentation-max-attempts",
        type=int,
        default=dg.task_presentations.DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts per task-presentation session (default: 3; maximum: 5)",
    )
    parser.add_argument(
        "--presentation-batch-size",
        type=int,
        default=dg.task_presentations.DEFAULT_BATCH_SIZE,
        help="Requested projected-row collection batch size (default: 50000)",
    )
    parser.add_argument(
        "--skip-schema-audit",
        action="store_true",
        help=(
            "Write the asset/companion freeze without opening session NWBs; this also "
            "skips unit, stimulus, and event-clock inventories"
        ),
    )
    parser.add_argument(
        "--audit-nwb-root-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reconcile the three scalar identity fields from every remote NWB root "
            "(default: enabled; use --no-audit-nwb-root-metadata to skip explicitly)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    started_at = datetime.datetime.now(datetime.UTC)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    audit_manifest_path = result_root / "manifests" / "audit_run.json"
    in_progress = {
        "manifest_schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
        "run_status": "in_progress",
        "authoritative": False,
        "polars_thread_pool_size": pl.thread_pool_size(),
        "script": str(pathlib.Path(__file__).relative_to(REPOSITORY_ROOT)),
        "started_at_utc": started_at.isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
    }
    # This marker is published before any other artifact, so an interrupted
    # rerun can never leave a stale successful manifest looking authoritative.
    dg.artifacts.write_json(in_progress, audit_manifest_path)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_system_exit_on_sigterm)
    try:
        try:
            _run_audit(arguments, started_at=started_at, result_root=result_root)
        except BaseException as error:
            failed = {
                **in_progress,
                "run_status": "failed",
                "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            with contextlib.suppress(Exception):
                dg.artifacts.write_json(failed, audit_manifest_path)
            raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _run_audit(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    result_root: pathlib.Path,
) -> None:
    authoritative_mode = _is_authoritative_mode(arguments)
    manifests = result_root / "manifests"
    cache = result_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    artifact_specs: dict[str, tuple[pathlib.Path, int | None]] = {}
    local_sources_at_start = _local_source_records()
    _report_phase("frozen asset and companion provenance")

    lock_source = REPOSITORY_ROOT / "config" / "analysis_lock.yaml"
    lock = dg.gates.read_analysis_lock(lock_source)
    lock_snapshot_path = manifests / "analysis_lock.yaml"
    dg.artifacts.write_text(
        lock_source.read_text(encoding="utf-8"),
        lock_snapshot_path,
    )
    artifact_specs["analysis_lock_snapshot"] = (lock_snapshot_path, None)

    manifest = dg.audit.fetch_asset_manifest()
    inventory = dg.audit.summarize_asset_manifest(manifest)
    _validate_frozen_inventory(manifest, inventory)
    asset_manifest_path = manifests / "dandiset_assets.parquet"
    asset_inventory_path = manifests / "asset_inventory.csv"
    dg.artifacts.write_frame(manifest, asset_manifest_path)
    dg.artifacts.write_frame(inventory, asset_inventory_path)
    artifact_specs["dandiset_assets"] = (asset_manifest_path, manifest.height)
    artifact_specs["asset_inventory"] = (asset_inventory_path, inventory.height)

    companion_path = cache / "master_stim_trials_table.csv"
    companion = dg.audit.cache_companion_trials(companion_path)
    companion_input_at_start = _artifact_record(
        companion_path,
        root=result_root,
        n_rows=None,
    )
    companion_provenance_path = manifests / "companion_trials_provenance.csv"
    dg.artifacts.write_frame(companion, companion_provenance_path)
    artifact_specs["companion_trials_provenance"] = (
        companion_provenance_path,
        companion.height,
    )
    session_metadata_path = cache / "dynamic_gating_session_metadata.csv"
    session_metadata = dg.audit.cache_companion_session_metadata(session_metadata_path)
    session_metadata_input_at_start = _artifact_record(
        session_metadata_path,
        root=result_root,
        n_rows=None,
    )
    session_metadata_provenance_path = manifests / "companion_session_metadata_provenance.csv"
    dg.artifacts.write_frame(session_metadata, session_metadata_provenance_path)
    artifact_specs["companion_session_metadata_provenance"] = (
        session_metadata_provenance_path,
        session_metadata.height,
    )
    session_inventory = _reconcile_session_inventory(manifest, session_metadata_path)
    nwb_root_metadata_status = "skipped_by_request"
    if arguments.audit_nwb_root_metadata:
        _report_phase("NWB root identifier reconciliation")
        session_inventory, root_identifier_audit = _add_nwb_root_identifier_audit(
            session_inventory,
            max_workers=arguments.max_workers,
        )
        root_identifier_path = manifests / "nwb_root_identifier_audit.csv"
        dg.artifacts.write_frame(root_identifier_audit, root_identifier_path)
        artifact_specs["nwb_root_identifier_audit"] = (
            root_identifier_path,
            root_identifier_audit.height,
        )
        dg.audit.validate_nwb_root_identifier_audit(
            root_identifier_audit,
            session_inventory,
        )
        nwb_root_metadata_status = "pass"
    session_inventory_path = manifests / "session_inventory.parquet"
    dg.artifacts.write_frame(session_inventory, session_inventory_path)
    artifact_specs["session_inventory"] = (session_inventory_path, session_inventory.height)

    sessions = manifest.filter(pl.col("asset_kind") == "session_nwb")
    schema_status = "skipped_by_request"
    schema_dtype_drift_status = "skipped_by_request"
    clock_path_coverage_status = "skipped_by_request"
    task_parameters_status = "skipped_by_request"
    unit_inventory_status = "skipped_by_request"
    raw_stimulus_inventory_status = "skipped_by_request"
    scalar_interval_clock_screen_status = "skipped_by_request"
    rowwise_event_alignment_status = "skipped_by_request"
    stimulus_semantics_status = "skipped_by_request"
    stimulus_estimability_status = "skipped_by_request"
    task_presentation_materialization_status = "skipped_by_request"
    coverage_counts: dict[str, int] = {}
    if not arguments.skip_schema_audit:
        _report_phase("table-path/schema and task-parameter audit")
        schema_audit = dg.audit.audit_nwb_schemas(
            sessions,
            max_workers=arguments.max_workers,
        )
        schema_summary = dg.audit.summarize_schema_audit(schema_audit)
        schema_audit_path = manifests / "schema_audit.parquet"
        schema_summary_path = manifests / "schema_summary.csv"
        dg.artifacts.write_frame(schema_audit, schema_audit_path)
        dg.artifacts.write_frame(schema_summary, schema_summary_path)
        artifact_specs["schema_audit"] = (schema_audit_path, schema_audit.height)
        artifact_specs["schema_summary"] = (schema_summary_path, schema_summary.height)

        task_parameters_audit = dg.audit.audit_task_parameters(
            sessions,
            max_workers=arguments.max_workers,
        )
        task_parameters_path = manifests / "task_parameters_audit.csv"
        dg.artifacts.write_frame(task_parameters_audit, task_parameters_path)
        artifact_specs["task_parameters_audit"] = (
            task_parameters_path,
            task_parameters_audit.height,
        )

        _validate_required_schema(schema_audit)
        dg.audit.validate_task_parameters_audit(task_parameters_audit)
        schema_status = "pass"
        task_parameters_status = "pass"

        _report_phase("per-session exact schema dtypes and variant counts")
        task_schema_paths = (
            dg.inventory.resolve_task_presentation_paths(session_inventory, schema_audit)
            .rename({"task_presentation_path": "table_path"})
            .with_columns(pl.lit("task_image_presentations").alias("table_id"))
        )
        exact_table_specs = (
            *dg.m0_integrity.DEFAULT_EXACT_TABLE_SPECS,
            dg.m0_integrity.TASK_PRESENTATION_TABLE_SPEC,
        )
        schema_dtype_audit = dg.m0_integrity.audit_exact_table_schema_metadata(
            session_inventory,
            table_specs=exact_table_specs,
            source_table_paths=task_schema_paths,
            max_workers=arguments.max_workers,
        )
        schema_dtype_audit_path = manifests / "schema_dtype_audit.parquet"
        dg.artifacts.write_frame(schema_dtype_audit, schema_dtype_audit_path)
        artifact_specs["schema_dtype_audit"] = (
            schema_dtype_audit_path,
            schema_dtype_audit.height,
        )
        # Publish source-scoped diagnostics before the hard gate. The run
        # marker remains non-authoritative unless every gate later succeeds.
        dg.m0_integrity.validate_exact_table_schema_audit(
            schema_dtype_audit,
            session_inventory,
            table_specs=exact_table_specs,
            source_table_paths=task_schema_paths,
            expected_schema_variant_counts=(dg.m0_integrity.FROZEN_EXPECTED_SCHEMA_VARIANT_COUNTS),
            maximum_schema_variant_counts=(dg.m0_integrity.FROZEN_EXPECTED_SCHEMA_VARIANT_COUNTS),
        )
        schema_dtype_summary = dg.m0_integrity.summarize_exact_table_schema_audit(
            schema_dtype_audit
        )
        schema_dtype_summary_path = manifests / "schema_dtype_summary.csv"
        dg.artifacts.write_frame(schema_dtype_summary, schema_dtype_summary_path)
        artifact_specs["schema_dtype_summary"] = (
            schema_dtype_summary_path,
            schema_dtype_summary.height,
        )
        schema_dtype_drift_status = "pass_exact_per_session_physical_dtypes"

        _report_phase("behavioral/acquisition clock path and shape coverage")
        clock_metadata_audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
            session_inventory,
            max_workers=arguments.max_workers,
        )
        raw_video_asset_audit = dg.m0_integrity.audit_raw_video_asset_absence(manifest)
        clock_diagnostic_outputs = {
            "behavior_acquisition_clock_inventory": (
                clock_metadata_audit.series,
                manifests / "behavior_acquisition_clock_inventory.parquet",
            ),
            "behavior_acquisition_clock_components": (
                clock_metadata_audit.components,
                manifests / "behavior_acquisition_clock_components.parquet",
            ),
            "raw_video_internal_path_audit": (
                clock_metadata_audit.raw_video,
                manifests / "raw_video_internal_path_audit.csv",
            ),
            "raw_video_asset_audit": (
                raw_video_asset_audit,
                manifests / "raw_video_asset_audit.csv",
            ),
        }
        _publish_frame_outputs(clock_diagnostic_outputs, artifact_specs)
        expected_eye_absences = _expected_eye_absent_sources(session_inventory)
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            clock_metadata_audit,
            session_inventory,
            expected_optional_absent_sources=expected_eye_absences,
        )
        clock_metadata_summary = dg.m0_integrity.summarize_behavior_acquisition_clock_metadata(
            clock_metadata_audit.series
        )
        _validate_raw_video_absence(clock_metadata_audit.raw_video, raw_video_asset_audit)
        clock_outputs = {
            "behavior_acquisition_clock_summary": (
                clock_metadata_summary,
                manifests / "behavior_acquisition_clock_summary.csv",
            ),
        }
        _publish_frame_outputs(clock_outputs, artifact_specs)
        clock_path_coverage_status = "pass_metadata_only_full_vector_audit_partial"

        _report_phase("projected scalar unit/electrode metadata inventory")
        unit_inventory = dg.inventory.scan_unit_inventory(session_inventory).collect()
        dg.inventory.validate_unit_inventory(unit_inventory, session_inventory)
        unit_session_reconciliation = dg.inventory.summarize_unit_session_reconciliation(
            unit_inventory,
            session_inventory,
        )
        unit_anatomy_coverage = dg.inventory.summarize_unit_anatomy_coverage(unit_inventory)
        unit_session_anatomy_coverage = dg.inventory.summarize_unit_session_anatomy_coverage(
            unit_inventory
        )

        _report_phase("isolated projected row-level task-presentation inventory")
        task_presentation_materialization = dg.task_presentations.materialize_isolated(
            session_inventory,
            schema_audit,
            max_workers=arguments.presentation_workers,
            timeout_seconds=arguments.presentation_timeout_seconds,
            max_attempts=arguments.presentation_max_attempts,
            batch_size=arguments.presentation_batch_size,
        )
        task_presentations = task_presentation_materialization.presentations
        task_presentation_materialization_audit = task_presentation_materialization.audit
        dg.inventory.validate_task_presentations(task_presentations, session_inventory)

        _report_phase("projected row-level trials and audited reward blocks")
        companion_reward_epochs = dg.data.scan_companion_trial_reward_epochs(
            companion_path
        ).collect()
        trial_inventory = dg.inventory.scan_trial_inventory(
            session_inventory,
            companion_reward_epochs,
        ).collect()
        task_structure_audit = dg.inventory.validate_trial_inventory(
            trial_inventory,
            session_inventory,
        )

        _report_phase("local frame alignment and stimulus-design integrity")
        trial_presentation_alignment = dg.inventory.align_trials_to_presentations(
            trial_inventory,
            task_presentations,
        )
        trial_presentation_alignment_summary = dg.inventory.validate_trial_presentation_alignment(
            trial_presentation_alignment,
            trial_inventory,
            task_presentations,
            session_inventory,
        )
        stimulus_semantics_audit = dg.inventory.summarize_stimulus_semantics(
            task_presentations,
            trial_presentation_alignment,
        )
        dg.inventory.validate_stimulus_semantics(
            stimulus_semantics_audit,
            session_inventory,
        )
        stimulus_estimability_summary = dg.inventory.summarize_stimulus_estimability(
            trial_presentation_alignment,
            session_inventory,
        )
        dg.inventory.validate_stimulus_estimability(
            stimulus_estimability_summary,
            session_inventory,
        )
        stimulus_design_aliases = dg.inventory.build_stimulus_design_aliases(
            task_presentations,
            stimulus_estimability_summary,
        )
        dg.inventory.validate_stimulus_design_aliases(stimulus_design_aliases)
        stimulus_inventory = dg.inventory.summarize_stimulus_inventory(task_presentations)
        stimulus_coverage = dg.inventory.summarize_stimulus_coverage(stimulus_inventory)

        trial_clock = dg.inventory.summarize_interval_clock(
            trial_inventory,
            table_kind="trials",
            event_time_column="change_time",
        )
        presentation_clock = dg.inventory.summarize_interval_clock(
            task_presentations,
            table_kind="task_presentations",
        )
        event_clock_audit = dg.inventory.combine_event_clock_audits(
            trial_clock,
            presentation_clock,
            session_inventory,
        )
        event_dictionary = dg.inventory.build_event_dictionary(
            event_clock_audit,
            trial_presentation_alignment_summary,
        )

        coverage_outputs = {
            "unit_inventory": (unit_inventory, manifests / "unit_inventory.parquet"),
            "unit_session_reconciliation": (
                unit_session_reconciliation,
                manifests / "unit_session_reconciliation.csv",
            ),
            "unit_anatomy_coverage": (
                unit_anatomy_coverage,
                manifests / "unit_anatomy_coverage.csv",
            ),
            "unit_session_anatomy_coverage": (
                unit_session_anatomy_coverage,
                manifests / "unit_session_anatomy_coverage.csv",
            ),
            "raw_stimulus_inventory": (
                stimulus_inventory,
                manifests / "raw_stimulus_inventory.parquet",
            ),
            "task_presentations_projected": (
                task_presentations,
                manifests / "task_presentations_projected.parquet",
            ),
            "task_presentation_materialization_audit": (
                task_presentation_materialization_audit,
                manifests / "task_presentation_materialization_audit.csv",
            ),
            "trials_projected": (
                trial_inventory,
                manifests / "trials_projected.parquet",
            ),
            "trial_task_structure_audit": (
                task_structure_audit,
                manifests / "trial_task_structure_audit.csv",
            ),
            "trial_presentation_alignment": (
                trial_presentation_alignment,
                manifests / "trial_presentation_alignment.parquet",
            ),
            "trial_presentation_alignment_summary": (
                trial_presentation_alignment_summary,
                manifests / "trial_presentation_alignment_summary.csv",
            ),
            "stimulus_semantics_audit": (
                stimulus_semantics_audit,
                manifests / "stimulus_semantics_audit.parquet",
            ),
            "stimulus_estimability_summary": (
                stimulus_estimability_summary,
                manifests / "stimulus_estimability_summary.csv",
            ),
            "stimulus_design_aliases": (
                stimulus_design_aliases,
                manifests / "stimulus_design_aliases.csv",
            ),
            "raw_stimulus_coverage": (
                stimulus_coverage,
                manifests / "raw_stimulus_coverage.csv",
            ),
            "scalar_interval_clock_screen": (
                event_clock_audit,
                manifests / "scalar_interval_clock_screen.csv",
            ),
            "scalar_event_dictionary": (
                event_dictionary,
                manifests / "scalar_event_dictionary.csv",
            ),
        }
        _publish_frame_outputs(coverage_outputs, artifact_specs)

        # Write the complete audit tables before applying the hard gate so a
        # failed run leaves inspectable diagnostics, but never records a pass.
        dg.inventory.validate_event_clock_audit(event_clock_audit, session_inventory)
        unit_inventory_status = "pass"
        raw_stimulus_inventory_status = "pass_projected_raw_rows_and_coverage"
        task_presentation_materialization_status = (
            "pass_all_sources_isolated_bounded_timeout_and_retries"
        )
        scalar_interval_clock_screen_status = "pass_limited_scope"
        rowwise_event_alignment_status = "pass_unique_session_frame_origin_and_label_integrity"
        stimulus_semantics_status = "internally_consistent_history_unverified"
        stimulus_estimability_status = "pass_state_x_contrast_within_im115"
        coverage_counts = {
            "n_units": unit_inventory.height,
            "n_well_isolated_units": int(unit_inventory.get_column("well_isolated").sum()),
            "n_unit_sessions": unit_inventory.get_column("_nwb_path").n_unique(),
            "n_task_presentations": task_presentations.height,
            "n_task_presentation_materialization_sessions": (
                task_presentation_materialization_audit.height
            ),
            "n_trials": trial_inventory.height,
            "n_trial_change_anchors": trial_presentation_alignment.height,
            "n_aligned_trial_change_anchors": int(
                trial_presentation_alignment.get_column("trial_presentation_alignment_pass").sum()
            ),
            "n_stimulus_inventory_rows": stimulus_inventory.height,
            "n_stimulus_sessions": task_presentations.get_column("_nwb_path").n_unique(),
            "n_stimulus_estimability_session_states": stimulus_estimability_summary.height,
            "n_clock_sessions": event_clock_audit.height,
            "n_schema_dtype_audit_rows": schema_dtype_audit.height,
            "n_schema_dtype_sessions": schema_dtype_audit.get_column("_nwb_path").n_unique(),
            "n_clock_metadata_series_rows": clock_metadata_audit.series.height,
            "n_clock_metadata_sessions": clock_metadata_audit.series.get_column(
                "_nwb_path"
            ).n_unique(),
            "n_eye_tracking_sessions_present": int(
                clock_metadata_audit.series.filter(pl.col("signal_id") == "eye_tracking")
                .get_column("series_present")
                .sum()
            ),
        }

    _report_phase("atomic artifact publication and provenance hashes")
    environment = dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT)
    environment_path = manifests / "software_environment.txt"
    dg.artifacts.write_text(environment, environment_path)
    artifact_specs["software_environment"] = (environment_path, None)
    artifacts = {
        name: _artifact_record(path, root=result_root, n_rows=n_rows)
        for name, (path, n_rows) in sorted(artifact_specs.items())
    }
    inputs = {
        "companion_trials_cache": _artifact_record(
            companion_path,
            root=result_root,
            n_rows=None,
        ),
        "companion_session_metadata_cache": _artifact_record(
            session_metadata_path,
            root=result_root,
            n_rows=None,
        ),
    }
    expected_inputs = {
        "companion_trials_cache": companion_input_at_start,
        "companion_session_metadata_cache": session_metadata_input_at_start,
    }
    if inputs != expected_inputs:
        raise RuntimeError("one or more companion input caches changed during the audit")
    local_sources = _local_source_records()
    if local_sources != local_sources_at_start:
        raise RuntimeError("one or more local source files changed during the audit")
    if (
        artifacts["analysis_lock_snapshot"]["sha256"]
        != local_sources["config/analysis_lock.yaml"]["sha256"]
    ):
        raise RuntimeError("analysis lock changed while audit artifacts were being generated")
    completed_at = datetime.datetime.now(datetime.UTC)
    dg.artifacts.write_json(
        {
            "manifest_schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
            "run_status": "complete",
            "authoritative": authoritative_mode,
            "run_scope": (
                "full_frozen_audit" if authoritative_mode else "diagnostic_with_requested_skips"
            ),
            "script": str(pathlib.Path(__file__).relative_to(REPOSITORY_ROOT)),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "polars_thread_pool_size": pl.thread_pool_size(),
            "dandiset_id": dg.data.DANDISET_ID,
            "dandiset_version": dg.data.DANDISET_VERSION,
            "dandiset_doi": dg.data.DANDISET_DOI,
            "analysis_lock_status": lock["overall_status"],
            "asset_inventory_status": "pass",
            "schema_audit_status": schema_status,
            "schema_dtype_drift_audit_status": schema_dtype_drift_status,
            "task_parameters_audit_status": task_parameters_status,
            "unit_inventory_status": unit_inventory_status,
            "unit_metadata_inventory_status": (
                "pass_scalar_metadata_no_spike_arrays"
                if unit_inventory_status == "pass"
                else unit_inventory_status
            ),
            "raw_stimulus_inventory_status": raw_stimulus_inventory_status,
            "task_presentation_materialization_status": (task_presentation_materialization_status),
            "task_presentation_materialization_policy": {
                "isolation": "one_canonical_session_per_subprocess",
                "canonical_source_transport": "frozen_https_dandi_s3_url",
                "max_workers": arguments.presentation_workers,
                "timeout_seconds_per_attempt": arguments.presentation_timeout_seconds,
                "maximum_attempts_per_session": arguments.presentation_max_attempts,
                "requested_collect_batch_size_rows": arguments.presentation_batch_size,
                "retry_delay_seconds": (dg.task_presentations.DEFAULT_RETRY_DELAY_SECONDS),
                "attempt_outputs": "atomic_temporary_non_reusable",
                "failed_source_policy": "fail_entire_run",
            },
            "rowwise_event_alignment_status": rowwise_event_alignment_status,
            "rowwise_event_alignment_policy": {
                "frame_origin": (
                    "one unique integer offset per canonical session from exact "
                    "non-omitted stimulus-token and physical-change labels"
                ),
                "coverage": "full_one_to_one_coverage_of_every_finite_trial_anchor",
                "post_alignment_time_check": (
                    "matched_presentation_start_strictly_within_trial_interval"
                ),
                "timestamp_role": "descriptive_only_not_used_to_infer_or_select_offset",
            },
            "stimulus_novelty_contrast_semantics_status": stimulus_semantics_status,
            "stimulus_estimability_status": stimulus_estimability_status,
            "scalar_interval_clock_screen_status": scalar_interval_clock_screen_status,
            "clock_path_coverage_status": clock_path_coverage_status,
            "full_event_acquisition_clock_audit_status": (
                "partial_path_coverage_and_scalar_intervals_only"
            ),
            "event_dictionary_status": "partial_scalar_fields_and_path_coverage",
            "nwb_root_metadata_audit_status": nwb_root_metadata_status,
            "companion_table_status": companion.get_column("cache_status").item(),
            "milestone_0_status": "partial",
            "milestone_0_remaining": [
                "full-value validation of every behavioral/acquisition timestamp vector",
                "protocol/history validation of first-exposure novelty semantics",
                (
                    "optotagging epoch-table content and receptive-field/gabor/flash "
                    "stimulus inventories"
                ),
                (
                    "no standalone or name-matched embedded raw-video candidate was found; "
                    "generic ImageSeries detection remains unvalidated"
                ),
                "frozen Allen CCF ontology version and parent-region mapping",
            ],
            "neural_outcomes_accessed": False,
            "spike_arrays_loaded": False,
            "trial_timestamp_vectors_loaded": False,
            "neural_access_scope": "scalar_unit_metadata_only_no_activity_outcomes",
            "coverage_counts": coverage_counts,
            "artifacts": artifacts,
            "inputs": inputs,
            "local_sources": local_sources,
        },
        manifests / "audit_run.json",
    )
    print(f"Wrote immutable audit artifacts to {manifests}")


def _is_authoritative_mode(arguments: argparse.Namespace) -> bool:
    """Return whether all required canonical audit phases were requested."""

    return not arguments.skip_schema_audit and arguments.audit_nwb_root_metadata


def _publish_frame_outputs(
    outputs: dict[str, tuple[pl.DataFrame, pathlib.Path]],
    artifact_specs: dict[str, tuple[pathlib.Path, int | None]],
) -> None:
    """Write named tables and register their row-count provenance."""

    for name, output in outputs.items():
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(f"frame output {name!r} must be a (frame, destination) tuple")
        frame, destination = output
        if not isinstance(frame, pl.DataFrame) or not isinstance(destination, pathlib.Path):
            raise TypeError(f"frame output {name!r} has invalid frame or destination types")
        dg.artifacts.write_frame(frame, destination)
        artifact_specs[name] = (destination, frame.height)


def _validate_frozen_inventory(manifest: pl.DataFrame, inventory: pl.DataFrame) -> None:
    observed_counts = {
        row["asset_kind"]: row["n_assets"] for row in inventory.iter_rows(named=True)
    }
    if observed_counts != EXPECTED_ASSET_COUNTS:
        raise RuntimeError(
            f"immutable DANDI inventory differs from expected counts: {observed_counts}"
        )
    sessions = manifest.filter(pl.col("asset_kind") == "session_nwb")
    observed_subjects = sessions.get_column("subject_id").n_unique()
    if observed_subjects != EXPECTED_SESSION_SUBJECTS:
        raise RuntimeError(
            f"expected {EXPECTED_SESSION_SUBJECTS} session subjects, observed {observed_subjects}"
        )
    required = ("asset_id", "path", "s3_url", "dandi_etag", "sha256")
    if any(sessions.get_column(column).null_count() for column in required):
        raise RuntimeError("session manifest has missing IDs, paths, URLs, or digests")


def _validate_required_schema(schema_audit: pl.DataFrame) -> None:
    failures = schema_audit.filter(
        pl.col("required")
        & (pl.col("audit_error").is_not_null() | ~pl.col("present").fill_null(False))
    )
    if failures.height:
        example = failures.select("path", "table_path", "column_name", "audit_error").head(5)
        raise RuntimeError(f"required NWB schema audit failed:\n{example}")
    candidate_failures = (
        schema_audit.select(
            "path",
            "n_image_presentation_candidates",
            "image_presentation_paths",
        )
        .unique()
        .filter(pl.col("n_image_presentation_candidates") != 1)
    )
    if candidate_failures.height:
        raise RuntimeError(
            f"one or more sessions lack exactly one task-image table:\n{candidate_failures.head(5)}"
        )


def _expected_eye_absent_sources(
    session_inventory: pl.DataFrame,
) -> dict[str, set[str]]:
    """Resolve the two frozen eye-absent session keys to current immutable URLs."""

    required = {"subject_id", "ecephys_session_id", "_nwb_path"}
    missing = required.difference(session_inventory.columns)
    if missing:
        raise RuntimeError(f"session inventory lacks eye-coverage keys: {sorted(missing)}")
    expected_keys = set(EXPECTED_EYE_ABSENT_SESSION_KEYS)
    observed = session_inventory.filter(
        pl.struct("subject_id", "ecephys_session_id").map_elements(
            lambda row: (row["subject_id"], row["ecephys_session_id"]) in expected_keys,
            return_dtype=pl.Boolean,
        )
    )
    observed_keys = set(observed.select("subject_id", "ecephys_session_id").iter_rows())
    if observed_keys != expected_keys or observed.height != len(expected_keys):
        raise RuntimeError(
            "frozen eye-absent session keys do not resolve exactly in session inventory; "
            f"expected={sorted(expected_keys)}, observed={sorted(observed_keys)}"
        )
    absent_sources = set(observed.get_column("_nwb_path").to_list())
    return {
        spec.signal_id: absent_sources for spec in dg.m0_integrity.OPTIONAL_EYE_CLOCK_SERIES_SPECS
    }


def _validate_raw_video_absence(
    internal_path_audit: pl.DataFrame,
    asset_audit: pl.DataFrame,
) -> None:
    """Fail if path-name screening finds a raw-video candidate."""

    if internal_path_audit.get_column("audit_error").null_count() != internal_path_audit.height:
        raise RuntimeError("raw-video internal-path audit contains metadata read errors")
    if internal_path_audit.get_column("n_raw_video_internal_paths").sum() != 0:
        raise RuntimeError("raw-video candidates unexpectedly exist inside session NWBs")
    if asset_audit.height != 1:
        raise RuntimeError("raw-video asset audit must contain exactly one summary row")
    if asset_audit.get_column("n_raw_video_asset_candidates").item() != 0:
        raise RuntimeError("raw-video candidates unexpectedly exist in frozen DANDI assets")


def _reconcile_session_inventory(
    manifest: pl.DataFrame,
    companion_path: pathlib.Path,
) -> pl.DataFrame:
    companion = dg.data.scan_companion_session_metadata(companion_path).collect()
    if companion.height != 99:
        raise RuntimeError(f"expected 99 companion session rows, observed {companion.height}")
    if companion.select("subject_id", "recording_day").n_unique() != companion.height:
        raise RuntimeError("companion session rows are not unique by mouse and recording day")

    sessions = (
        manifest.filter(pl.col("asset_kind") == "session_nwb")
        .with_columns(pl.col("s3_url").alias("_nwb_path"))
        .join(
            companion,
            on=["subject_id", "recording_day"],
            how="left",
            validate="1:1",
        )
        .sort("path")
    )
    if sessions.get_column("ecephys_session_id").null_count():
        raise RuntimeError("one or more DANDI sessions did not match companion metadata")
    if sessions.get_column("ecephys_session_id").n_unique() != sessions.height:
        raise RuntimeError("companion ecephys session IDs are not one-to-one with DANDI sessions")
    expected_lfp = sessions.group_by("subject_id").agg(
        pl.when(pl.col("has_lfp"))
        .then(pl.col("n_probes"))
        .otherwise(0)
        .sum()
        .alias("n_expected_subject_lfp_assets")
    )
    observed_lfp = (
        manifest.filter(pl.col("asset_kind") == "probe_lfp_nwb")
        .group_by("subject_id")
        .agg(pl.len().alias("n_subject_lfp_assets"))
    )
    lfp_audit = (
        expected_lfp.join(observed_lfp, on="subject_id", how="left", validate="1:1")
        .with_columns(pl.col("n_subject_lfp_assets").fill_null(0))
        .with_columns(
            (pl.col("n_subject_lfp_assets") == pl.col("n_expected_subject_lfp_assets")).alias(
                "subject_lfp_count_agrees"
            ),
            (
                (pl.col("n_subject_lfp_assets") > 0)
                == (pl.col("n_expected_subject_lfp_assets") > 0)
            ).alias("subject_lfp_availability_agrees"),
        )
    )
    if not lfp_audit.get_column("subject_lfp_availability_agrees").all():
        raise RuntimeError("DANDI probe-LFP availability disagrees with companion session labels")
    return sessions.join(lfp_audit, on="subject_id", how="left", validate="m:1")


def _add_nwb_root_identifier_audit(
    session_inventory: pl.DataFrame,
    *,
    max_workers: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    root_audit = dg.audit.audit_nwb_root_identifiers(
        session_inventory,
        max_workers=max_workers,
    )
    reconciled = session_inventory.join(
        root_audit.select(
            "_nwb_path",
            "nwb_subject_id",
            "subject_id_agrees",
            "nwb_ecephys_session_id",
            "ecephys_session_id_valid",
            "ecephys_session_id_agrees",
            "session_identifier_conflict",
            "root_identifier_audit_pass",
            pl.col("audit_error").alias("root_identifier_audit_error"),
        ),
        on="_nwb_path",
        how="left",
        validate="1:1",
    )
    return reconciled, root_audit


def _artifact_record(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    n_rows: int | None,
) -> dict[str, object]:
    """Hash one stable file and return its root-relative content record."""

    resolved = path.resolve()
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while it was being hashed: {resolved}")
    return {
        "path": str(resolved.relative_to(root.resolve())),
        "size_bytes": after.st_size,
        "sha256": digest.hexdigest(),
        "n_rows": n_rows,
    }


def _local_source_records() -> dict[str, dict[str, object]]:
    """Hash every local source/configuration file loaded by the M0 audit."""

    return {
        relative_path: _artifact_record(
            REPOSITORY_ROOT / relative_path,
            root=REPOSITORY_ROOT,
            n_rows=None,
        )
        for relative_path in LOCAL_SOURCE_PATHS
    }


def _report_phase(label: str) -> None:
    """Emit a timestamped phase marker without weakening atomic publication."""

    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    print(f"[{timestamp}] M0 phase: {label}", flush=True)


def _raise_system_exit_on_sigterm(signum: int, _frame: object) -> None:
    """Route termination through the fail-closed manifest lifecycle."""

    raise SystemExit(f"received termination signal {signum}")


if __name__ == "__main__":
    main()
