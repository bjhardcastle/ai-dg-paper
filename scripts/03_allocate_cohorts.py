# /// script
# dependencies = [
#   "polars>=1.32",
# ]
# requires-python = ">=3.11"
# ///
"""Write the final D05 split using behavior, metadata, and regional presence."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from typing import Any

import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.cohorts  # noqa: E402
import dg.gates  # noqa: E402
import dg.ontology  # noqa: E402

ANALYSIS_ID = "d05_mouse_grouped_cohort_allocation"
REQUIRED_DECISIONS = ("D01", "D03", "D05", "D12")
DISCOVERY_MICE = 12
CONFIRMATION_MICE = 6
SPLIT_SEED = 1051
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/cohorts.py",
    "src/dg/gates.py",
    "src/dg/ontology.py",
    "scripts/03_allocate_cohorts.py",
)
DEPENDENCY_LOCK_PATHS = ("pyproject.toml", "uv.lock")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse repository-relative artifact inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Artifact root containing the frozen inventory and behavior QC table",
    )
    parser.add_argument(
        "--analysis-lock",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
        help="Human-approved analysis lock",
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Publish a fail-closed run marker around the complete D05 allocation."""

    arguments = parse_arguments()
    started_at = datetime.datetime.now(datetime.UTC)
    results_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = results_root / "manifests" / "cohort_allocation_run.json"
    in_progress = {
        "schema_version": dg.cohorts.COHORT_MANIFEST_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "analysis_status": "in_progress",
        "run_status": "in_progress",
        "authoritative": False,
        "script": "scripts/03_allocate_cohorts.py",
        "started_at_utc": started_at.isoformat(),
    }
    dg.artifacts.write_json(in_progress, manifest_path)
    try:
        _run_allocation(
            arguments,
            started_at=started_at,
            results_root=results_root,
        )
    except BaseException as error:
        failed = {
            **in_progress,
            "analysis_status": "failed",
            "run_status": "failed",
            "authoritative": False,
            "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(failed, manifest_path)
        raise


def _run_allocation(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    results_root: pathlib.Path,
) -> None:
    """Run and provenance the outcome-blind exhaustive allocation."""

    manifests = results_root / "manifests"
    tables = results_root / "tables"
    inventory_path = manifests / "session_inventory.parquet"
    session_qc_path = tables / "session_qc.parquet"
    m0_manifest_path = manifests / "audit_run.json"
    behavior_manifest_path = manifests / "behavior_analysis_run.json"

    lock_text = arguments.analysis_lock.read_text(encoding="utf-8")
    lock = dg.gates.read_analysis_lock(arguments.analysis_lock)
    if json.loads(lock_text) != lock:
        raise RuntimeError("analysis lock changed while it was being read")
    dg.gates.require_approved_decisions(
        lock,
        REQUIRED_DECISIONS,
        stage="D05 mouse-grouped cohort allocation",
    )

    input_records = {
        "analysis_lock": _file_record(arguments.analysis_lock),
        "m0_audit_run": _file_record(m0_manifest_path),
        "session_inventory": _file_record(inventory_path),
        "session_qc": _file_record(session_qc_path),
        "behavior_analysis_run": _file_record(behavior_manifest_path),
    }
    dependency_records = {
        pathlib.Path(path).name: _file_record(REPOSITORY_ROOT / path)
        for path in DEPENDENCY_LOCK_PATHS
    }
    input_records.update(
        {f"dependency_lockfile__{name}": record for name, record in dependency_records.items()}
    )
    inventory = pl.read_parquet(inventory_path)
    dandiset = _single_dandiset_record(inventory)
    m0_manifest, m0_dependencies = _read_validated_m0_manifest(
        m0_manifest_path,
        results_root=results_root,
        analysis_lock_record=input_records["analysis_lock"],
        session_inventory_record=input_records["session_inventory"],
        expected_dandiset_version=dandiset["dandiset_version"],
    )
    input_records.update(m0_dependencies)
    behavior_manifest, behavior_dependencies = _read_validated_behavior_manifest(
        behavior_manifest_path,
        results_root=results_root,
        m0_manifest=m0_manifest,
        m0_manifest_record=input_records["m0_audit_run"],
        analysis_lock_record=input_records["analysis_lock"],
        session_inventory_record=input_records["session_inventory"],
        session_qc_record=input_records["session_qc"],
        expected_dandiset_version=dandiset["dandiset_version"],
    )
    input_records.update(behavior_dependencies)
    session_qc = pl.read_parquet(session_qc_path)
    anatomy_coverage = _read_validated_m0_session_anatomy_coverage(
        results_root / m0_manifest["artifacts"]["unit_session_anatomy_coverage"]["path"],
        inventory,
    )
    _require_unchanged_inputs(input_records)

    source_records = _local_source_records()
    ontology_sources = dg.ontology.cache_d05_ontology_sources(
        results_root / "cache" / "d05_ontology"
    )
    for resource_key, path in sorted(ontology_sources.paths.items()):
        input_records[f"ontology_source__{resource_key}"] = _file_record(path)
    ontology = dg.ontology.load_d05_major_division_ontology(ontology_sources)
    regional_mapping = dg.ontology.map_session_anatomy_to_major_divisions(
        anatomy_coverage,
        ontology,
    )
    mouse_regional_coverage = dg.cohorts.summarize_mouse_major_division_coverage(
        regional_mapping.session_presence,
        session_qc,
        ontology.divisions,
    )
    _require_unchanged_inputs(input_records)

    allocation = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        session_qc,
        mouse_major_division_coverage=mouse_regional_coverage,
        discovery_mice=DISCOVERY_MICE,
        confirmation_mice=CONFIRMATION_MICE,
        seed=SPLIT_SEED,
    )
    if _local_source_records() != source_records:
        raise RuntimeError("cohort-allocation source changed during the exhaustive search")
    _require_unchanged_inputs(input_records)

    output_paths = {
        "mouse_assignments_csv": tables / "mouse_cohort_assignments.csv",
        "mouse_assignments_parquet": tables / "mouse_cohort_assignments.parquet",
        "session_assignments_csv": tables / "session_cohort_assignments.csv",
        "session_assignments_parquet": tables / "session_cohort_assignments.parquet",
        "metadata_balance": tables / "cohort_metadata_balance.csv",
        "balance_precision_summary": tables / "cohort_allocation_summary.csv",
        "ontology_source_provenance": manifests / "d05_ontology_source_provenance.csv",
        "major_division_definitions": tables / "d05_major_division_definitions.csv",
        "ontology_validation": tables / "d05_ontology_validation.csv",
        "raw_acronym_mapping_diagnostics": (tables / "d05_raw_acronym_mapping_diagnostics.csv"),
        "expected_candidate_mapping_audit": (tables / "d05_expected_candidate_mapping_audit.csv"),
        "session_major_division_presence": (tables / "d05_session_major_division_presence.parquet"),
        "mouse_major_division_coverage_csv": (tables / "d05_mouse_major_division_coverage.csv"),
        "mouse_major_division_coverage_parquet": (
            tables / "d05_mouse_major_division_coverage.parquet"
        ),
        "software_environment": manifests / "cohort_allocation_environment.txt",
    }
    output_frames = {
        "mouse_assignments_csv": allocation.mice,
        "mouse_assignments_parquet": allocation.mice,
        "session_assignments_csv": allocation.sessions,
        "session_assignments_parquet": allocation.sessions,
        "metadata_balance": allocation.balance,
        "balance_precision_summary": allocation.summary,
        "ontology_source_provenance": ontology_sources.provenance,
        "major_division_definitions": ontology.divisions,
        "ontology_validation": ontology.validation,
        "raw_acronym_mapping_diagnostics": regional_mapping.acronym_diagnostics,
        "expected_candidate_mapping_audit": regional_mapping.candidate_diagnostics,
        "session_major_division_presence": regional_mapping.session_presence,
        "mouse_major_division_coverage_csv": mouse_regional_coverage,
        "mouse_major_division_coverage_parquet": mouse_regional_coverage,
    }
    for name, frame in output_frames.items():
        dg.artifacts.write_frame(frame, output_paths[name])
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        output_paths["software_environment"],
    )
    lock_snapshot_path = manifests / "cohort_allocation_lock.yaml"
    dg.artifacts.write_text(lock_text, lock_snapshot_path)
    if _local_source_records() != source_records:
        raise RuntimeError("cohort-allocation source changed while outputs were written")
    _require_unchanged_inputs(input_records)

    output_records = {name: _file_record(path) for name, path in output_paths.items()}
    output_records["analysis_lock_snapshot"] = _file_record(lock_snapshot_path)
    summary = allocation.summary.row(0, named=True)
    completed_at = datetime.datetime.now(datetime.UTC)
    assignment_identity = {
        f"{cohort}_mouse_ids": (
            allocation.mice.filter(pl.col("cohort_assignment") == cohort)
            .get_column("subject_id")
            .sort()
            .to_list()
        )
        for cohort in ("discovery", "confirmation", "excluded")
    }
    manifest = {
        "schema_version": dg.cohorts.COHORT_MANIFEST_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "allocation_status": "final",
        "assignment_table_status": summary["allocation_status"],
        "ready_for_neural_discovery": summary["ready_for_neural_discovery"],
        "confirmation_holdout_accessed": False,
        "stage": "approved_behavior_metadata_regional_presence_only_final_allocation",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "script": "scripts/03_allocate_cohorts.py",
        "code_version": _code_version(),
        "required_approved_decisions": list(REQUIRED_DECISIONS),
        "decision_statuses": {
            decision_id: lock["decisions"][decision_id]["status"]
            for decision_id in REQUIRED_DECISIONS
        },
        "split_seed": SPLIT_SEED,
        "requested_discovery_mice": DISCOVERY_MICE,
        "requested_confirmation_mice": CONFIRMATION_MICE,
        "assignment_identity": assignment_identity,
        "eligibility_rule": (
            "mouse eligible iff at least one inventory session is D03 "
            "threshold-selected (session_qc.is_good_session)"
        ),
        "session_retention_rule": (
            "retain every inventory session and inherit the mouse split; activate "
            "discovery_analysis_included only for D03-selected discovery sessions; "
            "keep confirmation_analysis_included false until the separate gate"
        ),
        "assignment_artifact_policy": (
            "Parquet assignment tables are authoritative for identifier dtypes; "
            "CSV companions are human-readable and must be read with subject_id as String"
        ),
        "discovery_source_activation": {
            "selector": "dg.cohorts.select_authenticated_discovery_session_sources",
            "status": "ready",
            "requirements": (
                "authenticated Parquet and completed authoritative D05 manifest; current "
                "authoritative M0 inventory and behavior QC; exact parent, output, source, "
                "environment, and dependency hashes; deterministic split recomputation; "
                "and no activated confirmation rows"
            ),
        },
        "allocation_inputs": (
            "threshold-selection boolean plus sex, genotype, project_code, "
            "recording-day summaries, inventory/selected session counts, and "
            "per-mouse official-major-division selected-session presence fractions"
        ),
        "neural_activity_used_for_allocation": False,
        "confirmation_policy": (
            "confirmation cohort remains sealed; all confirmation-use flags are false "
            "until the separate one-time confirmation gate"
        ),
        "coarse_regional_coverage_contract": {
            "author_tree_commit": dg.ontology.AUTHOR_REPOSITORY_COMMIT,
            "structure_graph_id": dg.ontology.STRUCTURE_GRAPH_ID,
            "official_structure_set_id": dg.ontology.MAJOR_DIVISION_STRUCTURE_SET_ID,
            "n_major_divisions": ontology.divisions.height,
            "mouse_feature": (
                "fraction of D03-selected sessions with at least one raw located unit "
                "in the major division"
            ),
            "unit_population": "all units regardless of isolation quality",
            "presence_only": True,
            "unit_count_magnitude_used": False,
            "neural_activity_rate_or_outcome_used": False,
            "minimum_represented_eligible_mice": (dg.cohorts.MINIMUM_REGION_REPRESENTED_MICE),
            "variance_requirement": "nonzero exact eligible-mouse sample variance",
            "included_divisions": (
                mouse_regional_coverage.filter(pl.col("include_in_balance"))
                .get_column("major_division_acronym")
                .unique()
                .sort()
                .to_list()
            ),
            "family_weighting": ("all included divisions form one equal-weight objective family"),
            "arithmetic": "exact rational numerator/denominator fractions",
            "expected_candidate_audit": regional_mapping.candidate_diagnostics.to_dicts(),
        },
        "d09_scope_separation": (
            "D09's proposed 293-node reporting hierarchy is separate from D05 and "
            "was not used for cohort allocation"
        ),
        "search": {
            "method": summary["search_method"],
            "candidates_evaluated": summary["n_candidates_evaluated"],
            "expected_candidates": math.comb(DISCOVERY_MICE + CONFIRMATION_MICE, CONFIRMATION_MICE),
            "objective": (
                "equal-family mean of within-family mean squared standardized "
                "discovery-minus-confirmation differences; standardizer is sample "
                "SD across all eligible mice"
            ),
            "arithmetic": summary["objective_arithmetic"],
            "objective_exact": summary["mean_family_mean_squared_smd_exact"],
            "objective_float": summary["mean_family_mean_squared_smd"],
            "root_mean_squared_smd": summary["root_mean_family_mean_squared_smd"],
            "maximum_absolute_smd": summary["maximum_absolute_smd"],
            "primary_objective_ties": summary["n_primary_objective_ties"],
            "secondary_objective_ties": summary["n_secondary_objective_ties"],
            "tie_break_rule": (
                "minimum maximum squared SMD, then minimum SHA-256 rank of "
                "seed plus sorted confirmation IDs, then lexical IDs"
            ),
            "selected_tie_break_sha256": summary["selected_tie_break_sha256"],
        },
        "cohort_counts": {
            "inventory_mice": summary["n_inventory_mice"],
            "eligible_mice": summary["n_eligible_mice"],
            "excluded_mice": summary["n_excluded_mice"],
            "discovery_mice": summary["n_discovery_mice"],
            "confirmation_mice": summary["n_confirmation_mice"],
            "inventory_sessions": summary["n_inventory_sessions"],
            "threshold_selected_sessions": summary["n_threshold_selected_sessions"],
            "discovery_threshold_selected_sessions": summary[
                "n_discovery_threshold_selected_sessions"
            ],
            "confirmation_threshold_selected_sessions": summary[
                "n_confirmation_threshold_selected_sessions"
            ],
        },
        "dandiset": dandiset,
        "behavior_analysis_reference": {
            "analysis_id": behavior_manifest.get("analysis_id"),
            "analysis_status": behavior_manifest.get("analysis_status"),
            "run_status": behavior_manifest.get("run_status"),
            "authoritative": behavior_manifest.get("authoritative"),
            "trial_input_mode": behavior_manifest.get("trial_input_mode"),
            "output_matches_trusted_checkpoint": behavior_manifest.get(
                "behavior_trials_checkpoint", {}
            ).get("output_matches_trusted_checkpoint"),
            "pending_behavior_decisions": behavior_manifest.get("pending_behavior_decisions"),
        },
        "m0_reference": {
            "manifest_schema_version": m0_manifest.get("manifest_schema_version"),
            "run_status": m0_manifest.get("run_status"),
            "authoritative": m0_manifest.get("authoritative"),
            "milestone_0_status": m0_manifest.get("milestone_0_status"),
            "asset_inventory_status": m0_manifest.get("asset_inventory_status"),
            "nwb_root_metadata_audit_status": m0_manifest.get("nwb_root_metadata_audit_status"),
            "unit_inventory_status": m0_manifest.get("unit_inventory_status"),
            "unit_metadata_inventory_status": m0_manifest.get("unit_metadata_inventory_status"),
            "neural_outcomes_accessed": m0_manifest.get("neural_outcomes_accessed"),
            "spike_arrays_loaded": m0_manifest.get("spike_arrays_loaded"),
        },
        "parent_hashes": {
            "m0_audit_run_sha256": input_records["m0_audit_run"]["sha256"],
            "behavior_analysis_run_sha256": input_records["behavior_analysis_run"]["sha256"],
            "session_inventory_sha256": input_records["session_inventory"]["sha256"],
            "session_qc_sha256": input_records["session_qc"]["sha256"],
        },
        "assignment_hashes": {
            "mouse_assignments_parquet_sha256": output_records["mouse_assignments_parquet"][
                "sha256"
            ],
            "session_assignments_parquet_sha256": output_records["session_assignments_parquet"][
                "sha256"
            ],
        },
        "inputs": input_records,
        "dependency_lockfiles": dependency_records,
        "local_sources": source_records,
        "outputs": output_records,
    }
    manifest_path = manifests / "cohort_allocation_run.json"
    dg.artifacts.write_json(manifest, manifest_path)
    print(
        "Finalized D05 allocation of "
        f"{summary['n_discovery_mice']} discovery and "
        f"{summary['n_confirmation_mice']} held-out confirmation mice from "
        f"{summary['n_eligible_mice']} eligible mice; "
        f"evaluated {summary['n_candidates_evaluated']} candidates."
    )
    print(f"Wrote {manifest_path}")


def _read_validated_m0_manifest(
    path: pathlib.Path,
    *,
    results_root: pathlib.Path,
    analysis_lock_record: dict[str, Any],
    session_inventory_record: dict[str, Any],
    expected_dandiset_version: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = _read_json_object(path, label="M0 audit manifest")
    if record.get("manifest_schema_version") != 3:
        raise RuntimeError("M0 audit manifest schema_version is not 3")
    if record.get("run_status") != "complete" or record.get("authoritative") is not True:
        raise RuntimeError("M0 audit is not complete and authoritative")
    if record.get("milestone_0_status") not in {"partial", "complete"}:
        raise RuntimeError("M0 milestone status is neither explicit partial nor complete")
    if record.get("dandiset_version") != expected_dandiset_version:
        raise RuntimeError("M0 dandiset version differs from the session inventory")
    if record.get("analysis_lock_status") != "approved":
        raise RuntimeError("M0 audit was not generated under an approved analysis lock")
    if record.get("asset_inventory_status") != "pass":
        raise RuntimeError("M0 asset inventory did not pass")
    if record.get("nwb_root_metadata_audit_status") != "pass":
        raise RuntimeError("M0 NWB root-metadata audit did not pass")
    if record.get("unit_inventory_status") != "pass":
        raise RuntimeError("M0 unit inventory did not pass")
    if record.get("unit_metadata_inventory_status") != "pass_scalar_metadata_no_spike_arrays":
        raise RuntimeError("M0 scalar unit-metadata inventory did not pass")
    if record.get("neural_outcomes_accessed") is not False:
        raise RuntimeError("M0 audit does not affirm that neural outcomes were not accessed")
    if record.get("spike_arrays_loaded") is not False:
        raise RuntimeError("M0 audit does not affirm that spike arrays were not loaded")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("M0 audit lacks an artifacts mapping")
    required = {
        "analysis_lock_snapshot",
        "session_inventory",
        "task_parameters_audit",
        "unit_inventory",
        "unit_session_anatomy_coverage",
    }
    missing = sorted(required.difference(artifacts))
    if missing:
        raise RuntimeError(f"M0 audit lacks required artifacts: {missing}")
    dependencies = {
        f"m0_artifact__{name}": _validate_recorded_artifact(
            artifact,
            base=results_root,
            label=f"M0 artifact {name}",
        )
        for name, artifact in sorted(artifacts.items())
    }
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise RuntimeError("M0 audit lacks its input provenance chain")
    dependencies.update(
        {
            f"m0_input__{name}": _validate_recorded_artifact(
                artifact,
                base=results_root,
                label=f"M0 input {name}",
            )
            for name, artifact in sorted(inputs.items())
        }
    )
    if (
        dependencies["m0_artifact__session_inventory"]["sha256"]
        != session_inventory_record["sha256"]
    ):
        raise RuntimeError("current session inventory differs from authoritative M0")
    if (
        dependencies["m0_artifact__analysis_lock_snapshot"]["sha256"]
        != analysis_lock_record["sha256"]
    ):
        raise RuntimeError("current analysis lock differs from authoritative M0 snapshot")
    local_sources = record.get("local_sources")
    if not isinstance(local_sources, dict) or not local_sources:
        raise RuntimeError("M0 audit lacks its local-source provenance chain")
    validated_sources = {
        f"m0_local_source__{name}": _validate_recorded_artifact(
            source,
            base=REPOSITORY_ROOT,
            label=f"M0 local source {name}",
        )
        for name, source in sorted(local_sources.items())
    }
    dependencies.update(validated_sources)
    recorded_lock_source = local_sources.get("config/analysis_lock.yaml", {})
    if recorded_lock_source.get("sha256") != analysis_lock_record["sha256"]:
        raise RuntimeError("authoritative M0 local-source lock hash is missing or stale")
    return record, dependencies


def _read_validated_behavior_manifest(
    path: pathlib.Path,
    *,
    results_root: pathlib.Path,
    m0_manifest: dict[str, Any],
    m0_manifest_record: dict[str, Any],
    analysis_lock_record: dict[str, Any],
    session_inventory_record: dict[str, Any],
    session_qc_record: dict[str, Any],
    expected_dandiset_version: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = _read_json_object(path, label="behavior analysis manifest")
    if (
        record.get("analysis_status") != "pass"
        or record.get("run_status") != "complete"
        or record.get("authoritative") is not True
    ):
        raise RuntimeError("behavior analysis is not a completed authoritative pass")
    if record.get("pending_behavior_decisions") != []:
        raise RuntimeError("behavior analysis does not report exactly zero pending decisions")
    if record.get("dandiset_version") != expected_dandiset_version:
        raise RuntimeError("behavior-analysis Dandiset version differs from the inventory")

    trial_input_mode = record.get("trial_input_mode")
    expected_checkpoint_status = (
        {
            "remote_nwb": "created_from_remote_nwb",
            "validated_existing_behavior_trials": "validated_existing_not_refreshed",
        }.get(trial_input_mode)
        if isinstance(trial_input_mode, str)
        else None
    )
    if expected_checkpoint_status is None:
        raise RuntimeError("behavior analysis has a non-canonical trial input mode")

    dependencies: dict[str, dict[str, Any]] = {}
    generator = _validate_recorded_artifact(
        record.get("generator"),
        base=REPOSITORY_ROOT,
        label="behavior generator",
    )
    dependencies["behavior_generator"] = generator
    local_sources = record.get("local_sources")
    if not isinstance(local_sources, list) or not local_sources:
        raise RuntimeError("behavior analysis lacks its local-source provenance chain")
    validated_behavior_sources = [
        _validate_recorded_artifact(
            source,
            base=REPOSITORY_ROOT,
            label=f"behavior local source {index}",
        )
        for index, source in enumerate(local_sources)
    ]
    dependencies.update(
        {
            f"behavior_local_source__{index}": source
            for index, source in enumerate(validated_behavior_sources)
        }
    )

    embedded_m0 = record.get("input_audit_run")
    if embedded_m0 != m0_manifest:
        raise RuntimeError("behavior analysis does not embed the current authoritative M0 manifest")
    actual_m0_manifest = _validate_recorded_artifact(
        record.get("input_audit_run_manifest"),
        base=REPOSITORY_ROOT,
        label="behavior input M0 manifest",
    )
    if actual_m0_manifest != m0_manifest_record:
        raise RuntimeError("behavior analysis M0-manifest record differs from current M0")
    dependencies["behavior_input_m0_manifest"] = actual_m0_manifest

    behavior_outputs = record.get("outputs")
    if not isinstance(behavior_outputs, dict):
        raise RuntimeError("behavior analysis lacks its output provenance chain")
    recorded_qc = behavior_outputs.get("session_qc", {})
    actual_qc = _validate_recorded_artifact(
        recorded_qc,
        base=REPOSITORY_ROOT,
        label="behavior session_qc",
    )
    if actual_qc["sha256"] != session_qc_record["sha256"]:
        raise RuntimeError("session_qc hash differs from the behavior manifest")
    if record.get("analysis_lock_sha256") != analysis_lock_record["sha256"]:
        raise RuntimeError("behavior analysis lock hash differs from the current approved lock")
    if record.get("session_inventory_sha256") != session_inventory_record["sha256"]:
        raise RuntimeError("behavior analysis inventory hash differs from current authoritative M0")
    inventory_path_value = record.get("session_inventory_path")
    if not isinstance(inventory_path_value, str) or not inventory_path_value:
        raise RuntimeError("behavior analysis lacks its session-inventory path")
    recorded_inventory_path = _resolve_path(
        inventory_path_value,
        base=REPOSITORY_ROOT,
    ).resolve()
    current_inventory_path = _resolve_path(
        session_inventory_record["path"],
        base=REPOSITORY_ROOT,
    ).resolve()
    if recorded_inventory_path != current_inventory_path:
        raise RuntimeError("behavior analysis inventory path differs from current authoritative M0")

    m0_task_audit = m0_manifest.get("artifacts", {}).get("task_parameters_audit")
    m0_task_audit_record = _validate_recorded_artifact(
        m0_task_audit,
        base=results_root,
        label="M0 task-parameters audit",
    )
    task_audit_path_value = record.get("task_parameters_audit_path")
    if not isinstance(task_audit_path_value, str) or not task_audit_path_value:
        raise RuntimeError("behavior analysis lacks its task-parameters-audit path")
    behavior_task_audit_record = _file_record(
        _resolve_path(task_audit_path_value, base=REPOSITORY_ROOT)
    )
    if record.get("task_parameters_audit_sha256") != behavior_task_audit_record["sha256"]:
        raise RuntimeError("behavior task-parameters-audit digest differs from the current file")
    if behavior_task_audit_record != m0_task_audit_record:
        raise RuntimeError("behavior task-parameters audit differs from authoritative M0")
    dependencies["behavior_task_parameters_audit"] = behavior_task_audit_record

    lock_snapshot_value = record.get("analysis_lock_snapshot")
    if not isinstance(lock_snapshot_value, str):
        raise RuntimeError("behavior analysis manifest lacks an analysis-lock snapshot path")
    behavior_lock_snapshot = _file_record(_resolve_path(lock_snapshot_value, base=REPOSITORY_ROOT))
    if behavior_lock_snapshot["sha256"] != analysis_lock_record["sha256"]:
        raise RuntimeError("behavior analysis lock snapshot differs from the current lock")
    analysis_lock_path_value = record.get("analysis_lock_path")
    if not isinstance(analysis_lock_path_value, str) or not analysis_lock_path_value:
        raise RuntimeError("behavior analysis lacks its analysis-lock path")
    if (
        _resolve_path(analysis_lock_path_value, base=REPOSITORY_ROOT).resolve()
        != _resolve_path(analysis_lock_record["path"], base=REPOSITORY_ROOT).resolve()
    ):
        raise RuntimeError("behavior analysis lock path differs from the current approved lock")

    checkpoint_reference = record.get("behavior_trials_checkpoint")
    if not isinstance(checkpoint_reference, dict):
        raise RuntimeError("behavior analysis lacks a behavior-trials checkpoint reference")
    checkpoint_record = _validate_recorded_artifact(
        checkpoint_reference,
        base=REPOSITORY_ROOT,
        label="behavior-trials checkpoint manifest",
    )
    checkpoint = _read_json_object(
        _resolve_path(checkpoint_reference["path"], base=REPOSITORY_ROOT),
        label="behavior-trials checkpoint",
    )
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("behavior-trials checkpoint schema_version is not 1")
    if checkpoint.get("checkpoint_kind") != "direct_remote_nwb_behavior_trials":
        raise RuntimeError("behavior-trials checkpoint is not a direct remote-NWB checkpoint")
    if checkpoint.get("trial_input_mode") != "remote_nwb":
        raise RuntimeError("behavior-trials checkpoint input mode is not remote_nwb")
    if checkpoint.get("dandiset_version") != expected_dandiset_version:
        raise RuntimeError("behavior checkpoint Dandiset version differs from the inventory")
    if (
        not isinstance(record.get("dandiset_id"), str)
        or not record["dandiset_id"]
        or checkpoint.get("dandiset_id") != record["dandiset_id"]
    ):
        raise RuntimeError("behavior checkpoint Dandiset ID differs from its parent run")
    if checkpoint_reference.get("status") != expected_checkpoint_status:
        raise RuntimeError("behavior checkpoint status is incompatible with the trial input mode")
    if checkpoint_reference.get("output_matches_trusted_checkpoint") is not True:
        raise RuntimeError("behavior output does not exactly match its trusted checkpoint")

    checkpoint_generator = _validate_recorded_artifact(
        checkpoint.get("generator"),
        base=REPOSITORY_ROOT,
        label="behavior checkpoint generator",
    )
    checkpoint_sources = checkpoint.get("local_sources")
    if not isinstance(checkpoint_sources, list) or not checkpoint_sources:
        raise RuntimeError("behavior checkpoint lacks its local-source provenance chain")
    validated_checkpoint_sources = [
        _validate_recorded_artifact(
            source,
            base=REPOSITORY_ROOT,
            label=f"behavior checkpoint local source {index}",
        )
        for index, source in enumerate(checkpoint_sources)
    ]
    if (
        checkpoint_generator != generator
        or validated_checkpoint_sources != validated_behavior_sources
    ):
        raise RuntimeError("behavior run and trusted checkpoint source hashes differ")

    checkpoint_inventory = checkpoint.get("session_inventory")
    if not isinstance(checkpoint_inventory, dict):
        raise RuntimeError("behavior-trials checkpoint lacks session inventory provenance")
    actual_checkpoint_inventory = _validate_recorded_artifact(
        checkpoint_inventory,
        base=REPOSITORY_ROOT,
        label="checkpoint session inventory",
    )
    if actual_checkpoint_inventory["sha256"] != session_inventory_record["sha256"]:
        raise RuntimeError("behavior checkpoint inventory differs from current authoritative M0")
    if actual_checkpoint_inventory != session_inventory_record:
        raise RuntimeError("behavior checkpoint inventory record differs from current M0")

    trusted_trials = checkpoint.get("behavior_trials")
    if not isinstance(trusted_trials, dict) or not trusted_trials.get("sha256"):
        raise RuntimeError("behavior-trials checkpoint lacks a trusted trial digest")
    referenced_trusted_sha = checkpoint_reference.get("trusted_behavior_trials_sha256")
    if referenced_trusted_sha != trusted_trials["sha256"]:
        raise RuntimeError("behavior manifest and checkpoint disagree on the trusted trial digest")
    reused_sha = record.get("reused_behavior_trials_sha256")
    if trial_input_mode == "remote_nwb" and reused_sha is not None:
        raise RuntimeError("direct remote behavior run unexpectedly reports a reused trial digest")
    if (
        trial_input_mode == "validated_existing_behavior_trials"
        and reused_sha != trusted_trials["sha256"]
    ):
        raise RuntimeError("reused behavior-trial digest differs from its trusted checkpoint")

    current_trials = behavior_outputs.get("behavior_trials")
    if not isinstance(current_trials, dict):
        raise RuntimeError("behavior analysis lacks current behavior-trials provenance")
    current_trials_record = _validate_recorded_artifact(
        current_trials,
        base=REPOSITORY_ROOT,
        label="current behavior trials",
    )
    trusted_trials_record = _validate_recorded_artifact(
        trusted_trials,
        base=REPOSITORY_ROOT,
        label="trusted checkpoint behavior trials",
    )
    if current_trials_record != trusted_trials_record:
        raise RuntimeError("current behavior trials do not exactly match the trusted checkpoint")

    companion = checkpoint.get("companion_trials")
    if not isinstance(companion, dict):
        raise RuntimeError("behavior checkpoint lacks companion-trial provenance")
    companion_record = _validate_recorded_artifact(
        companion,
        base=REPOSITORY_ROOT,
        label="checkpoint companion trials",
    )
    companion_provenance = record.get("companion_trials_provenance")
    if not isinstance(companion_provenance, dict):
        raise RuntimeError("behavior analysis lacks companion-trial provenance")
    companion_path_value = companion_provenance.get("cached_path")
    if not isinstance(companion_path_value, str) or not companion_path_value:
        raise RuntimeError("behavior companion-trial provenance lacks cached_path")
    current_companion_record = _file_record(
        _resolve_path(companion_path_value, base=REPOSITORY_ROOT)
    )
    if (
        companion_provenance.get("sha256") != current_companion_record["sha256"]
        or companion_provenance.get("content_size_bytes") != current_companion_record["size_bytes"]
        or current_companion_record != companion_record
    ):
        raise RuntimeError("behavior companion-trial provenance differs from the checkpoint")
    for field in ("source_url", "repository_commit"):
        if companion_provenance.get(field) != companion.get(field):
            raise RuntimeError(f"behavior companion-trial {field} differs from the checkpoint")

    dependencies.update(
        {
            "behavior_analysis_lock_snapshot": behavior_lock_snapshot,
            "behavior_trials_checkpoint": checkpoint_record,
            "behavior_checkpoint_generator": checkpoint_generator,
            "behavior_checkpoint_session_inventory": actual_checkpoint_inventory,
            "behavior_checkpoint_companion_trials": companion_record,
            "behavior_checkpoint_trials": trusted_trials_record,
            "behavior_current_trials": current_trials_record,
            "behavior_current_session_qc": actual_qc,
        }
    )
    dependencies.update(
        {
            f"behavior_checkpoint_local_source__{index}": source
            for index, source in enumerate(validated_checkpoint_sources)
        }
    )
    return record, dependencies


def _read_validated_m0_session_anatomy_coverage(
    path: pathlib.Path,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    coverage = pl.read_csv(
        path,
        schema_overrides={"_nwb_path": pl.String, "subject_id": pl.String},
    )
    required = (
        "_nwb_path",
        "subject_id",
        "anatomy_scope",
        "analysis_scope",
    )
    missing = sorted(set(required).difference(coverage.columns))
    if missing:
        raise RuntimeError(f"M0 unit-session anatomy coverage lacks columns: {missing}")
    if coverage.is_empty():
        raise RuntimeError("M0 unit-session anatomy coverage is empty")
    if coverage.get_column("_nwb_path").dtype != pl.String:
        raise RuntimeError("M0 unit-session anatomy coverage _nwb_path must be String")
    if coverage.get_column("subject_id").dtype != pl.String:
        raise RuntimeError("M0 unit-session anatomy coverage subject_id must be String")
    if coverage.get_column("subject_id").null_count():
        raise RuntimeError("M0 unit-session anatomy coverage subject_id contains nulls")
    expected_sessions = set(inventory.get_column("_nwb_path").to_list())
    observed_sessions = set(coverage.get_column("_nwb_path").to_list())
    if observed_sessions != expected_sessions:
        missing_sessions = sorted(expected_sessions - observed_sessions)
        unexpected_sessions = sorted(observed_sessions - expected_sessions)
        raise RuntimeError(
            "M0 unit-session anatomy coverage differs from session inventory: "
            f"missing={missing_sessions[:5]}, unexpected={unexpected_sessions[:5]}"
        )
    coverage_identity = coverage.select("_nwb_path", "subject_id").unique()
    inventory_identity = inventory.select("_nwb_path", "subject_id")
    if coverage_identity.height != inventory_identity.height or set(
        coverage_identity.iter_rows()
    ) != set(inventory_identity.iter_rows()):
        raise RuntimeError("M0 unit-session anatomy coverage subject/session identities differ")
    anatomy_scopes = coverage.get_column("anatomy_scope").unique().to_list()
    analysis_scopes = coverage.get_column("analysis_scope").unique().to_list()
    if anatomy_scopes != ["raw_allen_acronym_no_parent_mapping"]:
        raise RuntimeError(f"unexpected M0 anatomy scope: {anatomy_scopes}")
    if analysis_scopes != ["coverage_only_no_neural_activity"]:
        raise RuntimeError(f"unexpected M0 analysis scope: {analysis_scopes}")
    return coverage


def _read_json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _validate_recorded_artifact(
    record: dict[str, Any],
    *,
    base: pathlib.Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} provenance must be a mapping")
    recorded_path = record.get("path")
    recorded_sha256 = record.get("sha256")
    recorded_size = record.get("size_bytes")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise RuntimeError(f"{label} provenance lacks a path")
    if (
        not isinstance(recorded_sha256, str)
        or len(recorded_sha256) != 64
        or any(character not in "0123456789abcdef" for character in recorded_sha256)
    ):
        raise RuntimeError(f"{label} provenance lacks path or SHA-256")
    if isinstance(recorded_size, bool) or not isinstance(recorded_size, int) or recorded_size < 0:
        raise RuntimeError(f"{label} provenance lacks integer size_bytes")
    actual = _file_record(_resolve_path(recorded_path, base=base))
    if actual["sha256"] != recorded_sha256 or actual["size_bytes"] != recorded_size:
        raise RuntimeError(f"{label} file differs from its recorded hash or size")
    return actual


def _resolve_path(value: str, *, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else base / path


def _single_dandiset_record(inventory: pl.DataFrame) -> dict[str, Any]:
    record = {}
    for column in ("dandiset_id", "dandiset_version", "dandiset_doi"):
        if column not in inventory.columns:
            raise RuntimeError(f"session inventory lacks {column}")
        values = inventory.get_column(column).drop_nulls().unique().to_list()
        if len(values) != 1:
            raise RuntimeError(f"session inventory does not have one {column}: {values}")
        record[column] = values[0]
    return record


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"file changed while it was being hashed: {resolved}")
    try:
        display_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": digest,
        "size_bytes": after.st_size,
    }


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_source_records() -> list[dict[str, Any]]:
    return [_file_record(REPOSITORY_ROOT / path) for path in LOCAL_SOURCE_PATHS]


def _require_unchanged_inputs(records: dict[str, dict[str, Any]]) -> None:
    for name, record in records.items():
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if _file_record(path) != record:
            raise RuntimeError(f"{name} changed during cohort allocation")


def _code_version() -> str:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{revision}+dirty" if dirty else revision


if __name__ == "__main__":
    main()
