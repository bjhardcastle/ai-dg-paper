# /// script
# dependencies = [
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Analyze real behavioral transitions and approved D03 session QC.

This script is a behavior-only quality-control and transition-resolution stage.
It reads the immutable outputs of ``01_analyze_behavior.py`` and does not read
NWB files, neural activity, or any planned Figure 2 neural result.
"""

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
import dg.behavior_transitions  # noqa: E402
import dg.data  # noqa: E402
import dg.gates  # noqa: E402
import dg.quality  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "behavior_transition_qc"
ANALYSIS_TIER = "exploratory"
REQUIRED_DECISIONS = ("D01", "D02", "D03")
DEFAULT_SEED = 1051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/behavior_transitions.py",
    "src/dg/data.py",
    "src/dg/gates.py",
    "src/dg/quality.py",
    "src/dg/statistics.py",
    "scripts/04_analyze_behavior_transitions.py",
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments with repository-relative safe defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
        help="Existing result root produced by scripts 00 and 01",
    )
    parser.add_argument(
        "--analysis-lock",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
        help="Approved analysis-lock path",
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
    return parser.parse_args(arguments)


def main() -> None:
    """Publish a fail-closed marker around behavior transition QC."""

    arguments = parse_arguments()
    _validate_arguments(arguments)
    started_at = datetime.datetime.now(datetime.UTC)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = result_root / "manifests" / "behavior_transition_analysis_run.json"
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "authoritative": False,
        "script": "scripts/04_analyze_behavior_transitions.py",
        "started_at_utc": started_at.isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
    }
    dg.artifacts.write_json(in_progress, manifest_path)
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
            dg.artifacts.write_json(failed, manifest_path)
        raise


def _run_analysis(
    arguments: argparse.Namespace,
    *,
    started_at: datetime.datetime,
    result_root: pathlib.Path,
) -> None:
    """Run behavior transition QC from the existing real-data artifacts."""

    tables_directory = result_root / "tables"
    manifests_directory = result_root / "manifests"
    behavior_trials_path = tables_directory / "behavior_trials.parquet"
    session_qc_path = tables_directory / "session_qc.parquet"
    behavior_run_path = manifests_directory / "behavior_analysis_run.json"
    behavior_lock_snapshot_path = manifests_directory / "behavior_analysis_lock.yaml"
    session_inventory_path = manifests_directory / "session_inventory.parquet"
    audit_run_path = manifests_directory / "audit_run.json"
    source_records_at_start = [
        _file_record(REPOSITORY_ROOT / relative_path) for relative_path in LOCAL_SOURCE_PATHS
    ]

    lock_text = arguments.analysis_lock.read_text(encoding="utf-8")
    lock = dg.gates.read_analysis_lock(arguments.analysis_lock)
    if json.loads(lock_text) != lock:
        raise RuntimeError("analysis lock changed while it was being read")
    dg.gates.require_approved_decisions(
        lock,
        REQUIRED_DECISIONS,
        stage="behavior transition QC",
    )
    _validate_locked_dandiset(lock)

    behavior_run = _read_behavior_run(behavior_run_path)
    input_records = _validate_behavior_inputs(
        behavior_run,
        behavior_trials_path=behavior_trials_path,
        session_qc_path=session_qc_path,
        behavior_run_path=behavior_run_path,
        behavior_lock_snapshot_path=behavior_lock_snapshot_path,
        session_inventory_path=session_inventory_path,
        audit_run_path=audit_run_path,
    )
    if input_records["behavior_analysis_lock_snapshot"]["sha256"] != _sha256_file(
        arguments.analysis_lock
    ):
        raise RuntimeError("upstream behavior artifacts were not generated under the current lock")
    behavior_trials_sha256 = input_records["behavior_trials"]["sha256"]
    session_qc_sha256 = input_records["session_qc"]["sha256"]
    trials = pl.read_parquet(behavior_trials_path)
    session_qc = pl.read_parquet(session_qc_path)
    if _sha256_file(behavior_trials_path) != behavior_trials_sha256:
        raise RuntimeError("behavior_trials.parquet changed while it was being read")
    if _sha256_file(session_qc_path) != session_qc_sha256:
        raise RuntimeError("session_qc.parquet changed while it was being read")
    _validate_real_inputs(trials, session_qc)

    config = dg.behavior_transitions.DEFAULT_TRANSITION_CONFIG
    tables = dg.behavior_transitions.build_transition_tables(
        trials,
        session_qc,
        config=config,
    )
    time_summary = dg.behavior_transitions.summarize_mouse_trajectory(
        tables.mouse_time_bins,
        bin_kind="time",
        seed=arguments.seed,
        n_resamples=arguments.bootstrap_resamples,
    )
    trial_summary = dg.behavior_transitions.summarize_mouse_trajectory(
        tables.mouse_trial_bins,
        bin_kind="trial",
        seed=arguments.seed + 10_000,
        n_resamples=arguments.bootstrap_resamples,
    )
    code_version = _code_version(REPOSITORY_ROOT)
    statistics = dg.behavior_transitions.build_transition_statistics(
        tables.mouse_effects,
        dandiset_version=dg.data.DANDISET_VERSION,
        code_version=code_version,
        seed=arguments.seed + 20_000,
        n_bootstrap_resamples=arguments.bootstrap_resamples,
        n_sign_flip_resamples=arguments.sign_flip_resamples,
        minimum_go_trials_per_effect_window=(config.minimum_go_trials_per_effect_window),
        minimum_catch_trials_per_effect_window=(config.minimum_catch_trials_per_effect_window),
    )
    _validate_results(
        tables,
        time_summary,
        trial_summary,
        statistics,
        session_qc=session_qc,
        config=config,
    )
    if [
        _file_record(REPOSITORY_ROOT / relative_path) for relative_path in LOCAL_SOURCE_PATHS
    ] != source_records_at_start:
        raise RuntimeError("behavior-transition source changed during the analysis")
    for input_id, input_path in (
        ("behavior_analysis_run", behavior_run_path),
        ("behavior_trials", behavior_trials_path),
        ("session_qc", session_qc_path),
    ):
        if _file_record(input_path) != input_records[input_id]:
            raise RuntimeError(f"upstream {input_id} changed during transition analysis")

    output_frames = {
        "behavior_transition_boundaries": (
            tables.boundaries,
            tables_directory / "behavior_transition_boundaries.csv",
        ),
        "session_behavior_transition_time_bins": (
            tables.session_time_bins,
            tables_directory / "session_behavior_transition_time_bins.parquet",
        ),
        "mouse_behavior_transition_time_bins": (
            tables.mouse_time_bins,
            tables_directory / "mouse_behavior_transition_time_bins.csv",
        ),
        "behavior_transition_time_trajectory_summary": (
            time_summary,
            tables_directory / "behavior_transition_time_trajectory_summary.csv",
        ),
        "session_behavior_transition_trial_bins": (
            tables.session_trial_bins,
            tables_directory / "session_behavior_transition_trial_bins.parquet",
        ),
        "mouse_behavior_transition_trial_bins": (
            tables.mouse_trial_bins,
            tables_directory / "mouse_behavior_transition_trial_bins.csv",
        ),
        "behavior_transition_trial_trajectory_summary": (
            trial_summary,
            tables_directory / "behavior_transition_trial_trajectory_summary.csv",
        ),
        "session_behavior_transition_effect_windows": (
            tables.session_effect_windows,
            tables_directory / "session_behavior_transition_effect_windows.parquet",
        ),
        "behavior_transition_effect_window_count_summary": (
            tables.effect_window_count_summary,
            tables_directory / "behavior_transition_effect_window_count_summary.csv",
        ),
        "session_behavior_transition_effects": (
            tables.session_effects,
            tables_directory / "session_behavior_transition_effects.csv",
        ),
        "mouse_behavior_transition_effects": (
            tables.mouse_effects,
            tables_directory / "mouse_behavior_transition_effects.csv",
        ),
        "behavior_transition_statistics": (
            statistics,
            tables_directory / "behavior_transition_statistics.csv",
        ),
        "behavior_transition_effect_statistic_sample_sizes": (
            tables.effect_statistic_sample_sizes,
            tables_directory / "behavior_transition_effect_statistic_sample_sizes.csv",
        ),
        "session_qc_d03_criterion_summary": (
            tables.qc_criterion_summary,
            tables_directory / "session_qc_d03_criterion_summary.csv",
        ),
        "session_qc_d03_metric_distributions": (
            tables.qc_metric_distributions,
            tables_directory / "session_qc_d03_metric_distributions.csv",
        ),
    }
    output_records: dict[str, dict[str, Any]] = {}
    for output_id, (frame, destination) in output_frames.items():
        dg.artifacts.write_frame(frame, destination)
        output_records[output_id] = _file_record(destination)

    environment_path = manifests_directory / "behavior_transition_environment.txt"
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        environment_path,
    )
    output_records["software_environment"] = _file_record(environment_path)
    lock_snapshot_path = manifests_directory / "behavior_transition_analysis_lock.yaml"
    dg.artifacts.write_text(lock_text, lock_snapshot_path)
    output_records["analysis_lock_snapshot"] = _file_record(lock_snapshot_path)

    source_records = [
        _file_record(REPOSITORY_ROOT / relative_path) for relative_path in LOCAL_SOURCE_PATHS
    ]
    if source_records != source_records_at_start:
        raise RuntimeError("behavior-transition source changed while outputs were written")
    completed_at = datetime.datetime.now(datetime.UTC)
    primary_statistics = statistics.filter(pl.col("p_value").is_not_null())
    run_record = {
        "analysis_id": ANALYSIS_ID,
        "script": "scripts/04_analyze_behavior_transitions.py",
        "analysis_scope": (
            "behavior-only transition QC; distinct from planned neural Figure 2 analyses"
        ),
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "analysis_status_definition": "execution and artifact validation passed",
        "evidentiary_status": "exploratory; not a confirmatory neural result",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "dandiset_doi": dg.data.DANDISET_DOI,
        "code_version": code_version,
        "analysis_lock": {
            "path": str(arguments.analysis_lock.resolve()),
            "sha256": _sha256_file(arguments.analysis_lock),
            "required_decisions": list(REQUIRED_DECISIONS),
            "status": "approved",
            "snapshot": output_records["analysis_lock_snapshot"],
        },
        "inputs": input_records,
        "local_sources": source_records,
        "configuration": dataclasses.asdict(config),
        "configuration_status": (
            "exploratory implementation choices outside D03; not frozen confirmatory lock decisions"
        ),
        "effect_minimum_trial_rationale": (
            "implementation-stage robustness rule added to prevent a one-trial session window "
            "from receiving full equal-session weight; exact support distributions are exported"
        ),
        "bin_definitions": {
            "time": ("half-open 120 s bins spanning [-360, +360) s relative to each anchor"),
            "trial": (
                "8-event-trial bins spanning ordinals -24:-1 and +1:+24; ordinals are "
                "assigned over all eligible go/catch events before condition stratification"
            ),
            "eligible_event_trial": (
                "completed, non-auto-rewarded, mutually exclusive go/catch trial with finite "
                "change time and canonical raw-lick response"
            ),
            "response": "exact-deduplicated raw lick in (0.150, 0.750] s after change",
            "latency": (
                "finite response latency among canonical responders only; conditional descriptive "
                "summary with separate missingness counts"
            ),
        },
        "boundary_definitions": {
            "reward_withdrawal_real": "first no-reward trial start",
            "reward_withdrawal_pseudo": "temporal midpoint of E1 within the same session",
            "reward_restoration_real": "final no-reward trial stop",
            "reward_restoration_pseudo": ("temporal midpoint of no reward within the same session"),
            "matching": (
                "each transition uses its own within-session, within-source-state temporal "
                "midpoint with identical bin/effect geometry; withdrawal and restoration "
                "kinetics are not pooled or constrained to match"
            ),
            "control_limitation": (
                "a single midpoint is a window-matched negative control, not a general smooth-"
                "drift control, and cannot exclude nonlinear within-block drift"
            ),
        },
        "effect_definitions": {
            "window": "3 minutes immediately before and after each real/pseudo anchor",
            "minimum_trial_support": (
                ">=5 go trials per window for go effects and >=3 catch trials per window for "
                "catch effects; all lower-support sessions remain in output with null estimates"
            ),
            "withdrawal_response_sign": "pre minus post; suppression is positive",
            "restoration_response_sign": "post minus pre; recovery is positive",
            "latency_sign": (
                "positive denotes slowing after withdrawal or speeding after restoration"
            ),
            "pseudo_controlled": "signed real step minus transition-specific pseudo step",
            "specificity": "pseudo-controlled go step minus pseudo-controlled catch step",
        },
        "inference_policy": {
            "primary_unit": "mouse",
            "aggregation": "equal-session mean within mouse; equal-mouse inference",
            "bootstrap_resamples": arguments.bootstrap_resamples,
            "sign_flip_resamples": arguments.sign_flip_resamples,
            "inferential_tests": [
                "pseudo-controlled go response effect for each transition",
                "pseudo-controlled go-minus-catch response specificity for each transition",
            ],
            "descriptive_only": [
                "real steps",
                "pseudo steps",
                "catch effects",
                "all responder-only latency effects",
            ],
            "multiplicity": (
                "Holm adjustment across withdrawal/restoration within separate go and "
                "go-minus-catch families"
            ),
        },
        "sample_sizes": {
            "n_sessions_inventory": session_qc.height,
            "n_mice_inventory": session_qc.get_column("subject_id").n_unique(),
            "n_technically_valid_sessions": session_qc.filter(
                pl.col("is_technically_valid")
            ).height,
            "n_technically_valid_mice": session_qc.filter(pl.col("is_technically_valid"))
            .get_column("subject_id")
            .n_unique(),
            "n_d03_threshold_selected_sessions": session_qc.filter(
                pl.col("is_good_session")
            ).height,
            "n_d03_threshold_selected_mice": session_qc.filter(pl.col("is_good_session"))
            .get_column("subject_id")
            .n_unique(),
            "n_transition_boundary_rows": tables.boundaries.height,
            "n_primary_inferential_rows": primary_statistics.height,
        },
        "outputs": output_records,
    }
    manifest_path = manifests_directory / "behavior_transition_analysis_run.json"
    dg.artifacts.write_json(run_record, manifest_path)
    print(
        "Wrote behavior-transition QC for "
        f"{run_record['sample_sizes']['n_technically_valid_sessions']} sessions and "
        f"{run_record['sample_sizes']['n_technically_valid_mice']} mice to {tables_directory}"
    )


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for name in ("seed", "bootstrap_resamples", "sign_flip_resamples"):
        value = getattr(arguments, name)
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            comparator = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {comparator} integer")


def _validate_locked_dandiset(lock: dict[str, Any]) -> None:
    value = lock["decisions"]["D01"]["value"]
    expected = f"DANDI:{dg.data.DANDISET_ID}/{dg.data.DANDISET_VERSION}"
    if value != expected:
        raise RuntimeError(f"D01 locks {value!r}; expected {expected!r}")


def _read_behavior_run(path: pathlib.Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read behavior run manifest {path}: {error}") from error
    if (
        record.get("analysis_status") != "pass"
        or record.get("run_status") != "complete"
        or record.get("authoritative") is not True
    ):
        raise RuntimeError("behavior analysis manifest is not a completed authoritative pass")
    if record.get("dandiset_version") != dg.data.DANDISET_VERSION:
        raise RuntimeError("behavior analysis manifest has the wrong DANDI version")
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("behavior analysis manifest has no outputs mapping")
    for output_id in ("behavior_trials", "session_qc"):
        output = outputs.get(output_id)
        if not isinstance(output, dict) or not output.get("sha256"):
            raise RuntimeError(f"behavior analysis manifest lacks {output_id!r} provenance")
    checkpoint = record.get("behavior_trials_checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("behavior analysis manifest lacks trusted trial-checkpoint provenance")
    if checkpoint.get("status") not in {
        "created_from_remote_nwb",
        "validated_existing_not_refreshed",
    }:
        raise RuntimeError("behavior analysis manifest has an untrusted trial-checkpoint status")
    if checkpoint.get("output_matches_trusted_checkpoint") is not True:
        raise RuntimeError("behavior output does not match its trusted canonical trial checkpoint")
    trusted_sha256 = checkpoint.get("trusted_behavior_trials_sha256")
    if (
        not isinstance(trusted_sha256, str)
        or trusted_sha256 != outputs["behavior_trials"]["sha256"]
    ):
        raise RuntimeError("trusted trial-checkpoint hash differs from behavior output provenance")
    return record


def _validate_behavior_inputs(
    behavior_run: dict[str, Any],
    *,
    behavior_trials_path: pathlib.Path,
    session_qc_path: pathlib.Path,
    behavior_run_path: pathlib.Path,
    behavior_lock_snapshot_path: pathlib.Path | None = None,
    session_inventory_path: pathlib.Path | None = None,
    audit_run_path: pathlib.Path | None = None,
) -> dict[str, dict[str, Any]]:
    try:
        current_behavior_run = json.loads(behavior_run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not re-read behavior run manifest: {error}") from error
    if current_behavior_run != behavior_run:
        raise RuntimeError("behavior run manifest changed while it was being validated")
    records = {
        "behavior_analysis_run": _file_record(behavior_run_path),
        "behavior_trials": _file_record(behavior_trials_path),
        "session_qc": _file_record(session_qc_path),
    }
    for output_id in ("behavior_trials", "session_qc"):
        expected = behavior_run["outputs"][output_id]["sha256"]
        if records[output_id]["sha256"] != expected:
            raise RuntimeError(f"{output_id} hash differs from behavior analysis manifest")
    if behavior_lock_snapshot_path is not None:
        snapshot_record = _file_record(behavior_lock_snapshot_path)
        if snapshot_record["sha256"] != behavior_run.get("analysis_lock_sha256"):
            raise RuntimeError("behavior lock snapshot hash differs from behavior manifest")
        records["behavior_analysis_lock_snapshot"] = snapshot_record
    if session_inventory_path is not None:
        inventory_record = _file_record(session_inventory_path)
        if inventory_record["sha256"] != behavior_run.get("session_inventory_sha256"):
            raise RuntimeError("session inventory hash differs from behavior manifest")
        records["session_inventory"] = inventory_record
    if audit_run_path is not None:
        audit_record = _file_record(audit_run_path)
        try:
            audit_value = json.loads(audit_run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not read upstream audit manifest: {error}") from error
        if audit_value != behavior_run.get("input_audit_run"):
            raise RuntimeError("upstream audit manifest differs from behavior analysis snapshot")
        if _sha256_file(audit_run_path) != audit_record["sha256"]:
            raise RuntimeError("upstream audit manifest changed while it was being validated")
        records["audit_run"] = audit_record
    return records


def _validate_real_inputs(trials: pl.DataFrame, session_qc: pl.DataFrame) -> None:
    trial_sources = trials.select("_nwb_path", "subject_id").unique()
    qc_sources = session_qc.select("_nwb_path", "subject_id").unique()
    if trial_sources.height != trials.get_column("_nwb_path").n_unique():
        raise RuntimeError("behavior trials map a session to multiple mice")
    if qc_sources.height != session_qc.height:
        raise RuntimeError("session_qc has duplicate session rows")
    if trial_sources.sort("_nwb_path").to_dicts() != qc_sources.sort("_nwb_path").to_dicts():
        raise RuntimeError("behavior trial and session-QC coverage differs")
    versions = trials.get_column("dandiset_version").drop_nulls().unique().to_list()
    if versions != [dg.data.DANDISET_VERSION]:
        raise RuntimeError(f"behavior trials contain unexpected DANDI versions: {versions!r}")
    reward_blocks = set(trials.get_column("reward_block").drop_nulls().unique())
    if reward_blocks != {"engaged_1", "no_reward", "engaged_2"}:
        raise RuntimeError(f"behavior trials contain unexpected reward blocks: {reward_blocks!r}")
    starts = trials.get_column("response_window_start_seconds").drop_nulls().unique().to_list()
    stops = trials.get_column("response_window_stop_seconds").drop_nulls().unique().to_list()
    if starts != [dg.quality.DEFAULT_RESPONSE_WINDOW_START_SECONDS] or stops != [
        dg.quality.DEFAULT_RESPONSE_WINDOW_STOP_SECONDS
    ]:
        raise RuntimeError("behavior trials do not use the approved canonical response window")
    _validate_d03_threshold_columns(session_qc)


def _validate_d03_threshold_columns(session_qc: pl.DataFrame) -> None:
    thresholds = dg.quality.DEFAULT_SESSION_QUALITY_THRESHOLDS
    expected = {
        "minimum_go_trials_threshold": thresholds.minimum_go_trials_per_block,
        "minimum_engaged_catch_trials_threshold": (
            thresholds.minimum_catch_trials_per_engaged_block
        ),
        "minimum_engaged_response_rate_threshold": thresholds.minimum_engaged_response_rate,
        "minimum_engaged_dprime_threshold": thresholds.minimum_engaged_dprime,
        "maximum_no_reward_response_rate_threshold": thresholds.maximum_no_reward_response_rate,
        "minimum_reward_suppression_drop_threshold": thresholds.minimum_reward_suppression_drop,
    }
    for column, expected_value in expected.items():
        values = session_qc.get_column(column).drop_nulls().unique().to_list()
        if values != [expected_value]:
            raise RuntimeError(f"{column} differs from approved D03: {values!r}")


def _validate_results(
    tables: dg.behavior_transitions.TransitionTables,
    time_summary: pl.DataFrame,
    trial_summary: pl.DataFrame,
    statistics: pl.DataFrame,
    *,
    session_qc: pl.DataFrame,
    config: dg.behavior_transitions.TransitionConfig,
) -> None:
    n_sessions = session_qc.height
    n_mice = session_qc.get_column("subject_id").n_unique()
    n_time_bins = int(2 * config.time_window_seconds / config.time_bin_width_seconds)
    n_trial_bins = 2 * config.trial_window_count // config.trial_bin_width
    if tables.boundaries.height != n_sessions * len(dg.behavior_transitions.TRANSITION_IDS):
        raise RuntimeError("transition boundary table does not preserve every session")
    technical_boundaries = tables.boundaries.filter(pl.col("is_technically_valid"))
    if not technical_boundaries.get_column("boundary_values_valid").all():
        raise RuntimeError("a technically valid session lacks valid transition boundaries")
    expected_time_rows = n_sessions * 2 * 2 * n_time_bins * 2
    expected_trial_rows = n_sessions * 2 * 2 * n_trial_bins * 2
    if tables.session_time_bins.height != expected_time_rows:
        raise RuntimeError("session time-bin grid is incomplete")
    if tables.session_trial_bins.height != expected_trial_rows:
        raise RuntimeError("session trial-bin grid is incomplete")
    if tables.mouse_time_bins.height != n_mice * 2 * 2 * n_time_bins * 2:
        raise RuntimeError("mouse time-bin grid is incomplete")
    if tables.mouse_trial_bins.height != n_mice * 2 * 2 * n_trial_bins * 2:
        raise RuntimeError("mouse trial-bin grid is incomplete")
    if time_summary.height != 2 * 2 * n_time_bins * 2:
        raise RuntimeError("time trajectory summary grid is incomplete")
    if trial_summary.height != 2 * 2 * n_trial_bins * 2:
        raise RuntimeError("trial trajectory summary grid is incomplete")
    if tables.effect_window_count_summary.height != 2 * 2 * 2 * 2:
        raise RuntimeError("effect-window count summary grid is incomplete")
    if tables.effect_statistic_sample_sizes.height != 30:
        raise RuntimeError("effect-statistic sample-size table is incomplete")
    if set(tables.effect_statistic_sample_sizes.get_column("result_id")) != set(
        statistics.get_column("result_id")
    ):
        raise RuntimeError("effect sample-size rows do not match transition statistics")
    dg.statistics.validate_statistics_table(statistics)
    if statistics.filter(pl.col("analysis_tier") != ANALYSIS_TIER).height:
        raise RuntimeError("transition statistics are not labeled exploratory")
    tested = statistics.filter(pl.col("p_value").is_not_null())
    if tested.height != 4:
        raise RuntimeError("unexpected number of inferential transition tests")
    if statistics.filter(
        pl.col("result_id").str.contains("latency") & pl.col("p_value").is_not_null()
    ).height:
        raise RuntimeError("responder-only latency must remain descriptive")


def _file_record(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"required file does not exist: {resolved}")
    try:
        display_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version(repository: pathlib.Path) -> str:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{revision}+dirty" if dirty else revision


if __name__ == "__main__":
    main()
