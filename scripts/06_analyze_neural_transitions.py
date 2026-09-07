# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Build discovery-only neural reward-transition outputs for Figure 2.

The analysis reads only behavior-eligible discovery sessions. Numeric unit
metadata and ragged spike times are projected through ``lazynwb.scan_nwb``;
the unit quality string is decoded with lazynwb's obstore range reader and
custom HDF5 parser. A runtime guard makes any legacy accessor call fatal.
Confirmation-session identities and neural data are never loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import gc
import hashlib
import json
import math
import pathlib
import subprocess
import sys
from typing import Any

import lazynwb
import lazynwb.file_io
import numpy as np
import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.behavior_transitions  # noqa: E402
import dg.data  # noqa: E402
import dg.lazynwb_obstore  # noqa: E402
import dg.neural_transitions  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "neural_transition_axis"
ANALYSIS_TIER = "discovery"
DEFAULT_SEED = 1051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
CACHE_SCHEMA_VERSION = 1
SPIKE_BATCH_SIZE = 32
COMPATIBLE_SESSION_CACHE_SIGNATURES = frozenset(
    {"9915c30fc9181d3daded3576fa81aa32d9177fc737749dd925f86292262bad00"}
)
EXPECTED_DISCOVERY_MICE = 12
EXPECTED_DISCOVERY_SESSIONS = 23
UNIT_NUMERIC_COLUMNS = (
    "id",
    "peak_channel_id",
    "isi_violations",
    "amplitude_cutoff",
    "firing_rate",
    "presence_ratio",
    "_table_index",
    "_nwb_path",
)
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/behavior_transitions.py",
    "src/dg/data.py",
    "src/dg/lazynwb_obstore.py",
    "src/dg/neural_transitions.py",
    "src/dg/statistics.py",
    "scripts/06_analyze_neural_transitions.py",
)
RESULT_IDS = {
    "reward_withdrawal": "reward_withdrawal_early_axis_real_minus_pseudo_mouse_mean",
    "reward_restoration": "reward_restoration_early_axis_real_minus_pseudo_mouse_mean",
    "oof_auc": "e1_vs_nr_axis_oof_performance_mouse_mean",
    "e2_recovery": "e2_axis_transfer_recovery_mouse_mean",
}


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse repository-relative output and deterministic inference options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
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
        "--recompute-session-cache",
        action="store_true",
        help="Ignore valid per-session score caches and repeat NWB reads",
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Run the discovery analysis and publish a complete or failed marker."""

    arguments = parse_arguments()
    _validate_arguments(arguments)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = result_root / "manifests" / "neural_transition_analysis_run.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "run_status": "in_progress",
        "authoritative": False,
        "script": "scripts/06_analyze_neural_transitions.py",
        "started_at_utc": started_at.isoformat(),
        "confirmation_accessed": False,
    }
    dg.artifacts.write_json(in_progress, manifest_path)
    try:
        _run_analysis(
            arguments,
            result_root=result_root,
            manifest_path=manifest_path,
            started_at=started_at,
        )
    except BaseException as error:
        with contextlib.suppress(Exception):
            dg.artifacts.write_json(
                {
                    **in_progress,
                    "analysis_status": "failed",
                    "run_status": "failed",
                    "failed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                manifest_path,
            )
        raise


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")
    if arguments.bootstrap_resamples < 1 or arguments.sign_flip_resamples < 1:
        raise ValueError("resample counts must be positive")


def _run_analysis(
    arguments: argparse.Namespace,
    *,
    result_root: pathlib.Path,
    manifest_path: pathlib.Path,
    started_at: datetime.datetime,
) -> None:
    tables_directory = result_root / "tables"
    manifests_directory = result_root / "manifests"
    cache_directory = result_root / "cache" / "neural_transition_sessions"
    cache_directory.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "behavior_trials": tables_directory / "behavior_trials.parquet",
        "session_qc": tables_directory / "session_qc.parquet",
        "session_cohort_assignments": tables_directory / "session_cohort_assignments.csv",
        "analysis_lock": REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
    }
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Figure 2 inputs: " + ", ".join(map(str, missing)))
    input_records = {
        key: _file_record(
            path,
            base=result_root if path.is_relative_to(result_root) else REPOSITORY_ROOT,
        )
        for key, path in input_paths.items()
    }
    local_source_records = {
        path: _file_record(REPOSITORY_ROOT / path, base=REPOSITORY_ROOT)
        for path in LOCAL_SOURCE_PATHS
    }
    analysis_signature = _analysis_signature(input_records, local_source_records)

    assignments = pl.read_csv(
        input_paths["session_cohort_assignments"],
        schema_overrides={"subject_id": pl.String},
    )
    discovery_sources = _select_discovery_sources(assignments)
    behavior_trials = pl.read_parquet(input_paths["behavior_trials"])
    session_qc = pl.read_parquet(input_paths["session_qc"])
    discovery_trials, discovery_qc = _select_discovery_behavior_inputs(
        behavior_trials,
        session_qc,
        discovery_sources,
    )
    selected_trials = dg.neural_transitions.select_lick_free_familiar_trials(discovery_trials)
    _validate_selected_trial_coverage(selected_trials, discovery_sources)
    boundaries = dg.behavior_transitions.build_transition_boundaries(
        discovery_trials,
        discovery_qc,
    )

    discovery_sources_path = tables_directory / "discovery_session_sources.parquet"
    dg.artifacts.write_frame(discovery_sources, discovery_sources_path)
    print(
        "Figure 2 discovery input: "
        f"{discovery_sources.height} sessions, "
        f"{discovery_sources.get_column('subject_id').n_unique()} mice, "
        f"{selected_trials.height} eligible trials",
        flush=True,
    )

    unit_tables: list[pl.DataFrame] = []
    score_tables: list[pl.DataFrame] = []
    session_metric_tables: list[pl.DataFrame] = []
    source_rows = discovery_sources.sort("subject_id", "ecephys_session_id").to_dicts()
    for session_index, source_row in enumerate(source_rows, start=1):
        source = source_row["_nwb_path"]
        session_trials = selected_trials.filter(pl.col("_nwb_path") == source)
        cache_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        cache_root = cache_directory / cache_key
        cached = None
        if not arguments.recompute_session_cache:
            cached = _read_session_cache(
                cache_root,
                source=source,
                analysis_signature=analysis_signature,
            )
        if cached is None:
            print(
                f"[{session_index:02d}/{len(source_rows):02d}] reading unit QC via lazynwb/obstore",
                flush=True,
            )
            unit_qc, trial_scores, session_metrics = _analyze_session(
                source_row,
                session_trials,
            )
            _write_session_cache(
                cache_root,
                source=source,
                analysis_signature=analysis_signature,
                unit_qc=unit_qc,
                trial_scores=trial_scores,
                session_metrics=session_metrics,
            )
        else:
            unit_qc, trial_scores, session_metrics = cached
            print(
                f"[{session_index:02d}/{len(source_rows):02d}] loaded validated session cache",
                flush=True,
            )
        unit_tables.append(unit_qc)
        score_tables.append(trial_scores)
        session_metric_tables.append(session_metrics)
        gc.collect()

    unit_qc = pl.concat(unit_tables, how="diagonal_relaxed").sort(
        "subject_id", "ecephys_session_id", "_table_index"
    )
    trial_scores = pl.concat(score_tables, how="diagonal_relaxed").sort(
        "subject_id", "ecephys_session_id", "change_time", "_table_index"
    )
    session_metrics = pl.concat(session_metric_tables, how="diagonal_relaxed").sort(
        "subject_id", "ecephys_session_id"
    )
    transition_tables = dg.neural_transitions.summarize_transition_scores(
        trial_scores,
        boundaries,
    )
    trajectory = _plot_facing_trajectory(transition_tables.mouse_trajectories)
    mouse_effects = _plot_facing_mouse_effects(transition_tables.session_contrasts)
    mouse_axis_metrics = _aggregate_axis_metrics(session_metrics)
    statistics = _build_statistics(
        mouse_effects,
        mouse_axis_metrics,
        session_metrics=session_metrics,
        session_contrasts=transition_tables.session_contrasts,
        seed=arguments.seed,
        n_bootstrap=arguments.bootstrap_resamples,
        n_sign_flips=arguments.sign_flip_resamples,
        code_version=_code_version(REPOSITORY_ROOT),
    )

    output_paths = {
        "discovery_session_sources": discovery_sources_path,
        "neural_transition_unit_qc": tables_directory / "neural_transition_unit_qc.parquet",
        "neural_transition_trial_scores": (
            tables_directory / "neural_transition_trial_scores.parquet"
        ),
        "neural_transition_axis_sessions": (
            tables_directory / "neural_transition_axis_sessions.csv"
        ),
        "mouse_neural_axis_metrics": tables_directory / "mouse_neural_axis_metrics.csv",
        "session_neural_transition_trajectories": (
            tables_directory / "session_neural_transition_trajectories.parquet"
        ),
        "session_neural_transition_effects": (
            tables_directory / "session_neural_transition_effects.csv"
        ),
        "neural_transition_trajectory": tables_directory / "neural_transition_trajectory.csv",
        "mouse_neural_transition_effects": (
            tables_directory / "mouse_neural_transition_effects.csv"
        ),
        "neural_transition_statistics": tables_directory / "neural_transition_statistics.csv",
        "neural_transition_environment": (
            manifests_directory / "neural_transition_environment.txt"
        ),
    }
    output_frames = {
        "neural_transition_unit_qc": unit_qc,
        "neural_transition_trial_scores": trial_scores,
        "neural_transition_axis_sessions": session_metrics,
        "mouse_neural_axis_metrics": mouse_axis_metrics,
        "session_neural_transition_trajectories": transition_tables.session_trajectories,
        "session_neural_transition_effects": transition_tables.session_contrasts,
        "neural_transition_trajectory": trajectory,
        "mouse_neural_transition_effects": mouse_effects,
        "neural_transition_statistics": statistics,
    }
    for key, frame in output_frames.items():
        dg.artifacts.write_frame(frame, output_paths[key])
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        output_paths["neural_transition_environment"],
    )

    outputs = {key: _file_record(path, base=result_root) for key, path in output_paths.items()}
    completed_at = datetime.datetime.now(datetime.UTC)
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "complete",
        "run_status": "complete",
        "authoritative": True,
        "preliminary_vertical_slice": True,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "code_version": _code_version(REPOSITORY_ROOT),
        "confirmation_accessed": False,
        "confirmation_policy": "sealed; no confirmation identity or neural source was loaded",
        "discovery_counts": {
            "mice": discovery_sources.get_column("subject_id").n_unique(),
            "sessions": discovery_sources.height,
            "all_units": unit_qc.height,
            "well_isolated_units": unit_qc.filter(pl.col("well_isolated")).height,
            "scored_trials": trial_scores.height,
        },
        "unit_filter": (
            "isi_violations < 0.5; amplitude_cutoff < 0.1; quality == good; "
            "D04 engaged-block consistency not yet applied in this preliminary discovery slice"
        ),
        "trial_filter": (
            "completed non-auto familiar full-contrast physical image changes; "
            "no lick in the closed +/-150 ms peristimulus interval"
        ),
        "model": dataclasses.asdict(dg.neural_transitions.DEFAULT_NEURAL_TRANSITION_CONFIG),
        "data_access": {
            "library": "lazynwb",
            "version": "1.0.0.dev8",
            "transport": "obstore byte-range reads",
            "numeric_and_spike_reader": "lazynwb.scan_nwb custom reader",
            "quality_reader": (
                "lazynwb obstore range reader plus custom HDF5 parser global-heap decoder"
            ),
            "spike_batch_size_units": SPIKE_BATCH_SIZE,
            "peak_spike_materialization": "one bounded unit batch per session",
            "legacy_accessor_forbidden_at_runtime": True,
            "legacy_accessor_calls": 0,
        },
        "inputs": input_records,
        "local_sources": local_source_records,
        "outputs": outputs,
    }
    dg.artifacts.write_json(manifest, manifest_path)
    print(
        "Completed Figure 2 neural analysis: "
        f"{unit_qc.filter(pl.col('well_isolated')).height} units, "
        f"{trial_scores.height} trials",
        flush=True,
    )


def _select_discovery_sources(assignments: pl.DataFrame) -> pl.DataFrame:
    required = {
        "_nwb_path",
        "subject_id",
        "cohort_assignment",
        "session_behavior_eligible",
        "neural_activity_used_for_allocation",
        "ecephys_session_id",
        "behavior_session_id",
    }
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"cohort assignments missing columns: {sorted(missing)}")
    if assignments.filter(pl.col("neural_activity_used_for_allocation").fill_null(True)).height:
        raise RuntimeError("D05 assignment used neural activity")
    cross_cohort = (
        assignments.filter(pl.col("cohort_assignment").is_in(("discovery", "confirmation")))
        .group_by("subject_id")
        .agg(pl.col("cohort_assignment").n_unique().alias("n_cohorts"))
        .filter(pl.col("n_cohorts") != 1)
    )
    if cross_cohort.height:
        raise RuntimeError("a mouse crosses discovery and confirmation cohorts")
    discovery = (
        assignments.filter(
            (pl.col("cohort_assignment") == "discovery")
            & pl.col("session_behavior_eligible").fill_null(False)
        )
        .select(
            "_nwb_path",
            pl.col("subject_id").cast(pl.String),
            "ecephys_session_id",
            "behavior_session_id",
            *(column for column in ("asset_id", "path") if column in assignments.columns),
        )
        .sort("subject_id", "ecephys_session_id")
    )
    if discovery.select("_nwb_path").n_unique() != discovery.height:
        raise ValueError("discovery session sources are not unique")
    if discovery.height != EXPECTED_DISCOVERY_SESSIONS:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_SESSIONS} behavior-eligible discovery sessions; "
            f"found {discovery.height}"
        )
    if discovery.get_column("subject_id").n_unique() != EXPECTED_DISCOVERY_MICE:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_MICE} discovery mice; "
            f"found {discovery.get_column('subject_id').n_unique()}"
        )
    return discovery


def _select_discovery_behavior_inputs(
    behavior_trials: pl.DataFrame,
    session_qc: pl.DataFrame,
    discovery_sources: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    source_keys = discovery_sources.select("_nwb_path")
    trials = behavior_trials.join(source_keys, on="_nwb_path", how="inner", validate="m:1")
    qc = session_qc.join(source_keys, on="_nwb_path", how="inner", validate="1:1")
    if qc.height != discovery_sources.height:
        raise RuntimeError("discovery assignment and session QC do not bind one-to-one")
    if qc.filter(~pl.col("is_good_session").fill_null(False)).height:
        raise RuntimeError("a behavior-ineligible session entered discovery neural analysis")
    trial_sources = set(trials.get_column("_nwb_path"))
    expected_sources = set(discovery_sources.get_column("_nwb_path"))
    if trial_sources != expected_sources:
        raise RuntimeError("behavior trials do not cover every discovery session exactly")
    return trials, qc


def _validate_selected_trial_coverage(
    trials: pl.DataFrame,
    discovery_sources: pl.DataFrame,
) -> None:
    coverage = trials.group_by("_nwb_path").agg(
        *(
            pl.col("reward_block").eq(block).sum().alias(block)
            for block in ("engaged_1", "no_reward", "engaged_2")
        )
    )
    if coverage.height != discovery_sources.height:
        raise RuntimeError("lick-free familiar trials do not cover every discovery session")
    minimum = dg.neural_transitions.DEFAULT_NEURAL_TRANSITION_CONFIG.minimum_axis_trials_per_state
    if coverage.filter(
        (pl.col("engaged_1") < minimum)
        | (pl.col("no_reward") < minimum)
        | (pl.col("engaged_2") < 1)
    ).height:
        raise RuntimeError("a discovery session lacks trials needed for the reward-state axis")


def _analyze_session(
    source_row: dict[str, Any],
    session_trials: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = str(source_row["_nwb_path"])
    session_trials = session_trials.sort("change_time", "_table_index")
    with _forbid_legacy_accessors():
        numeric = (
            lazynwb.scan_nwb(
                source,
                "/units",
                raise_on_missing=True,
                disable_progress=True,
            )
            .select(*UNIT_NUMERIC_COLUMNS)
            .collect()
        )
        quality = dg.lazynwb_obstore.read_vlen_string_column(
            source,
            "/units",
            "quality",
        )
        unit_qc = _build_unit_qc(numeric, quality, source_row=source_row)
        selected = unit_qc.filter(pl.col("well_isolated")).sort("_table_index")
        if selected.is_empty():
            raise RuntimeError("a discovery session has no well-isolated units")
        selected_indices = selected.get_column("_table_index").to_list()
        print(
            f"    {selected.height}/{unit_qc.height} units selected; "
            f"reading spike_times in batches of {SPIKE_BATCH_SIZE}",
            flush=True,
        )
        features = _build_feature_matrix_from_obstore(
            source,
            selected_indices,
            session_trials.get_column("change_time").to_numpy(),
        )

    axis = dg.neural_transitions.fit_reward_state_axis(
        features,
        session_trials.get_column("reward_block").to_numpy(),
        session_trials.get_column("change_time").to_numpy(),
    )
    trial_scores = session_trials.select(
        "_nwb_path",
        "subject_id",
        "ecephys_session_id",
        "behavior_session_id",
        "_table_index",
        "id",
        "change_time",
        "reward_block",
        "change_image_name",
        "novel_image_id",
        "lick_exclusion_half_width_seconds",
    ).with_columns(
        pl.Series("axis_score", axis.scores, dtype=pl.Float64),
        pl.Series("score_role", axis.score_roles, dtype=pl.String),
        pl.lit(axis.selected_lambda).alias("axis_selected_lambda"),
    )
    weights = pl.DataFrame(
        {
            "_table_index": selected_indices,
            "axis_weight": axis.feature_weights,
        },
        schema={"_table_index": pl.UInt32, "axis_weight": pl.Float64},
    )
    unit_qc = unit_qc.join(weights, on="_table_index", how="left", validate="1:1").with_columns(
        pl.when(pl.col("well_isolated"))
        .then(pl.lit("final_e1_nr_reward_state_axis"))
        .otherwise(pl.lit("excluded_by_isolation_filter"))
        .alias("axis_model_role"),
        pl.lit(axis.intercept).alias("axis_intercept"),
        pl.lit(axis.selected_lambda).alias("axis_selected_lambda"),
    )
    block_means = {
        block: float(
            trial_scores.filter(pl.col("reward_block") == block).get_column("axis_score").mean()
        )
        for block in ("engaged_1", "no_reward", "engaged_2")
    }
    session_metrics = pl.DataFrame(
        {
            "_nwb_path": [source],
            "subject_id": [str(source_row["subject_id"])],
            "ecephys_session_id": [int(source_row["ecephys_session_id"])],
            "n_units": [selected.height],
            "n_trials": [trial_scores.height],
            "n_engaged_1_trials": [axis.n_engaged_1],
            "n_no_reward_trials": [axis.n_no_reward],
            "n_engaged_2_trials": [axis.n_engaged_2],
            "engaged_1_oof_mean_axis_score": [block_means["engaged_1"]],
            "no_reward_oof_mean_axis_score": [block_means["no_reward"]],
            "engaged_2_transfer_mean_axis_score": [block_means["engaged_2"]],
            "e1_nr_oof_auc": [axis.oof_auc],
            "e1_nr_oof_mean_difference": [block_means["engaged_1"] - block_means["no_reward"]],
            "e2_transfer_recovery": [block_means["engaged_2"] - block_means["no_reward"]],
            "axis_selected_lambda": [axis.selected_lambda],
            "outer_selected_lambdas": [json.dumps(axis.outer_selected_lambdas)],
            "unit_filter": ["isolation_only_preliminary_d04_not_applied"],
        }
    )
    print(
        f"    axis fit complete: {trial_scores.height} trials, OOF AUC={axis.oof_auc:.3f}",
        flush=True,
    )
    del features
    return unit_qc, trial_scores, session_metrics


def _build_feature_matrix_from_obstore(
    source: str,
    selected_indices: list[int],
    event_times: np.ndarray,
) -> np.ndarray:
    """Reduce bounded spike-time batches directly into a compact feature matrix."""

    matrix = np.empty((event_times.size, len(selected_indices)), dtype=float)
    next_column = 0
    n_batches = math.ceil(len(selected_indices) / SPIKE_BATCH_SIZE)

    def consume(batch_rows: tuple[int, ...], spike_rows: list[list[Any]]) -> None:
        nonlocal next_column
        expected = tuple(selected_indices[next_column : next_column + len(batch_rows)])
        if batch_rows != expected:
            raise RuntimeError("obstore spike batch order differs from selected unit order")
        batch_features = dg.neural_transitions.build_spike_feature_matrix(
            spike_rows,
            event_times,
        )
        matrix[:, next_column : next_column + len(batch_rows)] = batch_features
        next_column += len(batch_rows)
        completed_batches = math.ceil(next_column / SPIKE_BATCH_SIZE)
        if completed_batches == n_batches or completed_batches % 10 == 0:
            print(
                f"      reduced spike batch {completed_batches}/{n_batches}",
                flush=True,
            )
        del batch_features

    dg.lazynwb_obstore.map_indexed_numeric_column_batches(
        source,
        "/units",
        "spike_times",
        selected_indices,
        consume,
        batch_size=SPIKE_BATCH_SIZE,
    )
    if next_column != len(selected_indices):
        raise RuntimeError("obstore spike batches did not cover every selected unit")
    return matrix


def _build_unit_qc(
    numeric: pl.DataFrame,
    quality: pl.DataFrame,
    *,
    source_row: dict[str, Any],
) -> pl.DataFrame:
    if numeric.height != quality.height:
        raise RuntimeError("numeric unit metadata and quality labels differ in length")
    units = numeric.join(
        quality,
        on=("_nwb_path", "_table_index"),
        how="left",
        validate="1:1",
    )
    if units.get_column("quality").null_count():
        raise RuntimeError("unit quality custom read left unmatched rows")
    units = units.with_columns(
        (
            pl.col("isi_violations").is_not_null()
            & pl.col("isi_violations").is_finite()
            & (pl.col("isi_violations") < 0.5)
        )
        .fill_null(False)
        .alias("isi_violations_pass"),
        (
            pl.col("amplitude_cutoff").is_not_null()
            & pl.col("amplitude_cutoff").is_finite()
            & (pl.col("amplitude_cutoff") < 0.1)
        )
        .fill_null(False)
        .alias("amplitude_cutoff_pass"),
        pl.col("quality").eq("good").fill_null(False).alias("quality_pass"),
    ).with_columns(
        pl.all_horizontal(
            "isi_violations_pass",
            "amplitude_cutoff_pass",
            "quality_pass",
        ).alias("well_isolated")
    )
    exclusion_reasons = []
    for row in units.select(
        "isi_violations_pass", "amplitude_cutoff_pass", "quality_pass"
    ).iter_rows(named=True):
        failed = [
            name
            for name, passed in (
                ("isi_violations", row["isi_violations_pass"]),
                ("amplitude_cutoff", row["amplitude_cutoff_pass"]),
                ("quality", row["quality_pass"]),
            )
            if not passed
        ]
        exclusion_reasons.append("included" if not failed else ";".join(failed))
    return units.with_columns(
        pl.lit(str(source_row["subject_id"])).alias("subject_id"),
        pl.lit(int(source_row["ecephys_session_id"])).alias("ecephys_session_id"),
        pl.Series("unit_inclusion_status", exclusion_reasons, dtype=pl.String),
        pl.lit("not_applied_preliminary_isolation_only").alias("engaged_block_consistency_status"),
    )


@contextlib.contextmanager
def _forbid_legacy_accessors():
    original_get_accessor = lazynwb.file_io._get_accessor
    original_open_hdf5 = getattr(lazynwb.file_io, "_open_hdf5", None)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "legacy NWB accessor invoked; Figure 2 requires lazynwb/obstore custom reads"
        )

    lazynwb.file_io._get_accessor = forbidden
    if original_open_hdf5 is not None:
        lazynwb.file_io._open_hdf5 = forbidden
    try:
        yield
    finally:
        lazynwb.file_io._get_accessor = original_get_accessor
        if original_open_hdf5 is not None:
            lazynwb.file_io._open_hdf5 = original_open_hdf5


def _plot_facing_trajectory(mouse_trajectories: pl.DataFrame) -> pl.DataFrame:
    transition_names = {
        "reward_withdrawal": "withdrawal",
        "reward_restoration": "restoration",
    }
    return mouse_trajectories.select(
        pl.col("subject_id").cast(pl.String),
        pl.col("transition_id").replace_strict(transition_names).alias("transition"),
        "anchor_type",
        (pl.col("relative_bin_center_seconds") / 60.0).alias("bin_center_minutes"),
        "mean_axis_score",
        "n_sessions",
        "n_trials",
    ).sort("transition", "anchor_type", "subject_id", "bin_center_minutes")


def _plot_facing_mouse_effects(session_contrasts: pl.DataFrame) -> pl.DataFrame:
    transition_names = {
        "reward_withdrawal": "withdrawal",
        "reward_restoration": "restoration",
    }
    return (
        session_contrasts.filter(pl.col("status") == "pass")
        .group_by("subject_id", "transition_id")
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("real_signed_step").mean().alias("real_effect"),
            pl.col("pseudo_signed_step").mean().alias("pseudo_effect"),
            pl.col("real_minus_pseudo").mean().alias("real_minus_pseudo"),
        )
        .with_columns(pl.col("transition_id").replace_strict(transition_names).alias("transition"))
        .select(
            pl.col("subject_id").cast(pl.String),
            "transition",
            "real_effect",
            "pseudo_effect",
            "real_minus_pseudo",
            "n_sessions",
        )
        .sort("transition", "subject_id")
    )


def _aggregate_axis_metrics(session_metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        session_metrics.group_by("subject_id")
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("n_units").sum().alias("n_units"),
            pl.col("n_engaged_1_trials").sum(),
            pl.col("n_no_reward_trials").sum(),
            pl.col("n_engaged_2_trials").sum(),
            pl.col("e1_nr_oof_auc").mean(),
            pl.col("e1_nr_oof_mean_difference").mean(),
            pl.col("e2_transfer_recovery").mean(),
        )
        .sort("subject_id")
    )


def _build_statistics(
    mouse_effects: pl.DataFrame,
    mouse_axis_metrics: pl.DataFrame,
    *,
    session_metrics: pl.DataFrame,
    session_contrasts: pl.DataFrame,
    seed: int,
    n_bootstrap: int,
    n_sign_flips: int,
    code_version: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, transition in enumerate(("withdrawal", "restoration")):
        mouse_values = mouse_effects.filter(pl.col("transition") == transition)
        internal_transition = f"reward_{transition}"
        supported_sessions = session_contrasts.filter(
            (pl.col("transition_id") == internal_transition) & (pl.col("status") == "pass")
        )
        support = _session_support_counts(
            supported_sessions,
            session_metrics,
            trial_count_columns=(
                "n_real_pre_trials",
                "n_real_post_trials",
                "n_pseudo_pre_trials",
                "n_pseudo_post_trials",
            ),
        )
        rows.append(
            _statistic_row(
                mouse_values,
                value_column="real_minus_pseudo",
                null_value=0.0,
                result_id=RESULT_IDS[internal_transition],
                contrast_id=f"{internal_transition}_real_minus_pseudo_six_minute_step",
                hypothesis=(
                    "Reward-state-consistent axis change is larger at the real "
                    "transition than at the source-state midpoint pseudo-boundary"
                ),
                scale="axis-score difference in signed differences",
                multiplicity_family="neural_transition_primary",
                seed=seed + index,
                n_bootstrap=n_bootstrap,
                n_sign_flips=n_sign_flips,
                code_version=code_version,
                support=support,
            )
        )

    axis_specs = (
        (
            "e1_nr_oof_auc",
            0.5,
            RESULT_IDS["oof_auc"],
            "e1_vs_nr_out_of_fold_auc_vs_chance",
            "A session-specific reward-state axis discriminates E1 from NR out of fold",
            "ROC AUC",
        ),
        (
            "e2_transfer_recovery",
            0.0,
            RESULT_IDS["e2_recovery"],
            "e2_transfer_minus_nr_axis_score",
            "E2 scores recover toward E1 on an axis never fit or tuned on E2",
            "axis-score difference",
        ),
    )
    for offset, (column, null_value, result_id, contrast_id, hypothesis, scale) in enumerate(
        axis_specs,
        start=2,
    ):
        if column == "e1_nr_oof_auc":
            n_trials = int(
                session_metrics.select(
                    (pl.col("n_engaged_1_trials") + pl.col("n_no_reward_trials")).sum()
                ).item()
            )
        else:
            n_trials = int(
                session_metrics.select(
                    (pl.col("n_engaged_2_trials") + pl.col("n_no_reward_trials")).sum()
                ).item()
            )
        support = {
            "n_mice": mouse_axis_metrics.height,
            "n_sessions": session_metrics.height,
            "n_units": int(session_metrics.get_column("n_units").sum()),
            "n_trials": n_trials,
        }
        rows.append(
            _statistic_row(
                mouse_axis_metrics,
                value_column=column,
                null_value=null_value,
                result_id=result_id,
                contrast_id=contrast_id,
                hypothesis=hypothesis,
                scale=scale,
                multiplicity_family="neural_axis_validation",
                seed=seed + offset,
                n_bootstrap=n_bootstrap,
                n_sign_flips=n_sign_flips,
                code_version=code_version,
                support=support,
            )
        )
    normalized = dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    return dg.statistics.add_holm_adjustment(normalized)


def _session_support_counts(
    supported_sessions: pl.DataFrame,
    session_metrics: pl.DataFrame,
    *,
    trial_count_columns: tuple[str, ...],
) -> dict[str, int]:
    keys = supported_sessions.select("_nwb_path").unique()
    sessions = session_metrics.join(keys, on="_nwb_path", how="inner", validate="1:1")
    return {
        "n_mice": supported_sessions.get_column("subject_id").n_unique(),
        "n_sessions": supported_sessions.height,
        "n_units": int(sessions.get_column("n_units").sum()),
        "n_trials": int(
            supported_sessions.select(pl.sum_horizontal(*trial_count_columns).sum()).item()
        ),
    }


def _statistic_row(
    mouse_values: pl.DataFrame,
    *,
    value_column: str,
    null_value: float,
    result_id: str,
    contrast_id: str,
    hypothesis: str,
    scale: str,
    multiplicity_family: str,
    seed: int,
    n_bootstrap: int,
    n_sign_flips: int,
    code_version: str,
    support: dict[str, int],
) -> dict[str, Any]:
    values = mouse_values.select(
        pl.col("subject_id").cast(pl.String),
        pl.col(value_column).cast(pl.Float64),
    )
    if values.height < 2:
        raise RuntimeError(f"{result_id} has fewer than two contributing mice")
    bootstrap = dg.statistics.bootstrap_mouse_mean(
        values,
        seed=seed,
        mouse_column="subject_id",
        value_column=value_column,
        n_resamples=n_bootstrap,
    )
    centered = values.with_columns((pl.col(value_column) - null_value).alias("centered"))
    test = dg.statistics.two_sided_sign_flip_test(
        centered,
        seed=seed,
        mouse_column="subject_id",
        value_column="centered",
        n_resamples=n_sign_flips,
    )
    return {
        "analysis_id": ANALYSIS_ID,
        "result_id": result_id,
        "contrast_id": contrast_id,
        "hypothesis": hypothesis,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "code_version": code_version,
        "seed": seed,
        "analysis_tier": ANALYSIS_TIER,
        "inclusion_definition": (
            "behavior-eligible discovery sessions; quality good, isi violations <0.5, "
            "amplitude cutoff <0.1; completed non-auto familiar full-contrast physical "
            "changes with no lick within +/-150 ms"
        ),
        "missingness_stratum": "estimable session-specific axis and declared trial support",
        "estimate": bootstrap.estimate,
        "scale": scale,
        "ci_low": bootstrap.ci_low,
        "ci_high": bootstrap.ci_high,
        "confidence_level": bootstrap.confidence_level,
        "ci_method": "mouse percentile bootstrap",
        "test_statistic": test.statistic,
        "test_method": test.method,
        "p_value": test.p_value,
        "adjusted_p_value": None,
        "adjustment_method": None,
        "multiplicity_family": multiplicity_family,
        "sidedness": "two-sided",
        "n_mice": support["n_mice"],
        "n_sessions": support["n_sessions"],
        "n_probes": None,
        "n_units": support["n_units"],
        "n_trials": support["n_trials"],
        "aggregation": "equal-session mean within mouse; equal-mouse mean",
        "bootstrap_id": f"mouse_percentile_{n_bootstrap}_seed_{seed}",
        "model_formula": (
            "ridge reward-state axis from sqrt-count change in [0,150) ms versus "
            "[-150,0) ms across simultaneously recorded units"
        ),
        "cv_grouping": (
            "nested contiguous-within-state folds within session; E2 excluded from all fitting"
        ),
        "status": "pass",
        "reason": "preliminary discovery vertical slice; D04 consistency filter not yet applied",
    }


def _analysis_signature(
    input_records: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "inputs": input_records,
        "sources": source_records,
        "config": dataclasses.asdict(dg.neural_transitions.DEFAULT_NEURAL_TRANSITION_CONFIG),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _read_session_cache(
    cache_root: pathlib.Path,
    *,
    source: str,
    analysis_signature: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("source") != source
            or manifest.get("analysis_signature")
            not in {analysis_signature, *COMPATIBLE_SESSION_CACHE_SIGNATURES}
        ):
            return None
        paths = {
            key: cache_root / name
            for key, name in {
                "unit_qc": "unit_qc.parquet",
                "trial_scores": "trial_scores.parquet",
                "session_metrics": "session_metrics.parquet",
            }.items()
        }
        records = manifest["outputs"]
        for key, path in paths.items():
            if not path.is_file() or _sha256(path) != records[key]["sha256"]:
                return None
        return tuple(pl.read_parquet(paths[key]) for key in paths)  # type: ignore[return-value]
    except (KeyError, OSError, json.JSONDecodeError, pl.exceptions.PolarsError):
        return None


def _write_session_cache(
    cache_root: pathlib.Path,
    *,
    source: str,
    analysis_signature: str,
    unit_qc: pl.DataFrame,
    trial_scores: pl.DataFrame,
    session_metrics: pl.DataFrame,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "unit_qc": dg.artifacts.write_frame(unit_qc, cache_root / "unit_qc.parquet"),
        "trial_scores": dg.artifacts.write_frame(
            trial_scores,
            cache_root / "trial_scores.parquet",
        ),
        "session_metrics": dg.artifacts.write_frame(
            session_metrics,
            cache_root / "session_metrics.parquet",
        ),
    }
    dg.artifacts.write_json(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source": source,
            "analysis_signature": analysis_signature,
            "outputs": {key: _file_record(path, base=cache_root) for key, path in paths.items()},
        },
        cache_root / "manifest.json",
    )


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(base.resolve())),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _sha256(path: pathlib.Path) -> str:
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
    return f"{revision}{'+dirty' if dirty else ''}"


if __name__ == "__main__":
    main()
