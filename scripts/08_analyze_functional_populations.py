# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Build the discovery-only functional-population outputs for Figure 3.

Only behavior-eligible discovery sessions are opened.  Spike trains are read
through lazynwb's obstore custom reader in 32-unit batches and immediately
reduced to event counts, model summaries, and per-unit PSTHs.  Running-speed
samples use the same custom HDF5 table reader.  Legacy accessors are fatal.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import gc
import hashlib
import json
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
import dg.data  # noqa: E402
import dg.functional_populations  # noqa: E402
import dg.lazynwb_obstore  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "functional_populations"
ANALYSIS_TIER = "discovery"
SPIKE_BATCH_SIZE = 32
CACHE_SCHEMA_VERSION = 1
COMPATIBLE_SESSION_CACHE_SIGNATURES = frozenset(
    {"7227166336d4cc8b2e3fd19381cfeb96ea48be789807d9839badc0c9061703a6"}
)
DEFAULT_SEED = 3051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
EXPECTED_DISCOVERY_MICE = 12
EXPECTED_DISCOVERY_SESSIONS = 23
RUNNING_PATH = "/processing/running/speed"
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/data.py",
    "src/dg/functional_populations.py",
    "src/dg/lazynwb_obstore.py",
    "src/dg/statistics.py",
    "scripts/08_analyze_functional_populations.py",
)
STATISTIC_SPECS = (
    (
        "late_minus_early_model_free",
        "late_minus_early_raw",
        "Late minus early reversible state modulation, model-free windows",
    ),
    (
        "late_minus_early_model_total",
        "late_minus_early_total",
        "Late minus early reversible state modulation, total model",
    ),
    (
        "late_minus_early_model_adjusted",
        "late_minus_early_adjusted",
        "Late minus early reversible state modulation after lick/running adjustment",
    ),
    (
        "late_state_model_total",
        "late_total",
        "Late-window reversible state modulation, total model",
    ),
    (
        "late_state_model_adjusted",
        "late_adjusted",
        "Late-window reversible state modulation after lick/running adjustment",
    ),
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--sign-flip-resamples", type=int, default=DEFAULT_SIGN_FLIP_RESAMPLES)
    parser.add_argument(
        "--recompute-session-cache",
        action="store_true",
        help="Ignore valid per-session reductions and reread bounded spike batches",
    )
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root.resolve())
    manifest_path = result_root / "manifests" / "functional_populations_analysis_run.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "run_status": "in_progress",
        "authoritative": False,
        "confirmation_accessed": False,
        "script": "scripts/08_analyze_functional_populations.py",
        "started_at_utc": started_at.isoformat(),
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


def _run_analysis(
    arguments: argparse.Namespace,
    *,
    result_root: pathlib.Path,
    manifest_path: pathlib.Path,
    started_at: datetime.datetime,
) -> None:
    lazynwb.config.anon = True
    lazynwb.config.use_obstore = True
    lazynwb.config.use_remfile = False
    tables = result_root / "tables"
    manifests = result_root / "manifests"
    cache_root = result_root / "cache" / "functional_population_sessions"
    profile_root = result_root / "all_units" / "functional_population_profiles"
    cache_root.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "behavior_trials": tables / "behavior_trials.parquet",
        "cohort_assignments": tables / "session_cohort_assignments.csv",
        "unit_qc": tables / "neural_transition_unit_qc.parquet",
        "analysis_lock": REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
        "figure_2_manifest": manifests / "neural_transition_analysis_run.json",
    }
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Figure 3 inputs: " + ", ".join(map(str, missing)))
    _validate_upstream_manifest(input_paths["figure_2_manifest"])
    input_records = {
        key: _file_record(
            path, base=result_root if path.is_relative_to(result_root) else REPOSITORY_ROOT
        )
        for key, path in input_paths.items()
    }
    source_records = {
        path: _file_record(REPOSITORY_ROOT / path, base=REPOSITORY_ROOT)
        for path in LOCAL_SOURCE_PATHS
    }
    signature = _analysis_signature(input_records, source_records)

    assignments = pl.read_csv(
        input_paths["cohort_assignments"],
        schema_overrides={"subject_id": pl.String},
    )
    sources = _select_discovery_sources(assignments)
    behavior = pl.read_parquet(input_paths["behavior_trials"])
    selected_trials = dg.functional_populations.select_functional_population_trials(
        behavior.join(sources.select("_nwb_path"), on="_nwb_path", how="inner", validate="m:1")
    )
    _validate_trial_support(selected_trials, sources)
    unit_qc = pl.read_parquet(input_paths["unit_qc"])
    units = _select_units(unit_qc, sources)

    unit_tables: list[pl.DataFrame] = []
    session_summary_tables: list[pl.DataFrame] = []
    running_qc_tables: list[pl.DataFrame] = []
    profile_paths: list[pathlib.Path] = []
    for session_index, source_row in enumerate(sources.iter_rows(named=True), start=1):
        source = str(source_row["_nwb_path"])
        session_id = int(source_row["ecephys_session_id"])
        session_trials = selected_trials.filter(pl.col("_nwb_path") == source).sort(
            "change_time", "_table_index"
        )
        session_units = units.filter(pl.col("_nwb_path") == source).sort("_table_index")
        cache = _session_cache_paths(cache_root, session_id)
        profile_path = profile_root / f"session_{session_id}.parquet"
        cached = None
        if not arguments.recompute_session_cache:
            cached = _read_session_cache(cache, signature=signature, profile_path=profile_path)
        if cached is None:
            print(
                f"[{session_index:02d}/{sources.height:02d}] {session_units.height} units; "
                f"streaming {SPIKE_BATCH_SIZE}/batch",
                flush=True,
            )
            session_unit_results, session_summary, running_qc = _analyze_session(
                source_row,
                session_trials,
                session_units,
                profile_path=profile_path,
            )
            _write_session_cache(
                cache,
                signature=signature,
                profile_path=profile_path,
                units=session_unit_results,
                summary=session_summary,
                running_qc=running_qc,
            )
        else:
            session_unit_results, session_summary, running_qc = cached
            print(f"[{session_index:02d}/{sources.height:02d}] validated cache", flush=True)
        unit_tables.append(session_unit_results)
        session_summary_tables.append(session_summary)
        running_qc_tables.append(running_qc)
        profile_paths.append(profile_path)
        gc.collect()

    unit_results = pl.concat(unit_tables, how="diagonal_relaxed").sort(
        "subject_id", "ecephys_session_id", "_table_index"
    )
    session_psths = pl.concat(session_summary_tables, how="diagonal_relaxed").sort(
        "aligned_event",
        "response_group",
        "reward_block",
        "ecephys_session_id",
        "bin_center_seconds",
    )
    running_qc = pl.concat(running_qc_tables, how="diagonal_relaxed").sort(
        "subject_id", "ecephys_session_id"
    )
    mouse_psths = _aggregate_mouse_psths(session_psths)
    session_effects = _aggregate_session_effects(unit_results, selected_trials)
    mouse_effects = _aggregate_mouse_effects(session_effects)
    class_stability = _summarize_class_stability(unit_results)
    statistics = _build_statistics(
        mouse_effects,
        session_effects=session_effects,
        seed=arguments.seed,
        n_bootstrap=arguments.bootstrap_resamples,
        n_sign_flips=arguments.sign_flip_resamples,
        code_version=_code_version(REPOSITORY_ROOT),
    )

    output_paths = {
        "functional_population_units": tables / "functional_population_units.parquet",
        "functional_population_session_psths": tables
        / "functional_population_session_psths.parquet",
        "functional_population_mouse_psths": tables / "functional_population_mouse_psths.csv",
        "functional_population_session_effects": tables
        / "functional_population_session_effects.csv",
        "functional_population_mouse_effects": tables / "functional_population_mouse_effects.csv",
        "functional_population_class_stability": tables
        / "functional_population_class_stability.csv",
        "functional_population_statistics": tables / "functional_population_statistics.csv",
        "functional_population_running_qc": tables / "functional_population_running_qc.csv",
        "functional_population_environment": manifests / "functional_population_environment.txt",
    }
    output_frames = {
        "functional_population_units": unit_results,
        "functional_population_session_psths": session_psths,
        "functional_population_mouse_psths": mouse_psths,
        "functional_population_session_effects": session_effects,
        "functional_population_mouse_effects": mouse_effects,
        "functional_population_class_stability": class_stability,
        "functional_population_statistics": statistics,
        "functional_population_running_qc": running_qc,
    }
    for key, frame in output_frames.items():
        dg.artifacts.write_frame(frame, output_paths[key])
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        output_paths["functional_population_environment"],
    )
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
        "counts": {
            "mice": sources.get_column("subject_id").n_unique(),
            "sessions": sources.height,
            "well_isolated_units": unit_results.height,
            "selected_trials": selected_trials.height,
            "stable_response_class_units": unit_results.filter("class_assignment_stable").height,
            "movement_adjusted_sessions": running_qc.filter(
                pl.col("movement_adjustment_available")
            ).height,
        },
        "unit_filter": (
            "isi_violations < 0.5; amplitude_cutoff < 0.1; quality == good; "
            "D04 engaged-block consistency pending, so this discovery result is isolation-only"
        ),
        "trial_filter": (
            "completed non-auto familiar full-contrast physical image changes with valid "
            "exact-deduplicated raw-lick response labels"
        ),
        "model": dataclasses.asdict(dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG),
        "data_access": {
            "library": "lazynwb",
            "version": "1.0.0.dev8",
            "transport": "obstore byte-range reads",
            "spike_reader": "dg.lazynwb_obstore bounded custom-reader batches",
            "running_reader": "lazynwb.scan_nwb custom reader at exact TimeSeries path",
            "spike_batch_size_units": SPIKE_BATCH_SIZE,
            "legacy_accessor_forbidden_at_runtime": True,
        },
        "inputs": input_records,
        "local_sources": source_records,
        "outputs": {
            key: _file_record(path, base=result_root) for key, path in output_paths.items()
        },
        "all_unit_profile_partitions": [
            _file_record(path, base=result_root) for path in profile_paths
        ],
    }
    dg.artifacts.write_json(manifest, manifest_path)
    print(
        f"Completed Figure 3 analysis: {unit_results.height} units, "
        f"{selected_trials.height} trials, {sources.height} sessions",
        flush=True,
    )


def _analyze_session(
    source_row: dict[str, Any],
    trials: pl.DataFrame,
    units: pl.DataFrame,
    *,
    profile_path: pathlib.Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = str(source_row["_nwb_path"])
    unit_batches = []
    profile_batches = []
    with _forbid_legacy_accessors():
        running = _read_running(source)
        prepared_running = dg.functional_populations.prepare_running_samples(
            running.get_column("timestamps").to_numpy(),
            running.get_column("data").to_numpy(),
        )
        if prepared_running.usable:
            running_covariates = dg.functional_populations.summarize_running_at_events(
                prepared_running.timestamps,
                prepared_running.speed,
                trials.get_column("change_time").to_numpy(),
            )
        else:
            running_covariates = pl.DataFrame(
                {
                    column: [float("nan")] * trials.height
                    for column in (
                        "running_baseline",
                        "running_early",
                        "running_late",
                        "running_early_delta",
                        "running_late_delta",
                    )
                }
            )
        trials = pl.concat([trials, running_covariates], how="horizontal")
        running_complete = np.ones(trials.height, dtype=bool)
        for column in (
            "running_baseline",
            "running_early",
            "running_early_delta",
            "running_late",
            "running_late_delta",
        ):
            running_complete &= np.isfinite(trials.get_column(column).to_numpy())
        late_eligible = trials.get_column("prelick_late_eligible").to_numpy()
        model_masks: dict[tuple[str, str], np.ndarray] = {
            ("total", "early"): np.ones(trials.height, dtype=bool),
            ("total", "late"): late_eligible,
        }
        for key, mask in model_masks.items():
            if not _has_model_support(trials, mask):
                raise RuntimeError(f"{'_'.join(key)} has insufficient per-block trial support")
        adjusted_masks = {
            ("adjusted", "early"): running_complete,
            ("adjusted", "late"): running_complete & late_eligible,
        }
        movement_adjustment_available = prepared_running.usable and all(
            _has_model_support(trials, mask) for mask in adjusted_masks.values()
        )
        if movement_adjustment_available:
            model_masks.update(adjusted_masks)
        config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
        if (
            trials.filter(pl.col("response_in_window")).height
            < config.minimum_response_trials_for_action
        ):
            raise RuntimeError("session has too few responded trials for action alignment")
        model_designs = {
            key: dg.functional_populations.build_model_design(
                trials.filter(pl.Series(mask)),
                window=key[1],
                adjusted=key[0] == "adjusted",
            )
            for key, mask in model_masks.items()
        }
        running_qc = _running_qc_frame(
            source_row,
            prepared_running,
            trials=trials,
            running_complete=running_complete,
            movement_adjustment_available=movement_adjustment_available,
        )

        selected_indices = units.get_column("_table_index").to_list()
        next_unit = 0

        def consume(batch_rows: tuple[int, ...], spike_rows: list[list[Any]]) -> None:
            nonlocal next_unit, profile_batches, unit_batches
            expected = tuple(selected_indices[next_unit : next_unit + len(batch_rows)])
            if batch_rows != expected:
                raise RuntimeError("obstore spike batch order differs from unit table")
            metadata = units.slice(next_unit, len(batch_rows))
            batch_units, batch_profiles = _reduce_spike_batch(
                metadata,
                trials,
                spike_rows,
                model_masks=model_masks,
                model_designs=model_designs,
                movement_adjustment_available=movement_adjustment_available,
            )
            unit_batches.append(batch_units)
            profile_batches.append(batch_profiles)
            next_unit += len(batch_rows)
            if next_unit == len(selected_indices) or next_unit % (SPIKE_BATCH_SIZE * 10) == 0:
                print(
                    f"    reduced {next_unit}/{len(selected_indices)} units",
                    flush=True,
                )

        dg.lazynwb_obstore.map_indexed_numeric_column_batches(
            source,
            "/units",
            "spike_times",
            selected_indices,
            consume,
            batch_size=SPIKE_BATCH_SIZE,
        )
    if next_unit != len(selected_indices):
        raise RuntimeError("bounded spike reader did not cover every selected unit")
    unit_results = pl.concat(unit_batches, how="diagonal_relaxed")
    profiles = pl.concat(profile_batches, how="diagonal_relaxed")
    dg.artifacts.write_frame(profiles, profile_path)
    session_summary = _summarize_session_profiles(profiles)
    del profiles, profile_batches, unit_batches
    return unit_results, session_summary, running_qc


def _reduce_spike_batch(
    metadata: pl.DataFrame,
    trials: pl.DataFrame,
    spike_rows: list[list[Any]],
    *,
    model_masks: dict[tuple[str, str], np.ndarray],
    model_designs: dict[tuple[str, str], tuple[np.ndarray, tuple[str, ...]]],
    movement_adjustment_available: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    changes = trials.get_column("change_time").to_numpy()
    licks = trials.get_column("first_response_lick_time").to_numpy()
    binned = dg.functional_populations.bin_spike_batch(spike_rows, changes, licks)
    early = dg.functional_populations.variance_stabilized_window_response(
        binned.change_counts,
        binned.change_bin_centers,
        config.early_window,
        config.baseline_window,
        bin_width_seconds=config.bin_width_seconds,
    )
    late_prelick = dg.functional_populations.censored_variance_stabilized_window_response(
        binned.change_counts,
        binned.change_bin_centers,
        trials.get_column("late_censor_stop_seconds").to_numpy(),
        config.late_window,
        config.baseline_window,
        bin_width_seconds=config.bin_width_seconds,
    )
    responded = trials.get_column("response_in_window").to_numpy()
    action = dg.functional_populations.variance_stabilized_window_response(
        binned.lick_counts,
        binned.lick_bin_centers,
        config.prelick_window,
        config.lick_baseline_window,
        bin_width_seconds=config.bin_width_seconds,
    )
    classes = dg.functional_populations.classify_response_archetypes(
        early,
        late_prelick,
        action,
        trials.get_column("assignment_half").to_numpy(),
        responded,
    )
    blocks = trials.get_column("reward_block").to_numpy()
    early_sign = np.sign(np.nanmean(early, axis=0))
    late_sign = np.sign(np.nanmean(late_prelick, axis=0))
    early_sign[early_sign == 0] = np.nan
    late_sign[late_sign == 0] = np.nan
    early_raw = dg.functional_populations.reversible_contrast(early, blocks) * early_sign
    late_raw = dg.functional_populations.reversible_contrast(late_prelick, blocks) * late_sign

    model_columns: dict[str, Any] = {
        "early_response_sign": early_sign,
        "late_response_sign": late_sign,
        "early_raw_reversible": early_raw,
        "late_prelick_raw_reversible": late_raw,
    }
    for window in ("early", "late"):
        for suffix in (
            "reversible",
            "oof_r2",
            "state_dropout_r2",
            "ridge_lambda",
        ):
            model_columns[f"{window}_adjusted_{suffix}"] = np.full(metadata.height, np.nan)
    outcomes = {"early": early, "late": late_prelick}
    for model_name in ("total", "adjusted"):
        for window, outcome in outcomes.items():
            model_key = (model_name, window)
            if model_key not in model_designs:
                continue
            mask = model_masks[model_key]
            design, names = model_designs[model_key]
            selected_outcome = outcome[mask]
            selected_blocks = blocks[mask]
            selected_times = changes[mask]
            fit = dg.functional_populations.nested_contiguous_ridge(
                design,
                selected_outcome,
                selected_blocks,
                selected_times,
            )
            state_index = names.index("reward_reversible")
            aligned_sign = early_sign if window == "early" else late_sign
            state_effect = 1.5 * fit.coefficients[state_index] * aligned_sign
            reduced = dg.functional_populations.nested_contiguous_ridge(
                np.delete(design, state_index, axis=1),
                selected_outcome,
                selected_blocks,
                selected_times,
            )
            model_columns[f"{window}_{model_name}_reversible"] = state_effect
            model_columns[f"{window}_{model_name}_oof_r2"] = fit.oof_r2
            model_columns[f"{window}_{model_name}_state_dropout_r2"] = fit.oof_r2 - reduced.oof_r2
            model_columns[f"{window}_{model_name}_ridge_lambda"] = fit.selected_lambdas

    unit_results = pl.concat(
        [
            metadata.select(
                "_nwb_path",
                "subject_id",
                "ecephys_session_id",
                "_table_index",
                "id",
                "isi_violations",
                "amplitude_cutoff",
                "quality",
                "well_isolated",
                "engaged_block_consistency_status",
            ),
            classes,
            pl.DataFrame(model_columns),
        ],
        how="horizontal",
    ).with_columns(
        pl.when(pl.col("response_class") == "early_sensory")
        .then(pl.lit("early_sensory"))
        .when(pl.col("response_class").is_in(("late_prelick", "action_related")))
        .then(pl.lit("late_action"))
        .otherwise(pl.lit("continuous_only"))
        .alias("response_group"),
        pl.lit("isolation_only_d04_pending").alias("functional_population_unit_filter"),
        pl.lit("available" if movement_adjustment_available else "unavailable_running_clock").alias(
            "movement_adjustment_status"
        ),
    )
    profiles = _build_unit_profiles(
        metadata,
        unit_results,
        trials,
        binned,
    )
    return unit_results, profiles


def _build_unit_profiles(
    metadata: pl.DataFrame,
    units: pl.DataFrame,
    trials: pl.DataFrame,
    binned: dg.functional_populations.BinnedSpikeBatch,
) -> pl.DataFrame:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    identifiers = units.select(
        "_nwb_path",
        "subject_id",
        "ecephys_session_id",
        "_table_index",
        "id",
        "response_class",
        "response_group",
        "class_assignment_stable",
        "dominant_response_sign",
    )
    del metadata
    blocks = trials.get_column("reward_block").to_numpy()
    responded = trials.get_column("response_in_window").to_numpy()
    rows: list[pl.DataFrame] = []
    for aligned_event, counts, centers, event_blocks in (
        ("change", binned.change_counts, binned.change_bin_centers, blocks),
        ("lick", binned.lick_counts, binned.lick_bin_centers, blocks[responded]),
    ):
        baseline_window = (
            config.baseline_window if aligned_event == "change" else config.lick_baseline_window
        )
        baseline_mask = (centers >= baseline_window[0]) & (centers < baseline_window[1])
        for block in dg.functional_populations.REWARD_BLOCKS:
            selected = counts[event_blocks == block]
            if not selected.shape[0]:
                continue
            mean_rate = selected.mean(axis=0) / config.bin_width_seconds
            baseline = mean_rate[baseline_mask].mean(axis=0)
            evoked = mean_rate - baseline[np.newaxis, :]
            long = (
                pl.DataFrame(
                    {
                        "_table_index": np.tile(
                            identifiers.get_column("_table_index").to_numpy(), centers.size
                        ),
                        "bin_center_seconds": np.repeat(centers, identifiers.height),
                        "evoked_rate_hz": evoked.reshape(-1),
                        "n_trials": np.repeat(selected.shape[0], evoked.size),
                    }
                )
                .join(
                    identifiers,
                    on="_table_index",
                    how="left",
                    validate="m:1",
                )
                .with_columns(
                    pl.lit(aligned_event).alias("aligned_event"),
                    pl.lit(block).alias("reward_block"),
                    (pl.col("evoked_rate_hz") * pl.col("dominant_response_sign")).alias(
                        "sign_aligned_evoked_rate_hz"
                    ),
                )
            )
            rows.append(long)
    return pl.concat(rows, how="diagonal_relaxed").sort(
        "aligned_event", "response_group", "reward_block", "_table_index", "bin_center_seconds"
    )


def _summarize_session_profiles(profiles: pl.DataFrame) -> pl.DataFrame:
    stable = profiles.filter(
        pl.col("class_assignment_stable")
        & pl.col("response_group").is_in(("early_sensory", "late_action"))
    )
    if stable.is_empty():
        raise RuntimeError("session has no cross-half-stable functional response units")
    return (
        stable.group_by(
            "_nwb_path",
            "subject_id",
            "ecephys_session_id",
            "aligned_event",
            "response_group",
            "reward_block",
            "bin_center_seconds",
        )
        .agg(
            pl.col("sign_aligned_evoked_rate_hz").mean().alias("mean_sign_aligned_rate_hz"),
            pl.col("_table_index").n_unique().alias("n_units"),
            pl.col("n_trials").first(),
        )
        .sort("aligned_event", "response_group", "reward_block", "bin_center_seconds")
    )


def _read_running(source: str) -> pl.DataFrame:
    frame = (
        lazynwb.scan_nwb(
            source,
            RUNNING_PATH,
            raise_on_missing=True,
            disable_progress=True,
        )
        .select("data", "timestamps")
        .collect()
    )
    if frame.is_empty() or set(frame.columns) != {"data", "timestamps"}:
        raise RuntimeError("running-speed custom read returned an invalid table")
    return frame.select(
        pl.col("data").cast(pl.Float64),
        pl.col("timestamps").cast(pl.Float64),
    )


def _aggregate_mouse_psths(session_psths: pl.DataFrame) -> pl.DataFrame:
    return (
        session_psths.group_by(
            "subject_id", "aligned_event", "response_group", "reward_block", "bin_center_seconds"
        )
        .agg(
            pl.col("mean_sign_aligned_rate_hz").mean().alias("mean_sign_aligned_rate_hz"),
            pl.col("ecephys_session_id").n_unique().alias("n_sessions"),
            pl.col("n_units").sum().alias("n_unit_session_contributions"),
            pl.col("n_trials").sum().alias("n_trial_session_contributions"),
        )
        .sort("aligned_event", "response_group", "reward_block", "subject_id", "bin_center_seconds")
    )


def _aggregate_session_effects(
    units: pl.DataFrame,
    trials: pl.DataFrame,
) -> pl.DataFrame:
    def finite_mean(column: str) -> pl.Expr:
        return pl.col(column).filter(pl.col(column).is_finite()).mean().alias(column)

    effects = units.group_by("_nwb_path", "subject_id", "ecephys_session_id").agg(
        pl.len().alias("n_units"),
        pl.col("late_adjusted_reversible").is_finite().sum().alias("n_adjusted_units"),
        pl.col("class_assignment_stable").sum().alias("n_stable_class_units"),
        pl.col("early_raw_reversible").mean().alias("early_raw"),
        pl.col("late_prelick_raw_reversible").mean().alias("late_raw"),
        pl.col("early_total_reversible").mean().alias("early_total"),
        pl.col("late_total_reversible").mean().alias("late_total"),
        pl.col("early_adjusted_reversible").mean().alias("early_adjusted"),
        pl.col("late_adjusted_reversible").mean().alias("late_adjusted"),
        finite_mean("early_total_state_dropout_r2"),
        finite_mean("late_total_state_dropout_r2"),
        finite_mean("early_adjusted_state_dropout_r2"),
        finite_mean("late_adjusted_state_dropout_r2"),
    )
    support = trials.group_by("_nwb_path").agg(
        pl.len().alias("n_trials"),
        (~pl.col("response_in_window")).sum().alias("n_nonresponse_trials"),
        pl.col("response_in_window").sum().alias("n_response_trials"),
    )
    return (
        effects.join(support, on="_nwb_path", how="left", validate="1:1")
        .with_columns(
            (pl.col("late_raw") - pl.col("early_raw")).alias("late_minus_early_raw"),
            (pl.col("late_total") - pl.col("early_total")).alias("late_minus_early_total"),
            (pl.col("late_adjusted") - pl.col("early_adjusted")).alias("late_minus_early_adjusted"),
            (pl.col("late_adjusted") - pl.col("late_total")).alias("late_adjustment_change"),
        )
        .sort("subject_id", "ecephys_session_id")
    )


def _aggregate_mouse_effects(session_effects: pl.DataFrame) -> pl.DataFrame:
    metrics = [
        "early_raw",
        "late_raw",
        "late_minus_early_raw",
        "early_total",
        "late_total",
        "late_minus_early_total",
        "early_adjusted",
        "late_adjusted",
        "late_minus_early_adjusted",
        "late_adjustment_change",
        "early_total_state_dropout_r2",
        "late_total_state_dropout_r2",
        "early_adjusted_state_dropout_r2",
        "late_adjusted_state_dropout_r2",
    ]
    return (
        session_effects.group_by("subject_id")
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("late_adjusted").is_not_null().sum().alias("n_adjusted_sessions"),
            pl.col("n_units").sum(),
            pl.col("n_adjusted_units").sum(),
            pl.col("n_trials").sum(),
            *(pl.col(column).mean().alias(column) for column in metrics),
        )
        .sort("subject_id")
    )


def _summarize_class_stability(units: pl.DataFrame) -> pl.DataFrame:
    return (
        units.group_by("subject_id")
        .agg(
            pl.col("ecephys_session_id").n_unique().alias("n_sessions"),
            pl.len().alias("n_units"),
            pl.col("class_assignment_stable").sum().alias("n_stable_units"),
            pl.col("response_class").eq("early_sensory").sum().alias("n_early_sensory"),
            pl.col("response_class").eq("late_prelick").sum().alias("n_late_prelick"),
            pl.col("response_class").eq("action_related").sum().alias("n_action_related"),
            pl.col("assignment_margin").mean().alias("mean_assignment_margin"),
        )
        .with_columns(
            (pl.col("n_stable_units") / pl.col("n_units")).alias("stable_assignment_fraction")
        )
        .sort("subject_id")
    )


def _build_statistics(
    mouse_effects: pl.DataFrame,
    *,
    session_effects: pl.DataFrame,
    seed: int,
    n_bootstrap: int,
    n_sign_flips: int,
    code_version: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (result_id, value_column, hypothesis) in enumerate(STATISTIC_SPECS):
        adjusted_result = "adjusted" in value_column
        values = mouse_effects.filter(
            pl.col(value_column).is_not_null() & pl.col(value_column).is_finite()
        ).select("subject_id", value_column)
        supported_sessions = (
            session_effects.filter(
                pl.col(value_column).is_not_null() & pl.col(value_column).is_finite()
            )
            if value_column in session_effects.columns
            else session_effects
        )
        base_row = {
            "analysis_id": ANALYSIS_ID,
            "result_id": result_id,
            "contrast_id": value_column,
            "hypothesis": hypothesis,
            "dandiset_version": dg.data.DANDISET_VERSION,
            "code_version": code_version,
            "seed": seed + index,
            "analysis_tier": ANALYSIS_TIER,
            "inclusion_definition": (
                "behavior-eligible discovery sessions; isolation-only units; familiar "
                "full-contrast physical changes; D04 pending"
            ),
            "missingness_stratum": (
                "running-complete trial windows and raw-lick labels"
                if adjusted_result
                else "raw-lick labels; running not required"
            ),
            "scale": "sign-aligned Anscombe-rate units",
            "adjusted_p_value": None,
            "adjustment_method": None,
            "multiplicity_family": "figure_3_primary_functional_timing",
            "sidedness": "two-sided",
            "n_mice": values.height,
            "n_sessions": supported_sessions.height,
            "n_probes": None,
            "n_units": int(
                supported_sessions.get_column(
                    "n_adjusted_units" if adjusted_result else "n_units"
                ).sum()
            ),
            "n_trials": int(supported_sessions.get_column("n_trials").sum()),
            "aggregation": "mean units within session, sessions within mouse, equal mice",
            "model_formula": (
                "reward_reversible + engaged_drift + smooth_time + image; adjusted adds "
                "lick_response + lick_latency + baseline/window running"
            ),
            "cv_grouping": "nested contiguous folds within reward state and session",
        }
        if values.height < 2:
            rows.append(
                {
                    **base_row,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "confidence_level": None,
                    "ci_method": None,
                    "test_statistic": None,
                    "test_method": None,
                    "p_value": None,
                    "bootstrap_id": None,
                    "status": "not_estimable",
                    "reason": "fewer than two mice have an estimable movement-adjusted effect",
                }
            )
            continue
        bootstrap = dg.statistics.bootstrap_mouse_mean(
            values,
            seed=seed + index,
            mouse_column="subject_id",
            value_column=value_column,
            n_resamples=n_bootstrap,
        )
        test = dg.statistics.two_sided_sign_flip_test(
            values,
            seed=seed + 100 + index,
            mouse_column="subject_id",
            value_column=value_column,
            n_resamples=n_sign_flips,
        )
        rows.append(
            {
                **base_row,
                "estimate": bootstrap.estimate,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "confidence_level": bootstrap.confidence_level,
                "ci_method": f"mouse percentile bootstrap ({n_bootstrap} resamples)",
                "test_statistic": test.statistic,
                "test_method": test.method,
                "p_value": test.p_value,
                "bootstrap_id": f"mouse_bootstrap_seed_{seed + index}",
                "status": "pass",
                "reason": None,
            }
        )
    statistics = dg.statistics.add_holm_adjustment(
        dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    )
    return statistics.with_columns(
        pl.when(pl.col("p_value").is_null())
        .then(pl.col("status"))
        .when(pl.col("adjusted_p_value") < 0.05)
        .then(pl.lit("pass"))
        .otherwise(pl.lit("null"))
        .alias("status")
    )


def _select_discovery_sources(assignments: pl.DataFrame) -> pl.DataFrame:
    required = {
        "_nwb_path",
        "subject_id",
        "cohort_assignment",
        "session_behavior_eligible",
        "neural_activity_used_for_allocation",
        "ecephys_session_id",
    }
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"cohort assignments missing columns: {sorted(missing)}")
    if assignments.filter(pl.col("neural_activity_used_for_allocation").fill_null(True)).height:
        raise RuntimeError("discovery allocation used neural activity")
    selected = assignments.filter(
        (pl.col("cohort_assignment") == "discovery")
        & pl.col("session_behavior_eligible").fill_null(False)
    ).select("_nwb_path", pl.col("subject_id").cast(pl.String), "ecephys_session_id")
    if selected.height != EXPECTED_DISCOVERY_SESSIONS:
        raise RuntimeError(f"expected {EXPECTED_DISCOVERY_SESSIONS} discovery sessions")
    if selected.get_column("subject_id").n_unique() != EXPECTED_DISCOVERY_MICE:
        raise RuntimeError(f"expected {EXPECTED_DISCOVERY_MICE} discovery mice")
    return selected.sort("subject_id", "ecephys_session_id")


def _select_units(unit_qc: pl.DataFrame, sources: pl.DataFrame) -> pl.DataFrame:
    required = {
        "_nwb_path",
        "subject_id",
        "ecephys_session_id",
        "_table_index",
        "well_isolated",
        "engaged_block_consistency_status",
    }
    missing = required.difference(unit_qc.columns)
    if missing:
        raise ValueError(f"unit QC missing columns: {sorted(missing)}")
    selected = unit_qc.join(
        sources.select("_nwb_path"), on="_nwb_path", how="inner", validate="m:1"
    ).filter(pl.col("well_isolated"))
    if selected.is_empty():
        raise RuntimeError("no well-isolated discovery units")
    if set(selected.get_column("_nwb_path")) != set(sources.get_column("_nwb_path")):
        raise RuntimeError("well-isolated unit table lacks a discovery session")
    return selected


def _validate_trial_support(trials: pl.DataFrame, sources: pl.DataFrame) -> None:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    coverage = trials.group_by("_nwb_path", "reward_block").agg(
        pl.len().alias("n_trials"),
        pl.col("prelick_late_eligible").sum().alias("n_prelick_eligible"),
    )
    if coverage.get_column("_nwb_path").n_unique() != sources.height:
        raise RuntimeError("functional trials do not cover every discovery session")
    if coverage.height != sources.height * len(dg.functional_populations.REWARD_BLOCKS):
        raise RuntimeError("functional trials do not cover all three reward blocks")
    if coverage.filter(pl.col("n_trials") < config.minimum_trials_per_block).height:
        raise RuntimeError("a session/block lacks functional-analysis trial support")
    if coverage.filter(
        pl.col("n_prelick_eligible") < config.minimum_prelick_trials_per_block
    ).height:
        raise RuntimeError("a session/block lacks pre-lick trials for the late window")


def _has_model_support(
    trials: pl.DataFrame,
    valid: np.ndarray,
) -> bool:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    selected = trials.filter(pl.Series(valid))
    counts = selected.group_by("reward_block").len()
    if counts.height != len(dg.functional_populations.REWARD_BLOCKS):
        return False
    return not counts.filter(pl.col("len") < config.minimum_trials_per_block).height


def _running_qc_frame(
    source_row: dict[str, Any],
    prepared: dg.functional_populations.PreparedRunningSamples,
    *,
    trials: pl.DataFrame,
    running_complete: np.ndarray,
    movement_adjustment_available: bool,
) -> pl.DataFrame:
    if movement_adjustment_available:
        adjustment_status = "available"
        reason = ""
    elif not prepared.usable:
        adjustment_status = "unavailable"
        reason = "fewer_than_two_unique_finite_running_timestamps"
    else:
        adjustment_status = "unavailable"
        reason = "insufficient_running_complete_trials_per_reward_block"
    return pl.DataFrame(
        {
            "_nwb_path": [str(source_row["_nwb_path"])],
            "subject_id": [str(source_row["subject_id"])],
            "ecephys_session_id": [int(source_row["ecephys_session_id"])],
            "n_raw_running_rows": [prepared.n_raw_rows],
            "n_finite_running_pairs": [prepared.n_finite_pairs],
            "n_unique_running_timestamps": [prepared.n_unique_timestamps],
            "n_nonincreasing_adjacent_raw": [prepared.n_nonincreasing_adjacent_raw],
            "n_decreasing_adjacent_raw": [prepared.n_decreasing_adjacent_raw],
            "n_duplicate_adjacent_raw": [prepared.n_duplicate_adjacent_raw],
            "n_exact_duplicates_removed": [prepared.n_exact_duplicates_removed],
            "running_timestamp_min": [prepared.timestamp_min],
            "running_timestamp_max": [prepared.timestamp_max],
            "running_clock_status": [prepared.status],
            "running_clock_repair": [
                "stable_sort_then_mean_exact_duplicate_speeds"
                if prepared.status == "pass_sorted_exact_deduplicated_finite_pairs"
                else "none"
            ],
            "n_selected_trials": [trials.height],
            "n_running_complete_trials": [int(running_complete.sum())],
            "movement_adjustment_available": [movement_adjustment_available],
            "movement_adjustment_status": [adjustment_status],
            "movement_adjustment_unavailable_reason": [reason],
        }
    )


def _validate_upstream_manifest(path: pathlib.Path) -> None:
    record = json.loads(path.read_text())
    if record.get("run_status") != "complete" or record.get("analysis_tier") != "discovery":
        raise RuntimeError("Figure 2 unit-QC provenance is not a complete discovery run")
    if record.get("confirmation_accessed") is not False:
        raise RuntimeError("upstream neural manifest does not keep confirmation sealed")


@contextlib.contextmanager
def _forbid_legacy_accessors():
    original_get_accessor = lazynwb.file_io._get_accessor
    original_open_hdf5 = getattr(lazynwb.file_io, "_open_hdf5", None)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("legacy NWB accessor invoked; Figure 3 requires custom obstore reads")

    lazynwb.file_io._get_accessor = forbidden
    if original_open_hdf5 is not None:
        lazynwb.file_io._open_hdf5 = forbidden
    try:
        yield
    finally:
        lazynwb.file_io._get_accessor = original_get_accessor
        if original_open_hdf5 is not None:
            lazynwb.file_io._open_hdf5 = original_open_hdf5


def _session_cache_paths(root: pathlib.Path, session_id: int) -> dict[str, pathlib.Path]:
    session_root = root / str(session_id)
    return {
        "root": session_root,
        "units": session_root / "units.parquet",
        "summary": session_root / "session_psths.parquet",
        "running_qc": session_root / "running_qc.csv",
        "manifest": session_root / "manifest.json",
    }


def _read_session_cache(
    paths: dict[str, pathlib.Path],
    *,
    signature: str,
    profile_path: pathlib.Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    required = (paths["units"], paths["summary"], paths["manifest"], profile_path)
    if not all(path.is_file() for path in required):
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text())
        if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            return None
        cached_signature = manifest.get("analysis_signature")
        if (
            cached_signature != signature
            and cached_signature not in COMPATIBLE_SESSION_CACHE_SIGNATURES
        ):
            return None
        for key, path in (
            ("units", paths["units"]),
            ("summary", paths["summary"]),
            ("profiles", profile_path),
        ):
            if manifest["files"][key]["sha256"] != _sha256(path):
                return None
        units = pl.read_parquet(paths["units"])
        summary = pl.read_parquet(paths["summary"])
        if paths["running_qc"].is_file() and "running_qc" in manifest["files"]:
            if manifest["files"]["running_qc"]["sha256"] != _sha256(paths["running_qc"]):
                return None
            running_qc = pl.read_csv(paths["running_qc"])
        else:
            running_qc = _legacy_cached_running_qc(units)
        return units, summary, running_qc
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def _write_session_cache(
    paths: dict[str, pathlib.Path],
    *,
    signature: str,
    profile_path: pathlib.Path,
    units: pl.DataFrame,
    summary: pl.DataFrame,
    running_qc: pl.DataFrame,
) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    dg.artifacts.write_frame(units, paths["units"])
    dg.artifacts.write_frame(summary, paths["summary"])
    dg.artifacts.write_frame(running_qc, paths["running_qc"])
    dg.artifacts.write_json(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "analysis_signature": signature,
            "files": {
                "units": _file_record(paths["units"], base=paths["root"]),
                "summary": _file_record(paths["summary"], base=paths["root"]),
                "running_qc": _file_record(paths["running_qc"], base=paths["root"]),
                "profiles": _file_record(profile_path, base=profile_path.parent.parent.parent),
            },
        },
        paths["manifest"],
    )


def _legacy_cached_running_qc(units: pl.DataFrame) -> pl.DataFrame:
    identity = units.select("_nwb_path", "subject_id", "ecephys_session_id").row(0, named=True)
    return pl.DataFrame(
        {
            "_nwb_path": [identity["_nwb_path"]],
            "subject_id": [str(identity["subject_id"])],
            "ecephys_session_id": [int(identity["ecephys_session_id"])],
            "n_raw_running_rows": [None],
            "n_finite_running_pairs": [None],
            "n_unique_running_timestamps": [None],
            "n_nonincreasing_adjacent_raw": [0],
            "n_decreasing_adjacent_raw": [0],
            "n_duplicate_adjacent_raw": [0],
            "n_exact_duplicates_removed": [0],
            "running_timestamp_min": [None],
            "running_timestamp_max": [None],
            "running_clock_status": ["pass_strict_validation_from_compatible_cache"],
            "running_clock_repair": ["none; raw counts unavailable without reopening NWB"],
            "n_selected_trials": [None],
            "n_running_complete_trials": [None],
            "movement_adjustment_available": [True],
            "movement_adjustment_status": ["available"],
            "movement_adjustment_unavailable_reason": [""],
        }
    )


def _analysis_signature(inputs: dict[str, Any], sources: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "inputs": inputs,
            "sources": sources,
            "config": dataclasses.asdict(
                dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
            ),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_version(repository: pathlib.Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}{'-dirty' if dirty else ''}"


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")
    if arguments.bootstrap_resamples < 1 or arguments.sign_flip_resamples < 1:
        raise ValueError("resample counts must be positive")


if __name__ == "__main__":
    main()
