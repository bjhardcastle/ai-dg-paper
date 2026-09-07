# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Build discovery-only simultaneous inter-regional outputs for Figure 5.

Ragged spike times are read from one session at a time through lazynwb's
obstore custom reader, in 32-unit batches.  Each batch is immediately reduced
to compact trial-by-unit windows.  Confirmation identities and neural data are
never loaded.
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

import lazynwb.file_io
import numpy as np
import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.artifacts  # noqa: E402
import dg.data  # noqa: E402
import dg.lazynwb_obstore  # noqa: E402
import dg.network_interactions  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "simultaneous_network_interaction"
ANALYSIS_TIER = "discovery"
DEFAULT_SEED = 1051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
CACHE_SCHEMA_VERSION = 1
SPIKE_BATCH_SIZE = 32
EXPECTED_DISCOVERY_MICE = 12
EXPECTED_DISCOVERY_SESSIONS = 23
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/data.py",
    "src/dg/lazynwb_obstore.py",
    "src/dg/network_interactions.py",
    "src/dg/statistics.py",
    "scripts/12_analyze_network.py",
)
REGION_COLUMN_PRIORITY = (
    "network_region",
    "analysis_region",
    "parent_region_acronym",
    "parent_structure_acronym",
    "parent_region",
    "major_division_acronym",
    "structure_acronym",
    "structure_layer",
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
    parser.add_argument("--recompute-session-cache", action="store_true")
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = result_root / "manifests" / "network_analysis_run.json"
    started_at = datetime.datetime.now(datetime.UTC)
    marker = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "run_status": "in_progress",
        "authoritative": False,
        "script": "scripts/12_analyze_network.py",
        "started_at_utc": started_at.isoformat(),
        "confirmation_accessed": False,
    }
    dg.artifacts.write_json(marker, manifest_path)
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
                    **marker,
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
    tables = result_root / "tables"
    manifests = result_root / "manifests"
    input_paths = {
        "behavior_trials": tables / "behavior_trials.parquet",
        "discovery_session_sources": tables / "discovery_session_sources.parquet",
        "neural_transition_unit_qc": tables / "neural_transition_unit_qc.parquet",
        "neural_transition_manifest": manifests / "neural_transition_analysis_run.json",
        "analysis_lock": REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
    }
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Figure 5 inputs: " + ", ".join(map(str, missing)))
    _validate_discovery_manifest(input_paths["neural_transition_manifest"])

    sources = pl.read_parquet(input_paths["discovery_session_sources"]).with_columns(
        pl.col("subject_id").cast(pl.String)
    )
    _validate_discovery_sources(sources)
    source_keys = sources.select("_nwb_path")
    behavior = pl.read_parquet(input_paths["behavior_trials"]).join(
        source_keys, on="_nwb_path", how="inner", validate="m:1"
    )
    network_trials = dg.network_interactions.select_network_trials(behavior)
    unit_qc = (
        pl.read_parquet(input_paths["neural_transition_unit_qc"])
        .join(source_keys, on="_nwb_path", how="inner", validate="m:1")
        .with_columns(pl.col("subject_id").cast(pl.String))
    )
    if set(unit_qc.get_column("_nwb_path")) != set(source_keys.get_column("_nwb_path")):
        raise RuntimeError("unit QC does not cover all and only discovery sources")

    anatomy_path = tables / "neural_unit_anatomy.parquet"
    input_records = {
        key: _file_record(
            path,
            base=result_root if path.is_relative_to(result_root) else REPOSITORY_ROOT,
        )
        for key, path in input_paths.items()
    }
    if anatomy_path.is_file():
        input_records["neural_unit_anatomy"] = _file_record(anatomy_path, base=result_root)
    source_records = {
        path: _file_record(REPOSITORY_ROOT / path, base=REPOSITORY_ROOT)
        for path in LOCAL_SOURCE_PATHS
    }
    signature = _analysis_signature(input_records, source_records)

    print("Figure 5: resolving discovery anatomy one session at a time", flush=True)
    unit_anatomy, anatomy_source = _load_or_read_unit_anatomy(
        sources,
        unit_qc,
        anatomy_path=anatomy_path,
    )
    coverage = dg.network_interactions.build_region_pair_coverage(unit_anatomy)
    candidate_pairs = coverage.filter(pl.col("coverage_eligible")).select(
        "source_region", "target_region"
    )
    if candidate_pairs.is_empty():
        raise RuntimeError("no ordered region pair passes the locked discovery coverage criteria")
    print(
        f"Figure 5: {candidate_pairs.height} ordered pairs pass coverage; "
        f"{network_trials.height} pre-lick trials",
        flush=True,
    )

    cache_root = result_root / "cache" / "network_sessions"
    cache_root.mkdir(parents=True, exist_ok=True)
    block_tables: list[pl.DataFrame] = []
    fold_tables: list[pl.DataFrame] = []
    status_tables: list[pl.DataFrame] = []
    for session_index, source_row in enumerate(
        sources.sort("subject_id", "ecephys_session_id").to_dicts(), start=1
    ):
        source = str(source_row["_nwb_path"])
        cache_key = hashlib.sha256(source.encode()).hexdigest()[:20]
        session_cache = cache_root / cache_key
        cached = None
        if not arguments.recompute_session_cache:
            cached = _read_session_cache(
                session_cache,
                source=source,
                analysis_signature=signature,
            )
        if cached is None:
            print(
                f"[{session_index:02d}/{sources.height:02d}] reducing spikes via lazynwb/obstore",
                flush=True,
            )
            block_frame, fold_frame, status_frame = _analyze_session(
                source_row,
                network_trials.filter(pl.col("_nwb_path") == source),
                unit_anatomy.filter(pl.col("_nwb_path") == source),
                candidate_pairs,
            )
            _write_session_cache(
                session_cache,
                source=source,
                analysis_signature=signature,
                block_performance=block_frame,
                fold_performance=fold_frame,
                session_status=status_frame,
            )
        else:
            block_frame, fold_frame, status_frame = cached
            print(
                f"[{session_index:02d}/{sources.height:02d}] loaded validated session cache",
                flush=True,
            )
        if block_frame.height:
            block_tables.append(block_frame)
        if fold_frame.height:
            fold_tables.append(fold_frame)
        status_tables.append(status_frame)
        gc.collect()
    if not block_tables:
        raise RuntimeError("no discovery session produced an estimable network pair")
    block_performance = pl.concat(block_tables, how="diagonal_relaxed").sort(
        "source_region", "target_region", "subject_id", "_nwb_path", "reward_block"
    )
    fold_performance = pl.concat(fold_tables, how="diagonal_relaxed").sort(
        "source_region",
        "target_region",
        "subject_id",
        "_nwb_path",
        "reward_block",
        "outer_fold",
    )
    session_status = pl.concat(status_tables, how="diagonal_relaxed").sort(
        "subject_id", "_nwb_path", "source_region", "target_region"
    )
    mouse_blocks, mouse_effects = dg.network_interactions.summarize_network_by_mouse(
        block_performance
    )
    pair_screen = dg.network_interactions.infer_and_nominate_network_pair(
        mouse_effects,
        seed=arguments.seed,
        n_bootstrap=arguments.bootstrap_resamples,
        n_sign_flips=arguments.sign_flip_resamples,
    )
    if pair_screen.is_empty() or pair_screen.filter(pl.col("nominated")).height != 1:
        raise RuntimeError("network screen could not nominate exactly one discovery pair")
    nominated = pair_screen.filter(pl.col("nominated")).row(0, named=True)
    statistics = _build_statistics(
        pair_screen,
        block_performance=block_performance,
        seed=arguments.seed,
        code_version=_code_version(REPOSITORY_ROOT),
    )

    output_paths = {
        "network_unit_anatomy": tables / "network_unit_anatomy.parquet",
        "region_pair_coverage": tables / "region_pair_coverage.csv",
        "network_pair_fold_performance": tables / "network_pair_fold_performance.parquet",
        "network_pair_block_performance": tables / "network_pair_block_performance.csv",
        "network_session_status": tables / "network_session_status.csv",
        "mouse_network_block_performance": tables / "mouse_network_block_performance.csv",
        "mouse_network_effects": tables / "mouse_network_effects.csv",
        "network_pair_screen": tables / "network_pair_screen.csv",
        "network_statistics": tables / "network_statistics.csv",
    }
    output_frames = {
        "network_unit_anatomy": unit_anatomy,
        "region_pair_coverage": coverage,
        "network_pair_fold_performance": fold_performance,
        "network_pair_block_performance": block_performance,
        "network_session_status": session_status,
        "mouse_network_block_performance": mouse_blocks,
        "mouse_network_effects": mouse_effects,
        "network_pair_screen": pair_screen,
        "network_statistics": statistics,
    }
    for key, frame in output_frames.items():
        dg.artifacts.write_frame(frame, output_paths[key])
    model_path = result_root / "models" / "nominated_network_models" / "nominated_pair.json"
    dg.artifacts.write_json(
        {
            "analysis_id": ANALYSIS_ID,
            "analysis_tier": ANALYSIS_TIER,
            "confirmation_accessed": False,
            "source_region": nominated["source_region"],
            "target_region": nominated["target_region"],
            "selection_rule": (
                "maximum controlled reversible contrast among coverage-eligible pairs"
            ),
            "selection_bias_warning": True,
            "config": dataclasses.asdict(dg.network_interactions.DEFAULT_NETWORK_CONFIG),
            "covariates": (
                "image identity, linear/quadratic session time, eventual response/latency, "
                "and target early population history"
            ),
            "primary_control": "within-block half-cycle trial shift of source population",
        },
        model_path,
    )
    environment_path = manifests / "network_environment.txt"
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        environment_path,
    )
    output_paths["nominated_network_pair"] = model_path
    output_paths["network_environment"] = environment_path
    outputs = {key: _file_record(path, base=result_root) for key, path in output_paths.items()}
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "run_status": "complete",
        "authoritative": True,
        "preliminary_vertical_slice": True,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "code_version": _code_version(REPOSITORY_ROOT),
        "confirmation_accessed": False,
        "confirmation_policy": "sealed; no confirmation identity or neural source loaded",
        "anatomy_source": anatomy_source,
        "nominated_pair": {
            "source_region": nominated["source_region"],
            "target_region": nominated["target_region"],
            "estimate": nominated["estimate"],
            "adjusted_p_value": nominated["adjusted_p_value"],
            "status": nominated["status"],
        },
        "model": dataclasses.asdict(dg.network_interactions.DEFAULT_NETWORK_CONFIG),
        "interpretation": (
            "positive-lag predictive association; not causal direction or direct connectivity"
        ),
        "limitations": (
            "preliminary isolation-only discovery slice; D04, running, and pupil adjustment "
            "remain outstanding; region pair nomination is selection-biased"
        ),
        "data_access": {
            "library": "lazynwb",
            "version": "1.0.0.dev8",
            "transport": "obstore byte-range reads",
            "string_reader": "dg.lazynwb_obstore.read_vlen_string_column",
            "spike_reader": "dg.lazynwb_obstore.map_indexed_numeric_column_batches",
            "spike_batch_size_units": SPIKE_BATCH_SIZE,
            "peak_materialization": "one bounded spike-time batch from one session",
            "legacy_accessor_forbidden_at_runtime": True,
            "legacy_accessor_calls": 0,
        },
        "inputs": input_records,
        "local_sources": source_records,
        "outputs": outputs,
    }
    dg.artifacts.write_json(manifest, manifest_path)
    print(
        "Completed Figure 5 discovery analysis: "
        f"{nominated['source_region']} -> {nominated['target_region']}, "
        f"controlled reversible delta-R2={nominated['estimate']:.4f}",
        flush=True,
    )


def _validate_discovery_manifest(path: pathlib.Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "analysis_tier": "discovery",
        "run_status": "complete",
        "authoritative": True,
        "confirmation_accessed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Figure 2 manifest {key!r} is not {expected!r}")


def _validate_discovery_sources(sources: pl.DataFrame) -> None:
    required = {"_nwb_path", "subject_id", "ecephys_session_id"}
    missing = required.difference(sources.columns)
    if missing:
        raise ValueError(f"discovery sources missing columns: {sorted(missing)}")
    if sources.select("_nwb_path").n_unique() != sources.height:
        raise ValueError("discovery sources are not unique")
    if sources.height != EXPECTED_DISCOVERY_SESSIONS:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_SESSIONS} discovery sessions; found {sources.height}"
        )
    if sources.get_column("subject_id").n_unique() != EXPECTED_DISCOVERY_MICE:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_MICE} discovery mice; "
            f"found {sources.get_column('subject_id').n_unique()}"
        )


def _load_or_read_unit_anatomy(
    sources: pl.DataFrame,
    unit_qc: pl.DataFrame,
    *,
    anatomy_path: pathlib.Path,
) -> tuple[pl.DataFrame, str]:
    selected = unit_qc.filter(pl.col("well_isolated")).select(
        "_nwb_path",
        "subject_id",
        "ecephys_session_id",
        "_table_index",
        "id",
        "well_isolated",
        "presence_ratio",
        "firing_rate",
    )
    if anatomy_path.is_file():
        schema = pl.read_parquet_schema(anatomy_path)
        region_column = next(
            (column for column in REGION_COLUMN_PRIORITY if column in schema), None
        )
        if region_column is not None:
            source_values = sources.get_column("_nwb_path").to_list()
            anatomy = (
                pl.scan_parquet(anatomy_path)
                .filter(pl.col("_nwb_path").is_in(source_values))
                .select(
                    "_nwb_path",
                    "_table_index",
                    pl.col(region_column).cast(pl.String).alias("source_region_label"),
                    *(
                        [pl.col("confirmation_accessed").cast(pl.Boolean)]
                        if "confirmation_accessed" in schema
                        else []
                    ),
                )
                .collect()
            )
            if (
                "confirmation_accessed" in anatomy.columns
                and anatomy.get_column("confirmation_accessed").fill_null(True).any()
            ):
                raise RuntimeError("external anatomy table indicates confirmation access")
            anatomy = anatomy.drop("confirmation_accessed", strict=False)
            anatomy = anatomy.with_columns(pl.lit(True).alias("anatomy_row_present"))
            joined = selected.join(
                anatomy,
                on=("_nwb_path", "_table_index"),
                how="left",
                validate="1:1",
            )
            if joined.get_column("anatomy_row_present").fill_null(False).all():
                return _finalize_unit_anatomy(
                    joined.drop("anatomy_row_present"), region_level=region_column
                ), (f"results/tables/neural_unit_anatomy.parquet:{region_column}")

    frames = []
    with _forbid_legacy_accessors():
        for index, row in enumerate(sources.sort("subject_id", "ecephys_session_id").to_dicts(), 1):
            source = str(row["_nwb_path"])
            units = selected.filter(pl.col("_nwb_path") == source).sort("_table_index")
            labels = dg.lazynwb_obstore.read_vlen_string_column(
                source,
                "/units",
                "structure_layer",
                row_indices=units.get_column("_table_index").to_list(),
            ).rename({"structure_layer": "source_region_label"})
            frames.append(
                units.join(
                    labels.select("_nwb_path", "_table_index", "source_region_label"),
                    on=("_nwb_path", "_table_index"),
                    how="left",
                    validate="1:1",
                )
            )
            print(f"    anatomy {index:02d}/{sources.height:02d}", flush=True)
    return _finalize_unit_anatomy(
        pl.concat(frames, how="diagonal_relaxed"), region_level="raw_structure_layer"
    ), "session-local /units/structure_layer via lazynwb obstore custom reader"


def _finalize_unit_anatomy(frame: pl.DataFrame, *, region_level: str) -> pl.DataFrame:
    excluded = ("", "root", "fiber tracts", "No Area", "out of brain", "nan", "None")
    return (
        frame.with_columns(
            pl.col("source_region_label").str.strip_chars().alias("source_region_label")
        )
        .with_columns(
            pl.when(
                pl.col("source_region_label").is_null()
                | pl.col("source_region_label").is_in(excluded)
            )
            .then(None)
            .otherwise(pl.col("source_region_label"))
            .alias("network_region")
        )
        .with_columns(
            pl.lit(region_level).alias("network_region_level"),
            pl.when(pl.col("network_region").is_null())
            .then(pl.lit("excluded_missing_or_non_neural_region"))
            .otherwise(pl.lit("included"))
            .alias("network_region_status"),
        )
        .sort("subject_id", "ecephys_session_id", "_table_index")
    )


def _analyze_session(
    source_row: dict[str, Any],
    trials: pl.DataFrame,
    units: pl.DataFrame,
    candidate_pairs: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = str(source_row["_nwb_path"])
    subject = str(source_row["subject_id"])
    session_id = int(source_row["ecephys_session_id"])
    status_rows: list[dict[str, Any]] = []
    counts = {
        block: trials.filter(pl.col("reward_block") == block).height
        for block in dg.network_interactions.REWARD_BLOCKS
    }
    minimum_trials = dg.network_interactions.DEFAULT_NETWORK_CONFIG.minimum_trials_per_block
    session_pairs = []
    for pair in candidate_pairs.iter_rows(named=True):
        source_count = units.filter(pl.col("network_region") == pair["source_region"]).height
        target_count = units.filter(pl.col("network_region") == pair["target_region"]).height
        if min(source_count, target_count) >= (
            dg.network_interactions.DEFAULT_NETWORK_CONFIG.minimum_units_per_region
        ):
            session_pairs.append(pair)
    if min(counts.values(), default=0) < minimum_trials or not session_pairs:
        reason = (
            "insufficient_trials"
            if min(counts.values(), default=0) < minimum_trials
            else "no_coverage_eligible_pair_in_session"
        )
        status_rows.append(_status_row(source, subject, session_id, None, None, reason, counts))
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame(status_rows)

    used_regions = {value for pair in session_pairs for value in pair.values()}
    selected_units = units.filter(pl.col("network_region").is_in(sorted(used_regions))).sort(
        "_table_index"
    )
    event_times = trials.get_column("change_time").to_numpy()
    with _forbid_legacy_accessors():
        early, late = _build_session_features(
            source,
            selected_units.get_column("_table_index").to_list(),
            event_times,
        )
    feature_position = {
        int(row_index): position
        for position, row_index in enumerate(selected_units.get_column("_table_index"))
    }
    nuisance, covariate_names = dg.network_interactions.make_trial_covariates(trials)
    block_frames: list[pl.DataFrame] = []
    fold_frames: list[pl.DataFrame] = []
    for pair in session_pairs:
        source_rows, target_rows = _balanced_region_unit_rows(
            selected_units,
            str(pair["source_region"]),
            str(pair["target_region"]),
        )
        source_positions = [feature_position[int(value)] for value in source_rows]
        target_positions = [feature_position[int(value)] for value in target_rows]
        try:
            fit = dg.network_interactions.fit_network_pair_by_block(
                early[:, source_positions],
                early[:, target_positions],
                late[:, target_positions],
                nuisance,
                trials.get_column("reward_block").to_numpy(),
                event_times,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            status_rows.append(
                _status_row(
                    source,
                    subject,
                    session_id,
                    str(pair["source_region"]),
                    str(pair["target_region"]),
                    f"model_failed:{type(error).__name__}",
                    counts,
                )
            )
            continue
        metadata = (
            pl.lit(source).alias("_nwb_path"),
            pl.lit(subject).alias("subject_id"),
            pl.lit(session_id).alias("ecephys_session_id"),
            pl.lit(str(pair["source_region"])).alias("source_region"),
            pl.lit(str(pair["target_region"])).alias("target_region"),
            pl.lit(json.dumps(source_rows)).alias("source_unit_table_indices"),
            pl.lit(json.dumps(target_rows)).alias("target_unit_table_indices"),
            pl.lit(json.dumps(covariate_names)).alias("nuisance_covariates"),
        )
        block_frames.append(fit.block_performance.with_columns(*metadata))
        fold_frames.append(fit.fold_performance.with_columns(*metadata))
        status_rows.append(
            _status_row(
                source,
                subject,
                session_id,
                str(pair["source_region"]),
                str(pair["target_region"]),
                "pass",
                counts,
            )
        )
    del early, late
    return (
        pl.concat(block_frames, how="diagonal_relaxed") if block_frames else pl.DataFrame(),
        pl.concat(fold_frames, how="diagonal_relaxed") if fold_frames else pl.DataFrame(),
        pl.DataFrame(status_rows),
    )


def _balanced_region_unit_rows(
    units: pl.DataFrame,
    source_region: str,
    target_region: str,
) -> tuple[list[int], list[int]]:
    config = dg.network_interactions.DEFAULT_NETWORK_CONFIG

    def ranked(region: str) -> list[int]:
        return (
            units.filter(pl.col("network_region") == region)
            .sort(
                "presence_ratio",
                "firing_rate",
                "_table_index",
                descending=(True, True, False),
                nulls_last=True,
            )
            .get_column("_table_index")
            .to_list()
        )

    source_rows = ranked(source_region)
    target_rows = ranked(target_region)
    n_units = min(len(source_rows), len(target_rows), config.maximum_units_per_region)
    if n_units < config.minimum_units_per_region:
        raise ValueError("region pair is below the minimum balanced neuron count")
    return source_rows[:n_units], target_rows[:n_units]


def _build_session_features(
    source: str,
    row_indices: list[int],
    event_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    early = np.empty((event_times.size, len(row_indices)), dtype=float)
    late = np.empty_like(early)
    next_column = 0
    n_batches = math.ceil(len(row_indices) / SPIKE_BATCH_SIZE)

    def consume(batch_rows: tuple[int, ...], spike_rows: list[list[Any]]) -> None:
        nonlocal next_column
        expected = tuple(row_indices[next_column : next_column + len(batch_rows)])
        if batch_rows != expected:
            raise RuntimeError("obstore spike batch order differs from selected unit order")
        batch_early, batch_late = dg.network_interactions.build_windowed_population_features(
            spike_rows,
            event_times,
        )
        stop = next_column + len(batch_rows)
        early[:, next_column:stop] = batch_early
        late[:, next_column:stop] = batch_late
        next_column = stop
        completed = math.ceil(next_column / SPIKE_BATCH_SIZE)
        if completed == n_batches or completed % 10 == 0:
            print(f"      reduced spike batch {completed}/{n_batches}", flush=True)
        del batch_early, batch_late

    dg.lazynwb_obstore.map_indexed_numeric_column_batches(
        source,
        "/units",
        "spike_times",
        row_indices,
        consume,
        batch_size=SPIKE_BATCH_SIZE,
    )
    if next_column != len(row_indices):
        raise RuntimeError("obstore spike batches did not cover every selected unit")
    return early, late


def _status_row(
    source: str,
    subject: str,
    session_id: int,
    source_region: str | None,
    target_region: str | None,
    status: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "_nwb_path": source,
        "subject_id": subject,
        "ecephys_session_id": session_id,
        "source_region": source_region,
        "target_region": target_region,
        "status": status,
        "n_engaged_1_trials": counts.get("engaged_1", 0),
        "n_no_reward_trials": counts.get("no_reward", 0),
        "n_engaged_2_trials": counts.get("engaged_2", 0),
    }


def _build_statistics(
    pair_screen: pl.DataFrame,
    *,
    block_performance: pl.DataFrame,
    seed: int,
    code_version: str,
) -> pl.DataFrame:
    rows = []
    for index, screen in enumerate(pair_screen.iter_rows(named=True)):
        source = screen["source_region"]
        target = screen["target_region"]
        session_rows = block_performance.filter(
            (pl.col("source_region") == source) & (pl.col("target_region") == target)
        )
        unit_sessions = session_rows.unique(subset=("_nwb_path", "source_region", "target_region"))
        rows.append(
            {
                "analysis_id": ANALYSIS_ID,
                "result_id": f"{source}_to_{target}_controlled_reversible_source_added_r2",
                "contrast_id": "real_minus_trial_shift_reversible_delta_r2",
                "hypothesis": (
                    "Positive-lag source-added target prediction is stronger in reward-available "
                    "blocks than NR beyond the same contrast after within-block trial shifting"
                ),
                "dandiset_version": dg.data.DANDISET_VERSION,
                "code_version": code_version,
                "seed": seed + index + 1,
                "analysis_tier": ANALYSIS_TIER,
                "inclusion_definition": (
                    "behavior-eligible discovery sessions; isolation QC; familiar full-contrast "
                    "physical changes; no lick from -150 through +300 ms; equal trials and units"
                ),
                "missingness_stratum": (
                    "running/pupil not included in preliminary slice; eventual lick response and "
                    "latency included; D04 consistency pending"
                ),
                "estimate": screen["estimate"],
                "scale": "held-out target R-squared difference in reversible contrasts",
                "ci_low": screen["ci_low"],
                "ci_high": screen["ci_high"],
                "confidence_level": 0.95,
                "ci_method": "mouse percentile bootstrap",
                "test_statistic": abs(float(screen["estimate"])),
                "test_method": screen["test_method"],
                "p_value": screen["p_value"],
                "adjusted_p_value": screen["adjusted_p_value"],
                "adjustment_method": "holm",
                "multiplicity_family": "discovery_region_pair_interaction_screen",
                "sidedness": "two-sided",
                "n_mice": screen["n_mice"],
                "n_sessions": unit_sessions.height,
                "n_probes": None,
                "n_units": int(
                    unit_sessions.select(
                        (pl.col("n_source_units") + pl.col("n_target_units")).sum()
                    ).item()
                ),
                "n_trials": int(session_rows.get_column("n_trials").sum()),
                "aggregation": "equal-session mean within mouse; equal-mouse mean",
                "bootstrap_id": f"mouse_percentile_seed_{seed + index + 1}",
                "model_formula": (
                    "target late [150,300) ms residual ~ positive-lag source early [0,150) ms "
                    "RRR after stimulus/time/eventual-response covariates and target early history"
                ),
                "cv_grouping": "nested contiguous trials within block and session",
                "status": screen["status"],
                "reason": screen["reason"],
            }
        )
    return dg.statistics.normalize_statistics_table(pl.DataFrame(rows))


@contextlib.contextmanager
def _forbid_legacy_accessors():
    original_get_accessor = lazynwb.file_io._get_accessor
    original_open_hdf5 = getattr(lazynwb.file_io, "_open_hdf5", None)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "legacy NWB accessor invoked; Figure 5 requires lazynwb/obstore custom reads"
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


def _analysis_signature(
    input_records: dict[str, dict[str, Any]],
    source_records: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "inputs": input_records,
        "sources": source_records,
        "config": dataclasses.asdict(dg.network_interactions.DEFAULT_NETWORK_CONFIG),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
            or manifest.get("analysis_signature") != analysis_signature
        ):
            return None
        names = {
            "block_performance": "block_performance.parquet",
            "fold_performance": "fold_performance.parquet",
            "session_status": "session_status.parquet",
        }
        paths = {key: cache_root / name for key, name in names.items()}
        for key, path in paths.items():
            if not path.is_file() or _sha256(path) != manifest["outputs"][key]["sha256"]:
                return None
        return tuple(pl.read_parquet(paths[key]) for key in names)  # type: ignore[return-value]
    except (KeyError, OSError, json.JSONDecodeError, pl.exceptions.PolarsError):
        return None


def _write_session_cache(
    cache_root: pathlib.Path,
    *,
    source: str,
    analysis_signature: str,
    block_performance: pl.DataFrame,
    fold_performance: pl.DataFrame,
    session_status: pl.DataFrame,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "block_performance": dg.artifacts.write_frame(
            block_performance, cache_root / "block_performance.parquet"
        ),
        "fold_performance": dg.artifacts.write_frame(
            fold_performance, cache_root / "fold_performance.parquet"
        ),
        "session_status": dg.artifacts.write_frame(
            session_status, cache_root / "session_status.parquet"
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
