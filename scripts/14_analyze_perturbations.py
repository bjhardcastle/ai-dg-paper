# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Apply the frozen Figure 2 axis to held-out contrast and novelty trials."""

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
import dg.data  # noqa: E402
import dg.lazynwb_obstore  # noqa: E402
import dg.neural_transitions  # noqa: E402
import dg.perturbations  # noqa: E402
import dg.statistics  # noqa: E402

ANALYSIS_ID = "heldout_perturbation_axis"
ANALYSIS_TIER = "discovery"
DEFAULT_SEED = 6100
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SIGN_FLIP_RESAMPLES = 100_000
SPIKE_BATCH_SIZE = 32
CACHE_SCHEMA_VERSION = 1
COMPATIBLE_SESSION_CACHE_SIGNATURES = frozenset(
    {"86d932ddb15cc3b4cc49f12b3e3b5a91bf994896ffb758c8a70ee080f6053d8e"}
)
EXPECTED_DISCOVERY_SESSIONS = 23
EXPECTED_DISCOVERY_MICE = 12
LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/data.py",
    "src/dg/lazynwb_obstore.py",
    "src/dg/neural_transitions.py",
    "src/dg/perturbations.py",
    "src/dg/statistics.py",
    "scripts/14_analyze_perturbations.py",
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--recompute-session-cache", action="store_true")
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments()
    if arguments.seed < 0:
        raise ValueError("--seed must be non-negative")
    if arguments.bootstrap_resamples < 1 or arguments.sign_flip_resamples < 1:
        raise ValueError("resample counts must be positive")

    result_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = result_root / "manifests" / "perturbation_analysis_run.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "run_status": "in_progress",
        "authoritative": False,
        "confirmation_accessed": False,
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
    tables = result_root / "tables"
    inputs = {
        "behavior_trials": tables / "behavior_trials.parquet",
        "discovery_sources": tables / "discovery_session_sources.parquet",
        "unit_axis": tables / "neural_transition_unit_qc.parquet",
        "axis_manifest": result_root / "manifests" / "neural_transition_analysis_run.json",
        "analysis_lock": REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
    }
    missing = [path for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Figure 6 inputs: " + ", ".join(map(str, missing)))

    axis_manifest = json.loads(inputs["axis_manifest"].read_text(encoding="utf-8"))
    if (
        axis_manifest.get("run_status") != "complete"
        or axis_manifest.get("analysis_tier") != "discovery"
        or axis_manifest.get("confirmation_accessed") is not False
    ):
        raise RuntimeError("Figure 2 axis is not a completed sealed discovery artifact")

    input_records = {
        key: _file_record(
            path,
            base=result_root if path.is_relative_to(result_root) else REPOSITORY_ROOT,
        )
        for key, path in inputs.items()
    }
    source_records = {
        path: _file_record(REPOSITORY_ROOT / path, base=REPOSITORY_ROOT)
        for path in LOCAL_SOURCE_PATHS
    }
    signature = _analysis_signature(input_records, source_records)

    sources = pl.read_parquet(inputs["discovery_sources"]).sort("subject_id", "ecephys_session_id")
    if sources.height != EXPECTED_DISCOVERY_SESSIONS:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_SESSIONS} discovery sessions; found {sources.height}"
        )
    if sources.get_column("subject_id").n_unique() != EXPECTED_DISCOVERY_MICE:
        raise RuntimeError(
            f"expected {EXPECTED_DISCOVERY_MICE} discovery mice; "
            f"found {sources.get_column('subject_id').n_unique()}"
        )

    behavior = pl.read_parquet(inputs["behavior_trials"]).join(
        sources.select("_nwb_path"),
        on="_nwb_path",
        how="inner",
        validate="m:1",
    )
    selected = dg.perturbations.select_perturbation_trials(behavior)
    if selected.get_column("_nwb_path").n_unique() != sources.height:
        raise RuntimeError("eligible perturbation trials do not cover every discovery session")

    units = pl.read_parquet(inputs["unit_axis"]).join(
        sources.select("_nwb_path"),
        on="_nwb_path",
        how="inner",
        validate="m:1",
    )
    selected_units = units.filter(
        pl.col("well_isolated").fill_null(False)
        & pl.col("axis_weight").is_not_null()
        & pl.col("axis_weight").is_finite()
    )
    if selected_units.get_column("_nwb_path").n_unique() != sources.height:
        raise RuntimeError("frozen axis units do not cover every discovery session")

    cache_root = result_root / "cache" / "perturbation_sessions"
    cache_root.mkdir(parents=True, exist_ok=True)
    scored_sessions: list[pl.DataFrame] = []
    for index, source_row in enumerate(sources.iter_rows(named=True), start=1):
        source = str(source_row["_nwb_path"])
        cache_path = cache_root / hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
        cached = None
        if not arguments.recompute_session_cache:
            cached = _read_session_cache(
                cache_path,
                source=source,
                signature=signature,
            )
        if cached is not None:
            print(f"[{index}/{sources.height}] reused perturbation checkpoint", flush=True)
            scored_sessions.append(cached)
            continue

        session_trials = selected.filter(pl.col("_nwb_path") == source).sort(
            "change_time", "_table_index"
        )
        session_units = selected_units.filter(pl.col("_nwb_path") == source).sort("_table_index")
        print(
            f"[{index}/{sources.height}] scoring {session_trials.height} trials with "
            f"{session_units.height} units in {SPIKE_BATCH_SIZE}-unit batches",
            flush=True,
        )
        scored = _score_session(source, session_trials, session_units)
        _write_session_cache(
            cache_path,
            source=source,
            signature=signature,
            scored=scored,
        )
        scored_sessions.append(scored)
        gc.collect()

    unique_scores = pl.concat(scored_sessions, how="diagonal_relaxed").sort(
        "_nwb_path", "change_time", "_table_index"
    )
    expanded_scores = dg.perturbations.expand_probe_conditions(unique_scores)
    session_blocks = dg.perturbations.summarize_session_blocks(expanded_scores)
    session_contrasts = dg.perturbations.compute_session_reversible_contrasts(session_blocks)
    mouse_contrasts = dg.perturbations.aggregate_mouse_contrasts(session_contrasts)
    mouse_interactions = dg.perturbations.compute_mouse_perturbation_interactions(mouse_contrasts)
    code_version = _code_version()
    statistics = _build_statistics(
        mouse_contrasts,
        mouse_interactions,
        session_contrasts=session_contrasts,
        selected_units=selected_units,
        seed=arguments.seed,
        n_bootstrap=arguments.bootstrap_resamples,
        n_sign_flips=arguments.sign_flip_resamples,
        code_version=code_version,
    )

    outputs = {
        "perturbation_trial_scores": unique_scores,
        "session_perturbation_blocks": session_blocks,
        "session_perturbation_contrasts": session_contrasts,
        "mouse_perturbation_contrasts": mouse_contrasts,
        "mouse_perturbation_interactions": mouse_interactions,
        "perturbation_statistics": statistics,
    }
    output_paths: dict[str, pathlib.Path] = {}
    for name, frame in outputs.items():
        suffix = ".parquet" if name == "perturbation_trial_scores" else ".csv"
        path = tables / f"{name}{suffix}"
        dg.artifacts.write_frame(frame, path)
        output_paths[name] = path

    environment_path = result_root / "manifests" / "perturbation_environment.txt"
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(
            repository=REPOSITORY_ROOT,
            package_names=("lazynwb", "numpy", "polars", "pyarrow"),
        ),
        environment_path,
    )
    completed_at = datetime.datetime.now(datetime.UTC)
    output_records = {
        **{key: _file_record(path, base=result_root) for key, path in output_paths.items()},
        "environment": _file_record(environment_path, base=result_root),
    }
    dg.artifacts.write_json(
        {
            "analysis_id": ANALYSIS_ID,
            "analysis_status": "complete",
            "run_status": "complete",
            "analysis_tier": ANALYSIS_TIER,
            "authoritative": True,
            "preliminary_vertical_slice": True,
            "confirmation_accessed": False,
            "confirmation_policy": "sealed; no confirmation identity or neural source was loaded",
            "dandiset_id": dg.data.DANDISET_ID,
            "dandiset_version": dg.data.DANDISET_VERSION,
            "code_version": code_version,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "inputs": input_records,
            "local_sources": source_records,
            "outputs": output_records,
            "data_access": {
                "library": "lazynwb",
                "version": lazynwb.__version__,
                "transport": "obstore byte-range reads",
                "spike_reader": "lazynwb.scan_nwb custom reader",
                "spike_batch_size_units": SPIKE_BATCH_SIZE,
                "peak_spike_materialization": "one bounded unit batch per session",
                "legacy_accessor_forbidden_at_runtime": True,
                "legacy_accessor_calls": 0,
            },
            "representation": (
                "Figure 2 final E1-versus-NR familiar full-contrast axis; reduced-contrast "
                "and designated-novel trials excluded from representation fitting"
            ),
            "novelty_scope": (
                "session-designated novel identity, not a pure first-exposure effect; "
                "identity and recording day remain confounded"
            ),
            "counts": {
                "mice": sources.get_column("subject_id").n_unique(),
                "sessions": sources.height,
                "well_isolated_axis_units": selected_units.height,
                "unique_scored_trials": unique_scores.height,
                "expanded_probe_rows": expanded_scores.height,
            },
            "config": dataclasses.asdict(dg.perturbations.DEFAULT_PERTURBATION_CONFIG),
        },
        manifest_path,
    )
    print(
        f"Figure 6 analysis complete: {sources.height} sessions, "
        f"{unique_scores.height} unique trials",
        flush=True,
    )


def _score_session(
    source: str,
    trials: pl.DataFrame,
    units: pl.DataFrame,
) -> pl.DataFrame:
    indices = units.get_column("_table_index").cast(pl.Int64).to_list()
    weights = units.get_column("axis_weight").to_numpy()
    intercepts = units.get_column("axis_intercept").drop_nulls().unique()
    if len(intercepts) != 1 or not math.isfinite(float(intercepts[0])):
        raise RuntimeError("session does not have exactly one finite frozen axis intercept")
    event_times = trials.get_column("change_time").to_numpy()
    scores = np.full(event_times.size, float(intercepts[0]), dtype=float)
    next_unit = 0

    def consume(batch_rows: tuple[int, ...], spike_rows: list[list[Any]]) -> None:
        nonlocal next_unit
        expected = tuple(indices[next_unit : next_unit + len(batch_rows)])
        if batch_rows != expected:
            raise RuntimeError("obstore spike batch order differs from frozen unit order")
        features = dg.neural_transitions.build_spike_feature_matrix(spike_rows, event_times)
        stop = next_unit + len(batch_rows)
        scores[:] += features @ weights[next_unit:stop]
        next_unit = stop
        del features

    with _forbid_legacy_accessors():
        dg.lazynwb_obstore.map_indexed_numeric_column_batches(
            source,
            "/units",
            "spike_times",
            indices,
            consume,
            batch_size=SPIKE_BATCH_SIZE,
        )
    if next_unit != len(indices):
        raise RuntimeError("obstore batches did not cover every frozen axis unit")
    if not np.isfinite(scores).all():
        raise RuntimeError("frozen-axis projection produced non-finite scores")
    return trials.with_columns(
        pl.Series("axis_score", scores, dtype=pl.Float64),
        pl.lit("final_familiar_full_contrast_axis_projection").alias("score_role"),
    )


def _build_statistics(
    mouse_contrasts: pl.DataFrame,
    mouse_interactions: pl.DataFrame,
    *,
    session_contrasts: pl.DataFrame,
    selected_units: pl.DataFrame,
    seed: int,
    n_bootstrap: int,
    n_sign_flips: int,
    code_version: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    for family in ("contrast", "novelty"):
        conditions = (
            ("full", "reduced") if family == "contrast" else ("familiar", "designated_novel")
        )
        for metric in ("behavior_response", "early_reward_axis"):
            for condition in conditions:
                values = mouse_contrasts.filter(
                    (pl.col("perturbation_family") == family)
                    & (pl.col("metric") == metric)
                    & (pl.col("condition") == condition)
                )
                support_sessions = session_contrasts.filter(
                    (pl.col("status") == "pass")
                    & (pl.col("perturbation_family") == family)
                    & (pl.col("metric") == metric)
                    & (pl.col("condition") == condition)
                )
                rows.append(
                    _statistic_row(
                        values,
                        value_column="reversible_contrast",
                        result_id=f"{metric}_{family}_{condition}_reversible_mouse_mean",
                        contrast_id=f"{metric}_{family}_{condition}_half_e1_e2_minus_nr",
                        hypothesis=(
                            f"{metric} follows reversible reward availability for the "
                            f"{family} probe condition {condition}"
                        ),
                        scale=(
                            "response-probability difference"
                            if metric == "behavior_response"
                            else "frozen-axis score difference"
                        ),
                        multiplicity_family=f"perturbation_{metric}_condition_contrasts",
                        seed=seed + offset,
                        n_bootstrap=n_bootstrap,
                        n_sign_flips=n_sign_flips,
                        code_version=code_version,
                        support=_support_counts(support_sessions, selected_units),
                        model_formula="0.5 * E1 - NR + 0.5 * E2 within session and condition",
                    )
                )
                offset += 1

            interaction_values = mouse_interactions.filter(
                (pl.col("perturbation_family") == family) & (pl.col("metric") == metric)
            )
            support_sessions = session_contrasts.filter(
                (pl.col("status") == "pass")
                & (pl.col("perturbation_family") == family)
                & (pl.col("metric") == metric)
            )
            rows.append(
                _statistic_row(
                    interaction_values,
                    value_column="interaction",
                    result_id=f"{metric}_{family}_state_by_perturbation_interaction_mouse_mean",
                    contrast_id=f"{metric}_{family}_test_minus_reference_reversible_contrast",
                    hypothesis=(
                        f"The reversible reward-state contrast differs between the held-out "
                        f"{family} condition and its reference"
                    ),
                    scale=(
                        "difference in response-probability contrasts"
                        if metric == "behavior_response"
                        else "difference in frozen-axis score contrasts"
                    ),
                    multiplicity_family=f"perturbation_{metric}_interactions",
                    seed=seed + offset,
                    n_bootstrap=n_bootstrap,
                    n_sign_flips=n_sign_flips,
                    code_version=code_version,
                    support=_support_counts(support_sessions, selected_units),
                    model_formula=(
                        "held-out perturbation minus reference difference in "
                        "0.5 * E1 - NR + 0.5 * E2"
                    ),
                )
            )
            offset += 1
    return dg.statistics.add_holm_adjustment(
        dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    )


def _support_counts(
    sessions: pl.DataFrame,
    units: pl.DataFrame,
) -> dict[str, int]:
    sources = sessions.select("_nwb_path").unique()
    supported_units = units.join(sources, on="_nwb_path", how="inner", validate="m:1")
    return {
        "n_mice": sessions.get_column("subject_id").n_unique(),
        "n_sessions": sources.height,
        "n_units": supported_units.height,
        "n_trials": int(sessions.get_column("n_trials").sum()),
    }


def _statistic_row(
    values: pl.DataFrame,
    *,
    value_column: str,
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
    model_formula: str,
) -> dict[str, Any]:
    mouse_values = values.select(
        pl.col("subject_id").cast(pl.String),
        pl.col(value_column).cast(pl.Float64),
    ).drop_nulls()
    if mouse_values.height < 2:
        raise RuntimeError(f"{result_id} has fewer than two contributing mice")
    bootstrap = dg.statistics.bootstrap_mouse_mean(
        mouse_values,
        seed=seed,
        mouse_column="subject_id",
        value_column=value_column,
        n_resamples=n_bootstrap,
    )
    test = dg.statistics.two_sided_sign_flip_test(
        mouse_values,
        seed=seed,
        mouse_column="subject_id",
        value_column=value_column,
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
            "behavior-eligible discovery sessions; Figure 2 isolation-only frozen axis; "
            "completed non-auto physical changes with no lick within +/-150 ms"
        ),
        "missingness_stratum": "complete E1/NR/E2 support with >=5 trials per session block",
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
        "model_formula": model_formula,
        "cv_grouping": (
            "frozen Figure 2 axis; contrast and designated-novel trials excluded from fitting"
        ),
        "status": "pass",
        "reason": (
            "discovery-only mechanistic constraint; D04 not applied; novelty is identity/day "
            "confounded and is not a pure first-exposure effect"
        ),
    }


@contextlib.contextmanager
def _forbid_legacy_accessors():
    original_get_accessor = lazynwb.file_io._get_accessor
    original_open_hdf5 = getattr(lazynwb.file_io, "_open_hdf5", None)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "legacy NWB accessor invoked; Figure 6 requires lazynwb/obstore custom reads"
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
        "config": dataclasses.asdict(dg.perturbations.DEFAULT_PERTURBATION_CONFIG),
        "spike_batch_size": SPIKE_BATCH_SIZE,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _read_session_cache(
    cache_root: pathlib.Path,
    *,
    source: str,
    signature: str,
) -> pl.DataFrame | None:
    manifest_path = cache_root / "manifest.json"
    score_path = cache_root / "trial_scores.parquet"
    if not manifest_path.is_file() or not score_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("source") != source
            or manifest.get("analysis_signature")
            not in {signature, *COMPATIBLE_SESSION_CACHE_SIGNATURES}
        ):
            return None
        record = _file_record(score_path, base=cache_root)
        if record != manifest.get("output"):
            return None
        return pl.read_parquet(score_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_session_cache(
    cache_root: pathlib.Path,
    *,
    source: str,
    signature: str,
    scored: pl.DataFrame,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    score_path = cache_root / "trial_scores.parquet"
    dg.artifacts.write_frame(scored, score_path)
    dg.artifacts.write_json(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "source": source,
            "analysis_signature": signature,
            "output": _file_record(score_path, base=cache_root),
        },
        cache_root / "manifest.json",
    )


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(base)),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _code_version() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
