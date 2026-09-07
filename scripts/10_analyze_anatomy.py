# /// script
# dependencies = [
#   "lazynwb @ git+https://github.com/bjhardcastle/lazynwb.git@387c250bee6a6fddd5c96b9cf1490b8f02c292f8",
#   "numpy>=2.0",
#   "polars>=1.32",
#   "pyarrow>=18.0",
# ]
# requires-python = ">=3.11"
# ///
"""Build discovery-only Figure 4 anatomical-axis enrichment artifacts.

Only scalar anatomy is read from one discovery NWB at a time.  Unit
``structure_layer`` strings use lazynwb's obstore-backed custom HDF5 reader;
peak-electrode coordinates use a projected lazy scan.  Spike times, waveforms,
legacy accessors, and confirmation-session identities are never accessed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

import lazynwb
import lazynwb.file_io
import polars as pl

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import dg.anatomical_enrichment  # noqa: E402
import dg.artifacts  # noqa: E402
import dg.data  # noqa: E402
import dg.lazynwb_obstore  # noqa: E402
import dg.ontology  # noqa: E402

ANALYSIS_ID = dg.anatomical_enrichment.ANALYSIS_ID
ANALYSIS_TIER = "discovery"
ELECTRODE_COLUMNS = ("id", "x", "y", "z", "probe_id", "_nwb_path")
LOCAL_SOURCE_PATHS = (
    "src/dg/anatomical_enrichment.py",
    "src/dg/artifacts.py",
    "src/dg/data.py",
    "src/dg/lazynwb_obstore.py",
    "src/dg/ontology.py",
    "src/dg/statistics.py",
    "scripts/10_analyze_anatomy.py",
)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=pathlib.Path,
        default=REPOSITORY_ROOT / "results",
    )
    parser.add_argument("--seed", type=int, default=dg.anatomical_enrichment.DEFAULT_SEED)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=dg.anatomical_enrichment.DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=dg.anatomical_enrichment.DEFAULT_PERMUTATIONS,
    )
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_arguments()
    _validate_arguments(arguments)
    results_root = dg.artifacts.initialize_results_tree(arguments.results_root)
    manifest_path = results_root / "manifests" / "anatomy_analysis_run.json"
    started_at = datetime.datetime.now(datetime.UTC)
    in_progress = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "in_progress",
        "run_status": "in_progress",
        "authoritative": False,
        "script": "scripts/10_analyze_anatomy.py",
        "started_at_utc": started_at.isoformat(),
        "confirmation_accessed": False,
    }
    dg.artifacts.write_json(in_progress, manifest_path)
    try:
        _run_analysis(
            arguments,
            results_root=results_root,
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
    if arguments.bootstrap_resamples < 1 or arguments.permutations < 1:
        raise ValueError("resample counts must be positive")


def _run_analysis(
    arguments: argparse.Namespace,
    *,
    results_root: pathlib.Path,
    manifest_path: pathlib.Path,
    started_at: datetime.datetime,
) -> None:
    table_root = results_root / "tables"
    manifest_root = results_root / "manifests"
    input_paths = {
        "neural_transition_unit_qc": table_root / "neural_transition_unit_qc.parquet",
        "discovery_session_sources": table_root / "discovery_session_sources.parquet",
        "neural_transition_analysis_run": (manifest_root / "neural_transition_analysis_run.json"),
        "analysis_lock": REPOSITORY_ROOT / "config" / "analysis_lock.yaml",
    }
    missing = [path for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Figure 4 inputs: " + ", ".join(map(str, missing)))
    neural_manifest = _validated_neural_manifest(
        input_paths["neural_transition_analysis_run"],
        results_root=results_root,
        unit_path=input_paths["neural_transition_unit_qc"],
        source_path=input_paths["discovery_session_sources"],
    )
    unit_qc = pl.read_parquet(input_paths["neural_transition_unit_qc"])
    discovery_sources = pl.read_parquet(input_paths["discovery_session_sources"])
    _validate_discovery_inputs(unit_qc, discovery_sources)

    ontology_cache = results_root / "cache" / "anatomy_ontology"
    pinned_sources = dg.ontology.cache_d05_ontology_sources(ontology_cache)
    ontology = dg.ontology.load_d05_major_division_ontology(pinned_sources)
    _validate_parent_contract(ontology.divisions)

    labels: list[pl.DataFrame] = []
    electrodes: list[pl.DataFrame] = []
    source_rows = discovery_sources.sort("subject_id", "ecephys_session_id").to_dicts()
    for index, source_row in enumerate(source_rows, start=1):
        source = str(source_row["_nwb_path"])
        session_units = unit_qc.filter(pl.col("_nwb_path") == source).sort("_table_index")
        print(
            f"[{index:02d}/{len(source_rows):02d}] reading bounded scalar anatomy "
            f"for {session_units.height} units",
            flush=True,
        )
        session_labels, session_electrodes = _read_session_scalar_anatomy(
            source,
            session_units,
        )
        labels.append(session_labels)
        electrodes.append(session_electrodes)

    structure_labels = pl.concat(labels, how="diagonal_relaxed")
    electrode_table = pl.concat(electrodes, how="diagonal_relaxed")
    unit_anatomy = dg.anatomical_enrichment.assemble_unit_anatomy(
        unit_qc,
        structure_labels,
        electrode_table,
        ontology.structures,
    )
    code_version = _code_version(REPOSITORY_ROOT)
    analysis = dg.anatomical_enrichment.analyze_anatomical_enrichment(
        unit_anatomy,
        seed=arguments.seed,
        n_bootstrap=arguments.bootstrap_resamples,
        n_permutations=arguments.permutations,
        code_version=code_version,
    )
    screen = dg.anatomical_enrichment.add_screen_annotations(
        analysis.statistics,
        analysis.region_coverage,
    )

    output_paths = {
        "neural_unit_anatomy": table_root / "neural_unit_anatomy.parquet",
        "anatomical_axis_unit_scores": table_root / "anatomical_axis_unit_scores.parquet",
        "anatomy_region_coverage": table_root / "anatomy_region_coverage.csv",
        "session_anatomical_enrichment": (table_root / "session_anatomical_enrichment.parquet"),
        "mouse_anatomical_enrichment": table_root / "mouse_anatomical_enrichment.csv",
        "anatomy_enrichment_statistics": table_root / "anatomy_enrichment_statistics.csv",
        "anatomy_region_screen": table_root / "anatomy_region_screen.csv",
        "anatomy_permutation_null": table_root / "anatomy_permutation_null.parquet",
        "anatomy_ontology_provenance": (manifest_root / "anatomy_ontology_provenance.csv"),
        "anatomy_ontology_validation": (manifest_root / "anatomy_ontology_validation.csv"),
        "anatomy_environment": manifest_root / "anatomy_environment.txt",
    }
    frames = {
        "neural_unit_anatomy": unit_anatomy,
        "anatomical_axis_unit_scores": analysis.unit_scores,
        "anatomy_region_coverage": analysis.region_coverage,
        "session_anatomical_enrichment": analysis.session_effects,
        "mouse_anatomical_enrichment": analysis.mouse_effects,
        "anatomy_enrichment_statistics": analysis.statistics,
        "anatomy_region_screen": screen,
        "anatomy_permutation_null": analysis.permutation_null,
        "anatomy_ontology_provenance": pinned_sources.provenance,
        "anatomy_ontology_validation": ontology.validation,
    }
    for key, frame in frames.items():
        dg.artifacts.write_frame(frame, output_paths[key])
    dg.artifacts.write_text(
        dg.artifacts.capture_software_environment(repository=REPOSITORY_ROOT),
        output_paths["anatomy_environment"],
    )

    input_records = {
        key: _file_record(
            path,
            base=results_root if path.is_relative_to(results_root) else REPOSITORY_ROOT,
        )
        for key, path in input_paths.items()
    }
    local_source_records = {
        path: _file_record(REPOSITORY_ROOT / path, base=REPOSITORY_ROOT)
        for path in LOCAL_SOURCE_PATHS
    }
    outputs = {key: _file_record(path, base=results_root) for key, path in output_paths.items()}
    eligible_regions = analysis.region_coverage.filter(pl.col("coverage_eligible"))
    nominated = screen.filter(pl.col("nominated_for_confirmation"))
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "analysis_tier": ANALYSIS_TIER,
        "analysis_status": "complete",
        "run_status": "complete",
        "authoritative": True,
        "preliminary_vertical_slice": True,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "dandiset_id": dg.data.DANDISET_ID,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "code_version": code_version,
        "confirmation_accessed": False,
        "confirmation_policy": "sealed; no confirmation identity or source was loaded",
        "upstream_neural_manifest_code_version": neural_manifest.get("code_version"),
        "discovery_counts": {
            "mice": discovery_sources.get_column("subject_id").n_unique(),
            "sessions": discovery_sources.height,
            "all_units": unit_anatomy.height,
            "well_isolated_finite_weight_units": analysis.unit_scores.height,
            "mapped_analysis_units": unit_anatomy.filter(
                pl.col("anatomy_analysis_included")
            ).height,
            "coverage_eligible_parent_regions": eligible_regions.height,
        },
        "parent_region_contract": {
            "ontology": "Allen CCFv3 graph 1",
            "level": "official 12 major divisions",
            "structure_set_id": dg.ontology.MAJOR_DIVISION_STRUCTURE_SET_ID,
            "regions": [
                {
                    "id": row[0],
                    "acronym": row[1],
                    "name": row[2],
                }
                for row in dg.anatomical_enrichment.PARENT_REGIONS
            ],
            "coverage_minima": {
                "mice": dg.anatomical_enrichment.MIN_REGION_MICE,
                "sessions": dg.anatomical_enrichment.MIN_REGION_SESSIONS,
                "median_units_per_represented_mouse": (
                    dg.anatomical_enrichment.MIN_MEDIAN_UNITS_PER_MOUSE
                ),
            },
        },
        "analysis_definition": {
            "functional_measure": "Figure 2 session-specific E1-versus-NR axis coefficient",
            "signed_measure": "axis coefficient standardized within session",
            "absolute_measure": "absolute axis coefficient standardized within session",
            "aggregation": "region within session, equal sessions within mouse, equal mice",
            "null": "shuffle fixed anatomy labels within session",
            "permutations": arguments.permutations,
            "bootstrap_resamples": arguments.bootstrap_resamples,
            "multiplicity": "Holm across every eligible region and both score types",
            "d04_status": "not applied; isolation-only preliminary analysis",
        },
        "nominated_regions": nominated.select(
            "parent_region_acronym",
            "parent_region_name",
            "estimate",
            "adjusted_p_value",
        ).to_dicts(),
        "data_access": {
            "library": "lazynwb",
            "version": "1.0.0.dev8",
            "transport": "obstore byte-range reads",
            "structure_reader": (
                "dg.lazynwb_obstore.read_vlen_string_column(/units/structure_layer)"
            ),
            "coordinate_reader": "lazynwb.scan_nwb projected scalar electrode columns",
            "session_batching": "one discovery session at a time",
            "spike_arrays_loaded": False,
            "waveform_arrays_loaded": False,
            "legacy_accessor_forbidden_at_runtime": True,
            "legacy_accessor_calls": 0,
        },
        "inputs": input_records,
        "local_sources": local_source_records,
        "outputs": outputs,
    }
    dg.artifacts.write_json(manifest, manifest_path)
    print(
        "Completed Figure 4 anatomy analysis: "
        f"{eligible_regions.height} coverage-eligible regions; "
        f"{nominated.height} conservative nominations",
        flush=True,
    )


def _read_session_scalar_anatomy(
    source: str,
    session_units: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    table_indices = session_units.get_column("_table_index").cast(pl.Int64).to_list()
    peak_channels = session_units.get_column("peak_channel_id").drop_nulls().unique().to_list()
    with _forbid_legacy_accessors():
        labels = dg.lazynwb_obstore.read_vlen_string_column(
            source,
            dg.data.UNITS_PATH,
            "structure_layer",
            row_indices=table_indices,
        )
        electrodes = (
            lazynwb.scan_nwb(
                source,
                dg.data.ELECTRODES_PATH,
                raise_on_missing=True,
                disable_progress=True,
            )
            .filter(pl.col("id").is_in(peak_channels))
            .select(*ELECTRODE_COLUMNS)
            .collect()
        )
    if labels.height != session_units.height:
        raise RuntimeError("structure_layer read did not cover every unit in the session")
    if set(labels.get_column("_table_index")) != set(table_indices):
        raise RuntimeError("structure_layer row identity differs from the Figure 2 unit table")
    if electrodes.get_column("id").n_unique() != len(peak_channels):
        raise RuntimeError("not every unit peak channel matched one electrode")
    return labels, electrodes


@contextlib.contextmanager
def _forbid_legacy_accessors():
    original_get_accessor = lazynwb.file_io._get_accessor
    original_open_hdf5 = getattr(lazynwb.file_io, "_open_hdf5", None)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "legacy NWB accessor invoked; Figure 4 requires lazynwb/obstore custom reads"
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


def _validate_discovery_inputs(
    unit_qc: pl.DataFrame,
    discovery_sources: pl.DataFrame,
) -> None:
    required_units = {
        "_nwb_path",
        "_table_index",
        "subject_id",
        "ecephys_session_id",
        "peak_channel_id",
        "well_isolated",
        "axis_weight",
    }
    required_sources = {"_nwb_path", "subject_id", "ecephys_session_id"}
    if missing := required_units.difference(unit_qc.columns):
        raise ValueError(f"unit table lacks columns: {sorted(missing)}")
    if missing := required_sources.difference(discovery_sources.columns):
        raise ValueError(f"discovery source table lacks columns: {sorted(missing)}")
    if unit_qc.select("_nwb_path", "_table_index").n_unique() != unit_qc.height:
        raise ValueError("unit table keys are not unique")
    expected = set(discovery_sources.get_column("_nwb_path"))
    observed = set(unit_qc.get_column("_nwb_path"))
    if observed != expected:
        raise RuntimeError("Figure 2 unit table does not exactly cover discovery sources")
    crosswalk = unit_qc.select("_nwb_path", "subject_id", "ecephys_session_id").unique()
    if crosswalk.height != discovery_sources.height:
        raise RuntimeError("unit-table session metadata is not one-to-one")
    disagreement = crosswalk.join(
        discovery_sources.select("_nwb_path", "subject_id", "ecephys_session_id"),
        on="_nwb_path",
        how="inner",
        suffix="_source",
        validate="1:1",
    ).filter(
        (pl.col("subject_id") != pl.col("subject_id_source"))
        | (pl.col("ecephys_session_id") != pl.col("ecephys_session_id_source"))
    )
    if disagreement.height:
        raise RuntimeError("unit and discovery-source session identities disagree")


def _validate_parent_contract(divisions: pl.DataFrame) -> None:
    expected = dg.anatomical_enrichment.parent_region_contract().select(
        "parent_region_id",
        "parent_region_acronym",
        "parent_region_name",
    )
    observed = divisions.select(
        pl.col("major_division_id").alias("parent_region_id"),
        pl.col("major_division_acronym").alias("parent_region_acronym"),
        pl.col("major_division_name").alias("parent_region_name"),
    )
    if not expected.sort("parent_region_id").equals(observed.sort("parent_region_id")):
        raise RuntimeError("loaded Allen ontology differs from the frozen D09 parent contract")


def _validated_neural_manifest(
    path: pathlib.Path,
    *,
    results_root: pathlib.Path,
    unit_path: pathlib.Path,
    source_path: pathlib.Path,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "analysis_id": "neural_transition_axis",
        "analysis_tier": "discovery",
        "run_status": "complete",
        "authoritative": True,
        "confirmation_accessed": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"upstream neural manifest has invalid {key!r}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("upstream neural manifest lacks outputs")
    for key, expected_path in (
        ("neural_transition_unit_qc", unit_path),
        ("discovery_session_sources", source_path),
    ):
        record = outputs.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"upstream neural manifest lacks {key}")
        actual = _file_record(expected_path, base=results_root)
        if any(record.get(field) != actual[field] for field in ("path", "sha256", "size_bytes")):
            raise RuntimeError(f"{key} changed after the upstream neural manifest was written")
    return manifest


def _file_record(path: pathlib.Path, *, base: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(base.resolve())),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _code_version(repository_root: pathlib.Path) -> str:
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+dirty" if dirty else commit


if __name__ == "__main__":
    main()
