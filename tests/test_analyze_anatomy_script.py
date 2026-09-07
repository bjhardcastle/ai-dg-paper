"""I/O boundary contracts for the Figure 4 anatomy analysis script."""

import hashlib
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "10_analyze_anatomy.py"
SPEC = importlib.util.spec_from_file_location("dg_analyze_anatomy_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ANALYZE_ANATOMY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE_ANATOMY
SPEC.loader.exec_module(ANALYZE_ANATOMY)


def _unit_rows(source: str = "fixture.nwb") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": [source, source],
            "_table_index": pl.Series([0, 2], dtype=pl.UInt32),
            "id": [10, 12],
            "peak_channel_id": [100, 102],
            "subject_id": ["1", "1"],
            "ecephys_session_id": [1000, 1000],
            "well_isolated": [True, False],
            "axis_weight": [0.2, None],
            "unit_inclusion_status": ["included", "excluded"],
            "engaged_block_consistency_status": ["not_applied", "not_applied"],
        }
    )


def test_scalar_reader_requests_only_unit_labels_and_peak_electrodes(monkeypatch) -> None:
    source = "fixture.nwb"
    requested = {}

    def fake_strings(path, table_path, column_name, *, row_indices):
        requested["strings"] = (path, table_path, column_name, row_indices)
        return pl.DataFrame(
            {
                "_nwb_path": [source, source],
                "_table_index": pl.Series([0, 2], dtype=pl.UInt32),
                "structure_layer": ["VISp", "LGd"],
            }
        )

    def fake_scan(path, table_path, **kwargs):
        requested["scan"] = (path, table_path, kwargs)
        return pl.DataFrame(
            {
                "id": [100, 101, 102],
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "z": [7.0, 8.0, 9.0],
                "probe_id": [1, 1, 2],
                "_nwb_path": [source] * 3,
            }
        ).lazy()

    monkeypatch.setattr(
        ANALYZE_ANATOMY.dg.lazynwb_obstore,
        "read_vlen_string_column",
        fake_strings,
    )
    monkeypatch.setattr(ANALYZE_ANATOMY.lazynwb, "scan_nwb", fake_scan)

    labels, electrodes = ANALYZE_ANATOMY._read_session_scalar_anatomy(
        source,
        _unit_rows(source),
    )

    assert labels.height == 2
    assert set(electrodes.get_column("id")) == {100, 102}
    assert requested["strings"] == (source, "/units", "structure_layer", [0, 2])
    assert requested["scan"][1] == "/general/extracellular_ephys/electrodes"
    assert set(electrodes.columns) == set(ANALYZE_ANATOMY.ELECTRODE_COLUMNS)


def test_discovery_input_validation_rejects_extra_source() -> None:
    units = _unit_rows()
    sources = pl.DataFrame(
        {
            "_nwb_path": ["fixture.nwb", "confirmation.nwb"],
            "subject_id": ["1", "2"],
            "ecephys_session_id": [1000, 2000],
        }
    )

    with pytest.raises(RuntimeError, match="exactly cover discovery"):
        ANALYZE_ANATOMY._validate_discovery_inputs(units, sources)


def test_upstream_manifest_requires_sealed_discovery_and_exact_hashes(
    tmp_path: pathlib.Path,
) -> None:
    results_root = tmp_path / "results"
    table_root = results_root / "tables"
    table_root.mkdir(parents=True)
    unit_path = table_root / "neural_transition_unit_qc.parquet"
    source_path = table_root / "discovery_session_sources.parquet"
    unit_path.write_text("units", encoding="utf-8")
    source_path.write_text("sources", encoding="utf-8")

    def record(path: pathlib.Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(results_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    manifest_path = results_root / "manifests" / "neural_transition_analysis_run.json"
    manifest_path.parent.mkdir(parents=True)
    payload = {
        "analysis_id": "neural_transition_axis",
        "analysis_tier": "discovery",
        "run_status": "complete",
        "authoritative": True,
        "confirmation_accessed": False,
        "outputs": {
            "neural_transition_unit_qc": record(unit_path),
            "discovery_session_sources": record(source_path),
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        ANALYZE_ANATOMY._validated_neural_manifest(
            manifest_path,
            results_root=results_root,
            unit_path=unit_path,
            source_path=source_path,
        )["confirmation_accessed"]
        is False
    )

    payload["confirmation_accessed"] = True
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="confirmation_accessed"):
        ANALYZE_ANATOMY._validated_neural_manifest(
            manifest_path,
            results_root=results_root,
            unit_path=unit_path,
            source_path=source_path,
        )
