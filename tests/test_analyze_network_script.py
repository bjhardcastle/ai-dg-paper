"""Orchestration contracts for the discovery-only Figure 5 analysis."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import polars as pl
import pytest

import dg.statistics

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "12_analyze_network.py"
SPEC = importlib.util.spec_from_file_location("dg_analyze_network_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ANALYZE_NETWORK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE_NETWORK
SPEC.loader.exec_module(ANALYZE_NETWORK)


def test_parse_arguments_has_bounded_deterministic_defaults(tmp_path: pathlib.Path) -> None:
    arguments = ANALYZE_NETWORK.parse_arguments(["--results-root", str(tmp_path)])

    assert arguments.results_root == tmp_path
    assert arguments.seed == 1051
    assert arguments.bootstrap_resamples == 10_000
    assert arguments.sign_flip_resamples == 100_000
    assert arguments.recompute_session_cache is False
    assert ANALYZE_NETWORK.SPIKE_BATCH_SIZE == 32


def test_discovery_manifest_must_keep_confirmation_sealed(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    ANALYZE_NETWORK._validate_discovery_manifest(path)

    path.write_text(
        json.dumps(
            {
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="confirmation_accessed"):
        ANALYZE_NETWORK._validate_discovery_manifest(path)


def test_finalize_anatomy_excludes_non_neural_labels_without_dropping_units() -> None:
    frame = pl.DataFrame(
        {
            "_nwb_path": ["s"] * 3,
            "subject_id": ["m"] * 3,
            "ecephys_session_id": [1] * 3,
            "_table_index": [0, 1, 2],
            "id": [10, 11, 12],
            "well_isolated": [True] * 3,
            "presence_ratio": [1.0] * 3,
            "firing_rate": [2.0] * 3,
            "source_region_label": [" VISp ", "root", None],
        }
    )

    result = ANALYZE_NETWORK._finalize_unit_anatomy(frame, region_level="fixture")

    assert result.height == 3
    assert result.get_column("network_region").to_list() == ["VISp", None, None]
    assert result.get_column("network_region_level").unique().item() == "fixture"


def test_external_parent_anatomy_accepts_matched_null_region_without_nwb_fallback(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    anatomy_path = tmp_path / "neural_unit_anatomy.parquet"
    pl.DataFrame(
        {
            "_nwb_path": ["session", "session"],
            "_table_index": [0, 1],
            "parent_region_acronym": ["Isocortex", None],
            "confirmation_accessed": [False, False],
        }
    ).write_parquet(anatomy_path)
    sources = pl.DataFrame(
        {"_nwb_path": ["session"], "subject_id": ["mouse"], "ecephys_session_id": [1]}
    )
    unit_qc = pl.DataFrame(
        {
            "_nwb_path": ["session", "session"],
            "subject_id": ["mouse", "mouse"],
            "ecephys_session_id": [1, 1],
            "_table_index": [0, 1],
            "id": [10, 11],
            "well_isolated": [True, True],
            "presence_ratio": [0.9, 0.8],
            "firing_rate": [2.0, 1.0],
        }
    )

    def forbidden_fallback(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NWB fallback should not run for complete external anatomy")

    monkeypatch.setattr(
        ANALYZE_NETWORK.dg.lazynwb_obstore,
        "read_vlen_string_column",
        forbidden_fallback,
    )
    result, source = ANALYZE_NETWORK._load_or_read_unit_anatomy(
        sources,
        unit_qc,
        anatomy_path=anatomy_path,
    )

    assert source.endswith(":parent_region_acronym")
    assert result.get_column("network_region").to_list() == ["Isocortex", None]
    assert result.height == 2


def _session_trials() -> pl.DataFrame:
    rows = []
    table_index = 0
    for block in ("engaged_1", "no_reward", "engaged_2"):
        for trial_index in range(16):
            rows.append(
                {
                    "_nwb_path": "session",
                    "subject_id": "mouse",
                    "ecephys_session_id": 11,
                    "_table_index": table_index,
                    "change_time": float(table_index),
                    "change_image_name": f"im{trial_index % 2:03d}_r-1.0",
                    "response_in_window": bool(trial_index % 2),
                    "response_latency_from_licks": 0.5 if trial_index % 2 else None,
                    "reward_block": block,
                }
            )
            table_index += 1
    return pl.DataFrame(rows)


def _session_units() -> pl.DataFrame:
    rows = []
    for region_index, region in enumerate(("VISp", "MOs")):
        for unit_index in range(8):
            rows.append(
                {
                    "_nwb_path": "session",
                    "subject_id": "mouse",
                    "ecephys_session_id": 11,
                    "_table_index": region_index * 100 + unit_index,
                    "id": region_index * 100 + unit_index,
                    "well_isolated": True,
                    "presence_ratio": 0.99 - unit_index * 0.01,
                    "firing_rate": 5.0,
                    "source_region_label": region,
                    "network_region": region,
                    "network_region_level": "fixture",
                    "network_region_status": "included",
                }
            )
    return pl.DataFrame(rows)


def test_analyze_session_reduces_once_then_fits_ordered_pair(monkeypatch) -> None:
    generator = np.random.default_rng(1051)
    early = generator.normal(size=(48, 16))
    late = generator.normal(size=(48, 16))
    calls = []

    def fake_features(source, row_indices, event_times):
        calls.append((source, tuple(row_indices), event_times.size))
        return early, late

    monkeypatch.setattr(ANALYZE_NETWORK, "_build_session_features", fake_features)
    block, folds, status = ANALYZE_NETWORK._analyze_session(
        {"_nwb_path": "session", "subject_id": "mouse", "ecephys_session_id": 11},
        _session_trials(),
        _session_units(),
        pl.DataFrame({"source_region": ["VISp"], "target_region": ["MOs"]}),
    )

    assert calls == [("session", tuple(list(range(8)) + list(range(100, 108))), 48)]
    assert block.height == 3
    assert folds.height == 12
    assert status.get_column("status").to_list() == ["pass"]
    assert set(block.get_column("source_region")) == {"VISp"}
    assert set(block.get_column("target_region")) == {"MOs"}


def test_session_cache_round_trip_is_hash_bound(tmp_path: pathlib.Path) -> None:
    frames = (
        pl.DataFrame({"reward_block": ["engaged_1"]}),
        pl.DataFrame({"outer_fold": [0]}),
        pl.DataFrame({"status": ["pass"]}),
    )
    ANALYZE_NETWORK._write_session_cache(
        tmp_path,
        source="session",
        analysis_signature="signature",
        block_performance=frames[0],
        fold_performance=frames[1],
        session_status=frames[2],
    )

    cached = ANALYZE_NETWORK._read_session_cache(
        tmp_path,
        source="session",
        analysis_signature="signature",
    )
    assert cached is not None
    assert [frame.to_dicts() for frame in cached] == [frame.to_dicts() for frame in frames]
    (tmp_path / "block_performance.parquet").write_bytes(b"changed")
    assert (
        ANALYZE_NETWORK._read_session_cache(
            tmp_path,
            source="session",
            analysis_signature="signature",
        )
        is None
    )


def test_network_statistics_use_canonical_contract() -> None:
    screen = pl.DataFrame(
        {
            "source_region": ["VISp"],
            "target_region": ["MOs"],
            "estimate": [0.02],
            "ci_low": [0.01],
            "ci_high": [0.03],
            "p_value": [0.03125],
            "adjusted_p_value": [0.03125],
            "test_method": ["exact sign flip"],
            "n_mice": [5],
            "n_sessions": [10],
            "n_trials": [600],
            "nominated": [True],
            "status": ["pass"],
            "reason": ["discovery_screen_selection_biased"],
        }
    )
    rows = []
    for session_index in range(10):
        for block in ("engaged_1", "no_reward", "engaged_2"):
            rows.append(
                {
                    "_nwb_path": f"s{session_index}",
                    "subject_id": f"m{session_index // 2}",
                    "source_region": "VISp",
                    "target_region": "MOs",
                    "reward_block": block,
                    "n_source_units": 8,
                    "n_target_units": 8,
                    "n_trials": 20,
                }
            )
    statistics = ANALYZE_NETWORK._build_statistics(
        screen,
        block_performance=pl.DataFrame(rows),
        seed=1051,
        code_version="test",
    )

    dg.statistics.validate_statistics_table(statistics)
    assert statistics.item(0, "n_mice") == 5
    assert statistics.item(0, "n_sessions") == 10
    assert statistics.item(0, "n_units") == 160
    assert statistics.item(0, "n_trials") == 600


def test_source_has_no_h5py_or_legacy_data_path() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import h5py" not in source
    assert "map_indexed_numeric_column_batches" in source
    assert "read_vlen_string_column" in source
    assert 'legacy_accessor_calls": 0' in source
