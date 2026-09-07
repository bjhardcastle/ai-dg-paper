import importlib.util
import json
import pathlib

import polars as pl
import pytest


@pytest.fixture
def analysis_module():
    path = pathlib.Path(__file__).parents[1] / "scripts" / "08_analyze_functional_populations.py"
    spec = importlib.util.spec_from_file_location("analyze_functional_populations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_discovery_sources_keeps_only_behavior_eligible_discovery(
    analysis_module,
) -> None:
    analysis_module.EXPECTED_DISCOVERY_SESSIONS = 2
    analysis_module.EXPECTED_DISCOVERY_MICE = 1
    assignments = pl.DataFrame(
        {
            "_nwb_path": ["d1", "d2", "c1", "bad"],
            "subject_id": ["1", "1", "2", "3"],
            "cohort_assignment": ["discovery", "discovery", "confirmation", "discovery"],
            "session_behavior_eligible": [True, True, True, False],
            "neural_activity_used_for_allocation": [False, False, False, False],
            "ecephys_session_id": [1, 2, 3, 4],
        }
    )

    selected = analysis_module._select_discovery_sources(assignments)

    assert selected.get_column("_nwb_path").to_list() == ["d1", "d2"]
    assert "c1" not in selected.get_column("_nwb_path").to_list()


def test_validate_upstream_manifest_requires_sealed_discovery(
    analysis_module, tmp_path: pathlib.Path
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "run_status": "complete",
                "analysis_tier": "discovery",
                "confirmation_accessed": False,
            }
        )
    )
    analysis_module._validate_upstream_manifest(path)

    path.write_text(
        json.dumps(
            {
                "run_status": "complete",
                "analysis_tier": "discovery",
                "confirmation_accessed": True,
            }
        )
    )
    with pytest.raises(RuntimeError, match="confirmation sealed"):
        analysis_module._validate_upstream_manifest(path)


def test_read_running_uses_lazynwb_custom_table_scan(analysis_module, monkeypatch) -> None:
    calls = []

    class FakeScan:
        def select(self, *columns):
            calls.append(("select", columns))
            return self

        def collect(self):
            return pl.DataFrame({"data": [1.0, 2.0], "timestamps": [0.0, 0.1]})

    def fake_scan(source, table_path, **kwargs):
        calls.append((source, table_path, kwargs))
        return FakeScan()

    monkeypatch.setattr(analysis_module.lazynwb, "scan_nwb", fake_scan)

    result = analysis_module._read_running("s3://session.nwb")

    assert result.shape == (2, 2)
    assert calls[0][1] == "/processing/running/speed"
    assert calls[0][2]["raise_on_missing"] is True
    assert calls[1] == ("select", ("data", "timestamps"))


def test_aggregate_effects_preserves_session_then_mouse_hierarchy(analysis_module) -> None:
    units = pl.DataFrame(
        {
            "_nwb_path": ["a", "a", "b"],
            "subject_id": ["1", "1", "1"],
            "ecephys_session_id": [1, 1, 2],
            "class_assignment_stable": [True, False, True],
            "early_raw_reversible": [1.0, 3.0, 5.0],
            "late_prelick_raw_reversible": [2.0, 6.0, 9.0],
            "early_total_reversible": [1.0, 3.0, 5.0],
            "late_total_reversible": [3.0, 7.0, 10.0],
            "early_adjusted_reversible": [0.5, 2.5, 4.0],
            "late_adjusted_reversible": [2.0, 6.0, 8.0],
            "early_total_state_dropout_r2": [0.01, 0.02, 0.03],
            "late_total_state_dropout_r2": [0.04, 0.05, 0.06],
            "early_adjusted_state_dropout_r2": [0.005, 0.01, 0.02],
            "late_adjusted_state_dropout_r2": [0.03, 0.04, 0.05],
        }
    )
    trials = pl.DataFrame(
        {
            "_nwb_path": ["a", "a", "b", "b"],
            "response_in_window": [True, False, True, False],
        }
    )

    sessions = analysis_module._aggregate_session_effects(units, trials)
    mice = analysis_module._aggregate_mouse_effects(sessions)

    assert sessions.height == 2
    assert mice.height == 1
    assert sessions.filter(pl.col("_nwb_path") == "a")["early_total"][0] == 2.0
    assert mice["early_total"][0] == pytest.approx(3.5)
    assert mice["late_minus_early_total"][0] == pytest.approx((3.0 + 5.0) / 2)


def test_compatible_legacy_cache_is_reused_without_running_reread(
    analysis_module, tmp_path: pathlib.Path
) -> None:
    cache = analysis_module._session_cache_paths(tmp_path / "cache", 123)
    cache["root"].mkdir(parents=True)
    profile_path = tmp_path / "all_units" / "functional_population_profiles" / "session_123.parquet"
    profile_path.parent.mkdir(parents=True)
    units = pl.DataFrame(
        {
            "_nwb_path": ["source"],
            "subject_id": ["mouse"],
            "ecephys_session_id": [123],
            "_table_index": [0],
        }
    )
    summary = pl.DataFrame({"value": [1.0]})
    profile = pl.DataFrame({"value": [2.0]})
    units.write_parquet(cache["units"])
    summary.write_parquet(cache["summary"])
    profile.write_parquet(profile_path)
    old_signature = next(iter(analysis_module.COMPATIBLE_SESSION_CACHE_SIGNATURES))
    cache["manifest"].write_text(
        json.dumps(
            {
                "cache_schema_version": analysis_module.CACHE_SCHEMA_VERSION,
                "analysis_signature": old_signature,
                "files": {
                    "units": {"sha256": analysis_module._sha256(cache["units"])},
                    "summary": {"sha256": analysis_module._sha256(cache["summary"])},
                    "profiles": {"sha256": analysis_module._sha256(profile_path)},
                },
            }
        )
    )

    cached = analysis_module._read_session_cache(
        cache,
        signature="new-signature",
        profile_path=profile_path,
    )

    assert cached is not None
    cached_units, cached_summary, running_qc = cached
    assert cached_units.equals(units)
    assert cached_summary.equals(summary)
    assert running_qc["movement_adjustment_available"][0] is True
    assert running_qc["running_clock_status"][0] == "pass_strict_validation_from_compatible_cache"
