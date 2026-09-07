"""Provenance tests for the main behavior-figure script."""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import types

import polars as pl
import pytest

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "02_plot_behavior.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_behavior_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_BEHAVIOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_BEHAVIOR
SPEC.loader.exec_module(PLOT_BEHAVIOR)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    inputs = {name: tmp_path / name for name in PLOT_BEHAVIOR.INPUT_FILENAMES}
    for index, path in enumerate(inputs.values()):
        path.write_bytes(f"fixture-{index}".encode())
    return inputs


def _write_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
    *,
    status: str = "pass",
    authoritative: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "analysis_id": "figure_1_behavior",
                "analysis_status": status,
                "analysis_tier": "exploratory",
                "run_status": "complete",
                "authoritative": authoritative,
                "behavior_trials_checkpoint": {
                    "output_matches_trusted_checkpoint": True,
                },
                "outputs": {
                    pathlib.Path(filename).stem: {
                        "path": str(input_path),
                        "sha256": _sha256(input_path),
                        "size_bytes": input_path.stat().st_size,
                    }
                    for filename, input_path in inputs.items()
                },
            }
        ),
        encoding="utf-8",
    )


def _install_successful_plot_fixture(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[pathlib.Path, pathlib.Path]:
    results_root = tmp_path / "results"
    table_root = results_root / "tables"
    inputs = {name: table_root / name for name in PLOT_BEHAVIOR.INPUT_FILENAMES}
    for index, path in enumerate(inputs.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture-{index}".encode())
    behavior_manifest_path = results_root / "manifests" / "behavior_analysis_run.json"
    behavior_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_manifest_path.write_text('{"fixture": true}', encoding="utf-8")
    prepared = types.SimpleNamespace(
        session_timing=pl.DataFrame({"value": [1]}),
        timing_summary=pl.DataFrame({"value": [1]}),
        trajectories=pl.DataFrame({"value": [1]}),
        gating=pl.DataFrame({"value": [1]}),
        attrition=pl.DataFrame({"value": [1]}),
        gating_summary_statistic=pl.DataFrame(
            {
                "dandiset_version": ["test-version"],
                "code_version": ["test-code"],
                "analysis_tier": ["exploratory"],
                "status": ["summary_only"],
                "reason": ["fixture"],
            }
        ),
    )
    monkeypatch.setattr(
        PLOT_BEHAVIOR,
        "parse_arguments",
        lambda: argparse.Namespace(results_root=results_root, dpi=72),
    )
    monkeypatch.setattr(
        PLOT_BEHAVIOR,
        "_validated_behavior_manifest",
        lambda path, input_paths: {
            "run_status": "complete",
            "path": str(path),
            "n_inputs": len(input_paths),
        },
    )
    monkeypatch.setattr(PLOT_BEHAVIOR.pl, "read_csv", lambda path: pl.DataFrame())
    monkeypatch.setattr(
        PLOT_BEHAVIOR.dg.figure_behavior,
        "prepare_behavior_figure_data",
        lambda *frames: prepared,
    )
    monkeypatch.setattr(PLOT_BEHAVIOR, "create_figure", lambda value: object())

    def save_fixture_figure(
        figure,
        destination: pathlib.Path,
        *,
        file_format: str,
        dpi: int,
    ) -> None:
        del figure, dpi
        destination.write_bytes(f"fixture-{file_format}".encode())

    monkeypatch.setattr(PLOT_BEHAVIOR, "_save_figure_atomic", save_fixture_figure)
    monkeypatch.setattr(PLOT_BEHAVIOR.importlib.metadata, "version", lambda package: "test")
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.close = lambda figure: None
    matplotlib.pyplot = pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    manifest_path = results_root / "figures" / "main" / "figure_1_behavior_manifest.json"
    return behavior_manifest_path, manifest_path


def test_figure_requires_content_addressed_behavior_inputs(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    manifest_path = tmp_path / "behavior_analysis_run.json"
    _write_manifest(manifest_path, inputs)

    manifest = PLOT_BEHAVIOR._validated_behavior_manifest(manifest_path, inputs)
    assert manifest["authoritative"] is True

    inputs["behavior_statistics.csv"].write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after its run"):
        PLOT_BEHAVIOR._validated_behavior_manifest(manifest_path, inputs)


@pytest.mark.parametrize(
    ("status", "authoritative"),
    (("failed", True), ("pass", False)),
)
def test_figure_rejects_nonpassing_or_nonauthoritative_behavior(
    tmp_path: pathlib.Path,
    status: str,
    authoritative: bool,
) -> None:
    inputs = _write_inputs(tmp_path)
    manifest_path = tmp_path / "behavior_analysis_run.json"
    _write_manifest(
        manifest_path,
        inputs,
        status=status,
        authoritative=authoritative,
    )

    with pytest.raises(RuntimeError, match="not a completed authoritative"):
        PLOT_BEHAVIOR._validated_behavior_manifest(manifest_path, inputs)


def test_main_replaces_stale_success_with_failed_marker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    manifest = results_root / "figures" / "main" / "figure_1_behavior_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"run_status": "complete", "authoritative": true}')
    monkeypatch.setattr(
        PLOT_BEHAVIOR,
        "parse_arguments",
        lambda: argparse.Namespace(results_root=results_root, dpi=400),
    )

    def fail_after_sentinel(*args, **kwargs) -> None:
        del args, kwargs
        marker = json.loads(manifest.read_text(encoding="utf-8"))
        assert marker["run_status"] == "in_progress"
        assert marker["authoritative"] is False
        raise KeyboardInterrupt("intentional figure interruption")

    monkeypatch.setattr(PLOT_BEHAVIOR, "_run_figure", fail_after_sentinel)

    with pytest.raises(KeyboardInterrupt, match="intentional figure interruption"):
        PLOT_BEHAVIOR.main()

    marker = json.loads(manifest.read_text(encoding="utf-8"))
    assert marker["run_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "KeyboardInterrupt"


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("behavior_manifest", "behavior-analysis manifest changed"),
        ("input", "behavior input changed"),
        ("local_source", "local Figure 1 source changed"),
    ],
)
def test_end_of_run_snapshot_rejects_parent_input_and_source_races(
    tmp_path: pathlib.Path,
    changed: str,
    message: str,
) -> None:
    result_root = tmp_path / "results"
    behavior_manifest_path = result_root / "manifests" / "behavior_analysis_run.json"
    behavior_manifest_path.parent.mkdir(parents=True)
    behavior_manifest_path.write_bytes(b"parent")
    inputs = {"one.csv": result_root / "tables" / "one.csv"}
    inputs["one.csv"].parent.mkdir(parents=True)
    inputs["one.csv"].write_bytes(b"input")
    local_sources = (tmp_path / "figure_source.py",)
    local_sources[0].write_bytes(b"source")
    parent_record = PLOT_BEHAVIOR._file_record(
        behavior_manifest_path,
        base=result_root,
    )
    input_records = PLOT_BEHAVIOR._file_records(inputs.values(), base=result_root)
    local_source_records = PLOT_BEHAVIOR._file_records(
        local_sources,
        base=PLOT_BEHAVIOR.REPOSITORY_ROOT,
    )
    target = {
        "behavior_manifest": behavior_manifest_path,
        "input": inputs["one.csv"],
        "local_source": local_sources[0],
    }[changed]
    target.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match=message):
        PLOT_BEHAVIOR._require_unchanged_figure_inputs(
            behavior_manifest_path=behavior_manifest_path,
            behavior_manifest_record=parent_record,
            inputs=inputs,
            input_records=input_records,
            local_sources=local_sources,
            local_source_records=local_source_records,
            result_root=result_root,
        )


def test_successful_main_publishes_complete_authoritative_marker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest_path = _install_successful_plot_fixture(tmp_path, monkeypatch)

    PLOT_BEHAVIOR.main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "complete"
    assert manifest["authoritative"] is True
    assert manifest["started_at_utc"] <= manifest["completed_at_utc"]
    assert manifest["generated_at_utc"] == manifest["completed_at_utc"]
    assert len(manifest["local_sources"]) == 3


def test_parent_race_after_render_replaces_in_progress_marker_with_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    behavior_manifest_path, figure_manifest_path = _install_successful_plot_fixture(
        tmp_path,
        monkeypatch,
    )
    original_check = PLOT_BEHAVIOR._require_unchanged_figure_inputs

    def change_parent_before_check(**kwargs) -> None:
        behavior_manifest_path.write_bytes(b"changed during rendering")
        original_check(**kwargs)

    monkeypatch.setattr(
        PLOT_BEHAVIOR,
        "_require_unchanged_figure_inputs",
        change_parent_before_check,
    )

    with pytest.raises(RuntimeError, match="manifest changed"):
        PLOT_BEHAVIOR.main()

    marker = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    assert marker["run_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "RuntimeError"
