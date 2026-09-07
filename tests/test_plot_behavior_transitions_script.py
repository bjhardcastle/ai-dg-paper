"""Provenance tests for the behavior-transition figure script."""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "05_plot_behavior_transitions.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_behavior_transitions_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_TRANSITIONS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_TRANSITIONS
SPEC.loader.exec_module(PLOT_TRANSITIONS)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    path: pathlib.Path,
    inputs: dict[str, pathlib.Path],
    *,
    status: str = "pass",
) -> None:
    path.write_text(
        json.dumps(
            {
                "analysis_id": "behavior_transition_qc",
                "analysis_status": status,
                "analysis_tier": "exploratory",
                "run_status": "complete",
                "authoritative": True,
                "dandiset_version": "0.260825.2232",
                "code_version": "test",
                "outputs": {
                    key: {
                        "path": str(input_path),
                        "sha256": _sha256(input_path),
                        "size_bytes": input_path.stat().st_size,
                    }
                    for key, input_path in inputs.items()
                },
            }
        ),
        encoding="utf-8",
    )


def test_transition_manifest_binds_all_figure_inputs(tmp_path: pathlib.Path) -> None:
    inputs = {
        key: tmp_path / pathlib.Path(relative).name
        for key, relative in PLOT_TRANSITIONS.INPUT_KEYS.items()
    }
    for index, path in enumerate(inputs.values()):
        path.write_bytes(f"input-{index}".encode())
    manifest = tmp_path / "behavior_transition_analysis_run.json"
    _write_manifest(manifest, inputs)

    record = PLOT_TRANSITIONS._validated_transition_manifest(manifest, inputs)
    assert record["analysis_status"] == "pass"

    next(iter(inputs.values())).write_bytes(b"changed after analysis")
    with pytest.raises(RuntimeError, match="changed after its run"):
        PLOT_TRANSITIONS._validated_transition_manifest(manifest, inputs)


def test_transition_manifest_rejects_nonpassing_analysis(tmp_path: pathlib.Path) -> None:
    inputs = {
        key: tmp_path / pathlib.Path(relative).name
        for key, relative in PLOT_TRANSITIONS.INPUT_KEYS.items()
    }
    for path in inputs.values():
        path.write_bytes(b"input")
    manifest = tmp_path / "behavior_transition_analysis_run.json"
    _write_manifest(manifest, inputs, status="failed")

    with pytest.raises(RuntimeError, match="not a completed authoritative"):
        PLOT_TRANSITIONS._validated_transition_manifest(manifest, inputs)


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0.02, "0.020"), (0.00099, "<0.001")),
)
def test_adjusted_p_format(value: float, expected: str) -> None:
    assert PLOT_TRANSITIONS._format_adjusted_p(value) == expected


def test_main_replaces_stale_success_with_failed_marker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    manifest = (
        results_root / "figures" / "supplementary" / "figure_s1_behavior_transitions_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"run_status": "complete", "authoritative": true}')
    monkeypatch.setattr(
        PLOT_TRANSITIONS,
        "parse_arguments",
        lambda: argparse.Namespace(results_root=results_root, dpi=400),
    )

    def fail_after_sentinel(*args, **kwargs) -> None:
        del args, kwargs
        marker = json.loads(manifest.read_text())
        assert marker["run_status"] == "in_progress"
        assert marker["authoritative"] is False
        raise RuntimeError("intentional figure failure")

    monkeypatch.setattr(PLOT_TRANSITIONS, "_run_figure", fail_after_sentinel)

    with pytest.raises(RuntimeError, match="intentional figure failure"):
        PLOT_TRANSITIONS.main()

    marker = json.loads(manifest.read_text())
    assert marker["run_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "RuntimeError"
