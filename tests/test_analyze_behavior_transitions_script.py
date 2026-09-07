"""Provenance tests for the behavior-transition orchestration script."""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

import dg.data

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "04_analyze_behavior_transitions.py"
SPEC = importlib.util.spec_from_file_location("dg_analyze_behavior_transitions_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ANALYZE_TRANSITIONS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE_TRANSITIONS
SPEC.loader.exec_module(ANALYZE_TRANSITIONS)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_behavior_input_hashes_must_match_parent_manifest(tmp_path: pathlib.Path) -> None:
    trials = tmp_path / "behavior_trials.parquet"
    qc = tmp_path / "session_qc.parquet"
    run = tmp_path / "behavior_analysis_run.json"
    trials.write_bytes(b"audited behavior trials")
    qc.write_bytes(b"audited session qc")
    run_record = {
        "analysis_status": "pass",
        "dandiset_version": dg.data.DANDISET_VERSION,
        "outputs": {
            "behavior_trials": {"sha256": _sha256(trials)},
            "session_qc": {"sha256": _sha256(qc)},
        },
    }
    run_record_text = json.dumps(run_record)
    run_record = json.loads(run_record_text)
    behavior_run = run_record
    # The manifest itself is also part of the provenance chain.
    (tmp_path / "behavior_analysis_run.json").write_text(run_record_text, encoding="utf-8")

    records = ANALYZE_TRANSITIONS._validate_behavior_inputs(
        behavior_run,
        behavior_trials_path=trials,
        session_qc_path=qc,
        behavior_run_path=run,
    )
    assert records["behavior_trials"]["sha256"] == _sha256(trials)

    trials.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="behavior_trials hash differs"):
        ANALYZE_TRANSITIONS._validate_behavior_inputs(
            behavior_run,
            behavior_trials_path=trials,
            session_qc_path=qc,
            behavior_run_path=run,
        )


def test_script_arguments_reject_invalid_resampling_counts() -> None:
    arguments = argparse.Namespace(seed=1051, bootstrap_resamples=0, sign_flip_resamples=100)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        ANALYZE_TRANSITIONS._validate_arguments(arguments)


def test_transition_main_replaces_stale_success_with_failed_marker(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    results_root = tmp_path / "results"
    manifest_path = results_root / "manifests" / "behavior_transition_analysis_run.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"analysis_status":"pass"}', encoding="utf-8")
    arguments = argparse.Namespace(
        results_root=results_root,
        analysis_lock=tmp_path / "lock.json",
        seed=1051,
        bootstrap_resamples=10,
        sign_flip_resamples=10,
    )
    monkeypatch.setattr(ANALYZE_TRANSITIONS, "parse_arguments", lambda: arguments)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(ANALYZE_TRANSITIONS, "_run_analysis", interrupt)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        ANALYZE_TRANSITIONS.main()

    marker = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert marker["analysis_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "KeyboardInterrupt"


def test_locked_dandiset_must_match_immutable_release() -> None:
    lock = {
        "decisions": {
            "D01": {
                "value": f"DANDI:{dg.data.DANDISET_ID}/wrong-version",
            }
        }
    }
    with pytest.raises(RuntimeError, match="D01 locks"):
        ANALYZE_TRANSITIONS._validate_locked_dandiset(lock)


def test_behavior_run_requires_output_matching_trusted_trial_checkpoint(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "behavior_analysis_run.json"
    record = {
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "dandiset_version": dg.data.DANDISET_VERSION,
        "outputs": {
            "behavior_trials": {"sha256": "canonical-sha"},
            "session_qc": {"sha256": "qc-sha"},
        },
        "behavior_trials_checkpoint": {
            "status": "validated_existing_not_refreshed",
            "trusted_behavior_trials_sha256": "canonical-sha",
            "output_matches_trusted_checkpoint": False,
        },
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match"):
        ANALYZE_TRANSITIONS._read_behavior_run(path)

    record["behavior_trials_checkpoint"]["output_matches_trusted_checkpoint"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    assert ANALYZE_TRANSITIONS._read_behavior_run(path) == record

    record["behavior_trials_checkpoint"]["trusted_behavior_trials_sha256"] = "wrong-sha"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint hash differs"):
        ANALYZE_TRANSITIONS._read_behavior_run(path)
