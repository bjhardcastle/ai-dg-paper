"""Regression tests for authoritative Milestone 0 run-manifest behavior."""

import argparse
import hashlib
import importlib.util
import json
import pathlib

import polars as pl
import pytest


def _load_script_module():
    script = pathlib.Path(__file__).parents[1] / "scripts" / "00_freeze_and_audit.py"
    spec = importlib.util.spec_from_file_location("freeze_and_audit_script", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_rerun_replaces_stale_pass_with_non_authoritative_marker(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    manifests = tmp_path / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "audit_run.json").write_text(
        '{"run_status":"complete","authoritative":true}\n',
        encoding="utf-8",
    )
    arguments = argparse.Namespace(results_root=tmp_path)
    monkeypatch.setattr(module, "parse_arguments", lambda: arguments)

    def fail(*args, **kwargs):
        marker = json.loads((manifests / "audit_run.json").read_text(encoding="utf-8"))
        assert marker["run_status"] == "in_progress"
        assert marker["authoritative"] is False
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(module, "_run_audit", fail)

    with pytest.raises(RuntimeError, match="fixture failure"):
        module.main()

    marker = json.loads((manifests / "audit_run.json").read_text(encoding="utf-8"))
    assert marker["manifest_schema_version"] == 3
    assert marker["run_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "RuntimeError"


def test_sigterm_routes_through_failed_manifest_and_restores_handler(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_script_module()
    arguments = argparse.Namespace(results_root=tmp_path)
    monkeypatch.setattr(module, "parse_arguments", lambda: arguments)
    current_handler = {"value": "previous-handler"}
    installed_handlers = []

    def install_handler(signum, handler):
        assert signum == module.signal.SIGTERM
        previous = current_handler["value"]
        current_handler["value"] = handler
        installed_handlers.append(handler)
        return previous

    def terminate(*args, **kwargs):
        del args, kwargs
        current_handler["value"](module.signal.SIGTERM, None)

    monkeypatch.setattr(module.signal, "signal", install_handler)
    monkeypatch.setattr(module, "_run_audit", terminate)

    with pytest.raises(SystemExit, match="termination signal"):
        module.main()

    marker = json.loads((tmp_path / "manifests" / "audit_run.json").read_text(encoding="utf-8"))
    assert marker["run_status"] == "failed"
    assert marker["authoritative"] is False
    assert marker["error_type"] == "SystemExit"
    assert installed_handlers == [
        module._raise_system_exit_on_sigterm,
        "previous-handler",
    ]


def test_artifact_record_hashes_root_relative_stable_file(tmp_path) -> None:
    module = _load_script_module()
    artifact = tmp_path / "table.csv"
    payload = b"a,b\n1,2\n"
    artifact.write_bytes(payload)

    record = module._artifact_record(artifact, root=tmp_path, n_rows=1)

    assert record == {
        "path": "table.csv",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "n_rows": 1,
    }


def test_all_declared_local_sources_exist() -> None:
    module = _load_script_module()

    assert all((module.REPOSITORY_ROOT / path).is_file() for path in module.LOCAL_SOURCE_PATHS)


@pytest.mark.parametrize(
    ("skip_schema_audit", "audit_nwb_root_metadata", "authoritative"),
    [
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ],
)
def test_only_full_default_audit_mode_can_be_authoritative(
    skip_schema_audit,
    audit_nwb_root_metadata,
    authoritative,
) -> None:
    module = _load_script_module()
    arguments = argparse.Namespace(
        skip_schema_audit=skip_schema_audit,
        audit_nwb_root_metadata=audit_nwb_root_metadata,
    )

    assert module._is_authoritative_mode(arguments) is authoritative


def test_frame_output_publication_requires_exact_frame_destination_pairs(tmp_path) -> None:
    module = _load_script_module()
    destination = tmp_path / "probe.csv"
    artifact_specs = {}

    module._publish_frame_outputs(
        {"probe": (pl.DataFrame({"value": [1, 2]}), destination)},
        artifact_specs,
    )

    assert pl.read_csv(destination).get_column("value").to_list() == [1, 2]
    assert artifact_specs == {"probe": (destination, 2)}
    with pytest.raises(ValueError, match="must be a .* tuple"):
        module._publish_frame_outputs(
            {"broken": (pl.DataFrame({"value": [1]}), pl.DataFrame(), destination)},
            {},
        )
