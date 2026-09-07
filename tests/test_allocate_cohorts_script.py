"""Orchestration tests using labeled, non-scientific allocation fixtures.

The fixture rows exercise artifact and gating contracts only. They are not
synthetic neuroscience observations and do not enter repository results.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

import dg.ontology

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "03_allocate_cohorts.py"
SPEC = importlib.util.spec_from_file_location("dg_allocate_cohorts_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
ALLOCATE_COHORTS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ALLOCATE_COHORTS
SPEC.loader.exec_module(ALLOCATE_COHORTS)


def _artifact_record(path: pathlib.Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": ALLOCATE_COHORTS._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_approved_lock(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "overall_status": "approved",
                "approval_record": {
                    "approved_by": "test fixture",
                    "approved_at": "2026-01-01T00:00:00Z",
                },
                "decisions": {
                    f"D{index:02d}": {
                        "title": f"fixture decision {index}",
                        "value": "fixture arithmetic contract",
                        "status": "approved",
                        "approved_by": "test fixture",
                        "approved_at": "2026-01-01T00:00:00Z",
                    }
                    for index in range(1, 16)
                },
            }
        ),
        encoding="utf-8",
    )


def _write_orchestration_fixture(
    results_root: pathlib.Path,
    lock_path: pathlib.Path,
) -> None:
    manifests = results_root / "manifests"
    tables = results_root / "tables"
    cache = results_root / "cache"
    manifests.mkdir(parents=True)
    tables.mkdir(parents=True)
    cache.mkdir(parents=True)
    inventory_rows = []
    qc_rows = []
    coverage_rows = []
    unit_rows = []
    for index in range(19):
        # Numeric-looking, zero-padded IDs exercise CSV/Parquet identity handling.
        subject_id = f"{index + 1:06d}"
        session = f"fixture_session_{index + 1:02d}.nwb"
        inventory_rows.append(
            {
                "_nwb_path": session,
                "subject_id": subject_id,
                "sex": "F" if index % 3 == 0 else "M",
                "genotype": f"fixture_genotype_{index % 3}",
                "project_code": f"fixture_project_{index % 2}",
                "recording_day": f"EPHYS_{index % 4 + 1}",
                "session_number": index % 4 + 1,
                "asset_id": f"fixture_asset_{index + 1:02d}",
                "path": f"sub-{subject_id}/{session}",
                "ecephys_session_id": str(index + 1),
                "dandiset_id": "fixture_dandiset",
                "dandiset_version": "fixture_version",
                "dandiset_doi": "fixture_doi",
            }
        )
        eligible = index < 18
        qc_rows.append(
            {
                "_nwb_path": session,
                "subject_id": subject_id,
                "is_good_session": eligible,
                "session_exclusion_reasons": None if eligible else "fixture_qc_failure",
            }
        )
        coverage_rows.append(
            {
                "subject_id": subject_id,
                "_nwb_path": session,
                "structure_acronym": (
                    "fixture_raw_acronym" if index < 10 else dg.ontology.MISSING_STRUCTURE_ACRONYM
                ),
                "n_units": 1,
                "anatomy_scope": "raw_allen_acronym_no_parent_mapping",
                "analysis_scope": "coverage_only_no_neural_activity",
            }
        )
        unit_rows.append(
            {
                "subject_id": subject_id,
                "_nwb_path": session,
                "unit_id": index + 1,
                "structure_acronym": "fixture_raw_acronym",
            }
        )

    inventory_path = manifests / "session_inventory.parquet"
    session_qc_path = tables / "session_qc.parquet"
    coverage_path = manifests / "unit_session_anatomy_coverage.csv"
    unit_inventory_path = manifests / "unit_inventory.parquet"
    task_parameters_audit_path = manifests / "task_parameters_audit.csv"
    pl.DataFrame(inventory_rows).write_parquet(inventory_path)
    pl.DataFrame(qc_rows).write_parquet(session_qc_path)
    pl.DataFrame(coverage_rows).write_csv(coverage_path)
    pl.DataFrame(unit_rows).write_parquet(unit_inventory_path)
    task_parameters_audit_path.write_text(
        "audit_scope,audit_status\nlabeled_contract_fixture,pass\n",
        encoding="utf-8",
    )

    m0_lock_snapshot = manifests / "analysis_lock.yaml"
    behavior_lock_snapshot = manifests / "behavior_analysis_lock.yaml"
    lock_text = lock_path.read_text(encoding="utf-8")
    m0_lock_snapshot.write_text(lock_text, encoding="utf-8")
    behavior_lock_snapshot.write_text(lock_text, encoding="utf-8")
    lock_sha256 = ALLOCATE_COHORTS._sha256_file(lock_path)
    m0_artifacts = {
        "analysis_lock_snapshot": _artifact_record(m0_lock_snapshot),
        "session_inventory": _artifact_record(inventory_path),
        "task_parameters_audit": _artifact_record(task_parameters_audit_path),
        "unit_inventory": _artifact_record(unit_inventory_path),
        "unit_session_anatomy_coverage": _artifact_record(coverage_path),
    }
    m0_manifest = {
        "manifest_schema_version": 3,
        "run_status": "complete",
        "authoritative": True,
        "milestone_0_status": "partial",
        "dandiset_version": "fixture_version",
        "analysis_lock_status": "approved",
        "asset_inventory_status": "pass",
        "nwb_root_metadata_audit_status": "pass",
        "unit_inventory_status": "pass",
        "unit_metadata_inventory_status": ("pass_scalar_metadata_no_spike_arrays"),
        "neural_outcomes_accessed": False,
        "spike_arrays_loaded": False,
        "artifacts": m0_artifacts,
        "inputs": {
            "labeled_m0_input_fixture": _artifact_record(inventory_path),
        },
        "local_sources": {
            "config/analysis_lock.yaml": _artifact_record(lock_path),
        },
    }
    audit_run_path = manifests / "audit_run.json"
    audit_run_path.write_text(json.dumps(m0_manifest), encoding="utf-8")

    behavior_trials_path = tables / "behavior_trials.parquet"
    pl.DataFrame({"fixture_trial_id": [1]}).write_parquet(behavior_trials_path)
    companion_path = cache / "fixture_companion_trials.csv"
    companion_path.write_text("fixture_trial_id\n1\n", encoding="utf-8")
    checkpoint_path = manifests / "behavior_trials_checkpoint.json"
    trusted_trials = _artifact_record(behavior_trials_path)
    behavior_generator = _artifact_record(
        ALLOCATE_COHORTS.REPOSITORY_ROOT / "scripts" / "01_analyze_behavior.py"
    )
    behavior_sources = [
        _artifact_record(ALLOCATE_COHORTS.REPOSITORY_ROOT / "src" / "dg" / "behavior.py")
    ]
    companion_record = {
        **_artifact_record(companion_path),
        "source_url": "https://example.invalid/labeled-contract-fixture",
        "repository_commit": "labeled-contract-fixture",
    }
    checkpoint = {
        "schema_version": 1,
        "checkpoint_kind": "direct_remote_nwb_behavior_trials",
        "trial_input_mode": "remote_nwb",
        "dandiset_id": "fixture_dandiset",
        "dandiset_version": "fixture_version",
        "session_inventory": _artifact_record(inventory_path),
        "behavior_trials": trusted_trials,
        "companion_trials": companion_record,
        "generator": behavior_generator,
        "local_sources": behavior_sources,
    }
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_reference = {
        **_artifact_record(checkpoint_path),
        "status": "validated_existing_not_refreshed",
        "trusted_behavior_trials_sha256": trusted_trials["sha256"],
        "output_matches_trusted_checkpoint": True,
    }
    (manifests / "behavior_analysis_run.json").write_text(
        json.dumps(
            {
                "analysis_id": "fixture_behavior_analysis",
                "analysis_status": "pass",
                "run_status": "complete",
                "authoritative": True,
                "pending_behavior_decisions": [],
                "dandiset_id": "fixture_dandiset",
                "dandiset_version": "fixture_version",
                "generator": behavior_generator,
                "local_sources": behavior_sources,
                "input_audit_run": m0_manifest,
                "input_audit_run_manifest": _artifact_record(audit_run_path),
                "analysis_lock_path": str(lock_path.resolve()),
                "analysis_lock_sha256": lock_sha256,
                "analysis_lock_snapshot": str(behavior_lock_snapshot.resolve()),
                "session_inventory_sha256": m0_artifacts["session_inventory"]["sha256"],
                "session_inventory_path": str(inventory_path.resolve()),
                "task_parameters_audit_path": str(task_parameters_audit_path.resolve()),
                "task_parameters_audit_sha256": m0_artifacts["task_parameters_audit"]["sha256"],
                "trial_input_mode": "validated_existing_behavior_trials",
                "behavior_trials_checkpoint": checkpoint_reference,
                "reused_behavior_trials_sha256": trusted_trials["sha256"],
                "companion_trials_provenance": {
                    "cached_path": str(companion_path.resolve()),
                    "sha256": companion_record["sha256"],
                    "content_size_bytes": companion_record["size_bytes"],
                    "source_url": companion_record["source_url"],
                    "repository_commit": companion_record["repository_commit"],
                },
                "outputs": {
                    "session_qc": _artifact_record(session_qc_path),
                    "behavior_trials": trusted_trials,
                },
            }
        ),
        encoding="utf-8",
    )


def _script_arguments(
    results_root: pathlib.Path,
    lock_path: pathlib.Path,
) -> list[str]:
    return [
        str(SCRIPT_PATH),
        "--results-root",
        str(results_root),
        "--analysis-lock",
        str(lock_path),
    ]


def _prepare_isolated_repository(root: pathlib.Path) -> None:
    """Copy only source/lock inputs needed by the authenticated API fixture."""

    actual_root = SCRIPT_PATH.parents[1]
    required = {
        *ALLOCATE_COHORTS.LOCAL_SOURCE_PATHS,
        *ALLOCATE_COHORTS.DEPENDENCY_LOCK_PATHS,
        "scripts/01_analyze_behavior.py",
        "src/dg/behavior.py",
    }
    for relative in sorted(required):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((actual_root / relative).read_bytes())


def _install_ontology_contract_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace network access with a labeled ontology interface fixture."""

    divisions = pl.DataFrame(
        [
            {
                "major_division_order": order,
                "major_division_id": identifier,
                "major_division_acronym": acronym,
                "major_division_name": name,
                "official_structure_id_path": f"/997/{identifier}/",
                "structure_graph_id": dg.ontology.STRUCTURE_GRAPH_ID,
                "structure_set_id": dg.ontology.MAJOR_DIVISION_STRUCTURE_SET_ID,
            }
            for order, (identifier, acronym, name) in enumerate(
                dg.ontology.EXPECTED_MAJOR_DIVISIONS,
                start=1,
            )
        ]
    )
    structures = pl.DataFrame(
        [
            {
                "structure_id": identifier,
                "acronym": acronym,
                "structure_name": name,
                "major_division_id": identifier,
                "major_division_acronym": acronym,
                "major_division_name": name,
                "major_division_mapping_status": "mapped_by_structure_id_ancestry",
            }
            for identifier, acronym, name in dg.ontology.EXPECTED_MAJOR_DIVISIONS
        ]
        + [
            {
                "structure_id": 99_999,
                "acronym": "fixture_raw_acronym",
                "structure_name": "Fixture raw acronym",
                "major_division_id": 1097,
                "major_division_acronym": "HY",
                "major_division_name": "Hypothalamus",
                "major_division_mapping_status": "mapped_by_structure_id_ancestry",
            }
        ]
    )
    ontology = dg.ontology.MajorDivisionOntology(
        structures=structures,
        divisions=divisions,
        validation=pl.DataFrame(
            {
                "check": ["labeled_orchestration_fixture"],
                "status": ["pass"],
                "detail": ["not scientific result data"],
            }
        ),
    )

    def cache_fixture(cache_directory: pathlib.Path) -> dg.ontology.CachedOntologySources:
        cache_directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        provenance_rows = []
        for resource in dg.ontology.PINNED_RESOURCES:
            payload = f"labeled cache contract fixture: {resource.key}\n".encode()
            path = cache_directory / resource.filename
            path.write_bytes(payload)
            paths[resource.key] = path
            provenance_rows.append(
                {
                    "resource_key": resource.key,
                    "authority": "test_fixture",
                    "source_url": "https://example.invalid/test-fixture",
                    "repository_commit": None,
                    "structure_graph_id": None,
                    "structure_set_id": None,
                    "cached_filename": resource.filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "cache_status": "test_fixture",
                    "cache_file_mode": "test_fixture",
                    "immutability_policy": "test_fixture",
                }
            )
        return dg.ontology.CachedOntologySources(
            paths=paths,
            provenance=pl.DataFrame(provenance_rows, infer_schema_length=None),
        )

    monkeypatch.setattr(
        ALLOCATE_COHORTS.dg.ontology,
        "cache_d05_ontology_sources",
        cache_fixture,
    )
    monkeypatch.setattr(
        ALLOCATE_COHORTS.dg.ontology,
        "load_d05_major_division_ontology",
        lambda _: ontology,
    )


def test_script_writes_hashed_complete_allocation_artifacts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    _install_ontology_contract_fixture(monkeypatch)
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    ALLOCATE_COHORTS.main()

    manifest_path = results_root / "manifests" / "cohort_allocation_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console = capsys.readouterr().out
    for subject_id in manifest["assignment_identity"]["confirmation_mouse_ids"]:
        assert subject_id not in console
    assert manifest["schema_version"] == 3
    assert manifest["analysis_status"] == "pass"
    assert manifest["run_status"] == "complete"
    assert manifest["authoritative"] is True
    assert manifest["ready_for_neural_discovery"] is True
    assert manifest["allocation_status"] == "final"
    assert manifest["assignment_table_status"] == ("final_coarse_regional_coverage_balanced")
    assert manifest["confirmation_holdout_accessed"] is False
    assert manifest["coarse_regional_coverage_contract"]["included_divisions"] == ["HY"]
    assert manifest["neural_activity_used_for_allocation"] is False
    assert manifest["m0_reference"]["authoritative"] is True
    assert manifest["cohort_counts"]["eligible_mice"] == 18
    assert manifest["cohort_counts"]["discovery_mice"] == 12
    assert manifest["cohort_counts"]["confirmation_mice"] == 6
    assert manifest["search"]["candidates_evaluated"] == 18_564
    assert set(manifest["dependency_lockfiles"]) == {"pyproject.toml", "uv.lock"}
    assert manifest["outputs"]["software_environment"]["sha256"]
    assert (
        manifest["parent_hashes"]["m0_audit_run_sha256"]
        == manifest["inputs"]["m0_audit_run"]["sha256"]
    )
    assert (
        manifest["assignment_hashes"]["session_assignments_parquet_sha256"]
        == manifest["outputs"]["session_assignments_parquet"]["sha256"]
    )
    for record in manifest["outputs"].values():
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = ALLOCATE_COHORTS.REPOSITORY_ROOT / path
        assert ALLOCATE_COHORTS._sha256_file(path) == record["sha256"]

    mouse_csv_path = results_root / "tables" / "mouse_cohort_assignments.csv"
    mouse_parquet_path = results_root / "tables" / "mouse_cohort_assignments.parquet"
    session_csv_path = results_root / "tables" / "session_cohort_assignments.csv"
    session_parquet_path = results_root / "tables" / "session_cohort_assignments.parquet"
    mouse_parquet = pl.read_parquet(mouse_parquet_path)
    session_parquet = pl.read_parquet(session_parquet_path)
    mouse_csv = pl.read_csv(mouse_csv_path, schema_overrides={"subject_id": pl.String})
    session_csv = pl.read_csv(session_csv_path, schema_overrides={"subject_id": pl.String})
    assert mouse_parquet.schema["subject_id"] == pl.String
    assert session_parquet.schema["subject_id"] == pl.String
    assert (
        mouse_csv.get_column("subject_id").to_list()
        == mouse_parquet.get_column("subject_id").to_list()
    )
    assert (
        session_csv.get_column("subject_id").to_list()
        == session_parquet.get_column("subject_id").to_list()
    )
    assert mouse_parquet.height == 19
    assert session_parquet.height == 19
    assert mouse_parquet.filter(pl.col("cohort_assignment") == "excluded").height == 1
    assert "analysis_included" not in session_parquet.columns
    assert session_parquet.get_column("discovery_analysis_included").any()
    assert not session_parquet.get_column("confirmation_analysis_included").any()


def test_script_rejects_non_authoritative_m0(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    audit_path = results_root / "manifests" / "audit_run.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["authoritative"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match="not complete and authoritative"):
        ALLOCATE_COHORTS.main()


def test_script_accepts_complete_m0_when_d05_component_gates_pass(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    audit_path = results_root / "manifests" / "audit_run.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["milestone_0_status"] = "complete"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    behavior_path = results_root / "manifests" / "behavior_analysis_run.json"
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    behavior["input_audit_run"] = audit
    behavior["input_audit_run_manifest"] = _artifact_record(audit_path)
    behavior["trial_input_mode"] = "remote_nwb"
    behavior["reused_behavior_trials_sha256"] = None
    behavior["behavior_trials_checkpoint"]["status"] = "created_from_remote_nwb"
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    _install_ontology_contract_fixture(monkeypatch)
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    ALLOCATE_COHORTS.main()

    manifest = json.loads(
        (results_root / "manifests" / "cohort_allocation_run.json").read_text(encoding="utf-8")
    )
    assert manifest["m0_reference"]["milestone_0_status"] == "complete"


def test_script_publishes_in_progress_then_failed_non_authoritative_marker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    observed_marker: dict[str, object] = {}

    def fail_after_observing_marker(*args: object, **kwargs: object) -> None:
        del args, kwargs
        marker_path = results_root / "manifests" / "cohort_allocation_run.json"
        observed_marker.update(json.loads(marker_path.read_text(encoding="utf-8")))
        raise RuntimeError("labeled fixture failure")

    monkeypatch.setattr(ALLOCATE_COHORTS, "_run_allocation", fail_after_observing_marker)
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match="labeled fixture failure"):
        ALLOCATE_COHORTS.main()

    assert observed_marker["run_status"] == "in_progress"
    assert observed_marker["authoritative"] is False
    failed = json.loads(
        (results_root / "manifests" / "cohort_allocation_run.json").read_text(encoding="utf-8")
    )
    assert failed["run_status"] == "failed"
    assert failed["analysis_status"] == "failed"
    assert failed["authoritative"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("milestone_0_status", "in_progress", "neither explicit partial nor complete"),
        ("dandiset_version", "different_version", "dandiset version differs"),
        ("asset_inventory_status", "fail", "asset inventory did not pass"),
        ("nwb_root_metadata_audit_status", "fail", "root-metadata audit did not pass"),
        (
            "unit_metadata_inventory_status",
            "fail",
            "scalar unit-metadata inventory did not pass",
        ),
    ],
)
def test_script_requires_exact_d05_safe_m0_component_gates(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    audit_path = results_root / "manifests" / "audit_run.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit[field] = value
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match=message):
        ALLOCATE_COHORTS.main()


def test_script_rejects_stale_behavior_qc_provenance(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    behavior_path = results_root / "manifests" / "behavior_analysis_run.json"
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    behavior["outputs"]["session_qc"]["sha256"] = "0" * 64
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match="differs from its recorded hash"):
        ALLOCATE_COHORTS.main()


@pytest.mark.parametrize(
    "relative_path",
    [
        "manifests/unit_session_anatomy_coverage.csv",
        "manifests/behavior_trials_checkpoint.json",
        "manifests/behavior_analysis_lock.yaml",
    ],
)
def test_script_rejects_tampered_upstream_artifact_chain(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    target = results_root / relative_path
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match="differs|snapshot"):
        ALLOCATE_COHORTS.main()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_status", "in_progress", "completed authoritative pass"),
        ("authoritative", False, "completed authoritative pass"),
        ("trial_input_mode", "untrusted_reuse", "non-canonical trial input mode"),
    ],
)
def test_script_rejects_noncanonical_behavior_parent_status(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    behavior_path = results_root / "manifests" / "behavior_analysis_run.json"
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    behavior[field] = value
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))

    with pytest.raises(RuntimeError, match=message):
        ALLOCATE_COHORTS.main()


def test_script_requires_true_behavior_checkpoint_match_and_canonical_status(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    behavior_path = results_root / "manifests" / "behavior_analysis_run.json"
    original = behavior_path.read_bytes()

    behavior = json.loads(original)
    behavior["behavior_trials_checkpoint"]["output_matches_trusted_checkpoint"] = False
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))
    with pytest.raises(RuntimeError, match="does not exactly match"):
        ALLOCATE_COHORTS.main()

    behavior = json.loads(original)
    behavior["behavior_trials_checkpoint"]["status"] = "created_from_remote_nwb"
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    with pytest.raises(RuntimeError, match="status is incompatible"):
        ALLOCATE_COHORTS.main()

    behavior = json.loads(original)
    behavior["behavior_trials_checkpoint"]["trusted_behavior_trials_sha256"] = "0" * 64
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    with pytest.raises(RuntimeError, match="disagree on the trusted trial digest"):
        ALLOCATE_COHORTS.main()


def test_script_rejects_behavior_source_and_task_audit_tampering(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "fixture_results"
    lock_path = tmp_path / "fixture_analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    behavior_path = results_root / "manifests" / "behavior_analysis_run.json"
    original = behavior_path.read_bytes()

    behavior = json.loads(original)
    behavior["local_sources"][0]["sha256"] = "0" * 64
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))
    with pytest.raises(RuntimeError, match="behavior local source 0 file differs"):
        ALLOCATE_COHORTS.main()

    behavior = json.loads(original)
    behavior["input_audit_run"]["milestone_0_status"] = "complete"
    behavior_path.write_text(json.dumps(behavior), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not embed the current authoritative M0"):
        ALLOCATE_COHORTS.main()

    behavior_path.write_bytes(original)
    task_audit_path = results_root / "manifests" / "task_parameters_audit.csv"
    task_audit_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="M0 artifact task_parameters_audit file differs"):
        ALLOCATE_COHORTS.main()


def test_authenticated_discovery_api_recomputes_split_and_rejects_tampering(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_root = tmp_path / "labeled_repository_contract_fixture"
    _prepare_isolated_repository(isolated_root)
    monkeypatch.setattr(ALLOCATE_COHORTS, "REPOSITORY_ROOT", isolated_root)
    results_root = isolated_root / "results"
    lock_path = isolated_root / "config" / "analysis_lock.yaml"
    _write_approved_lock(lock_path)
    _write_orchestration_fixture(results_root, lock_path)
    _install_ontology_contract_fixture(monkeypatch)
    monkeypatch.setattr(sys, "argv", _script_arguments(results_root, lock_path))
    ALLOCATE_COHORTS.main()

    manifest_path = results_root / "manifests" / "cohort_allocation_run.json"
    assignments_path = results_root / "tables" / "session_cohort_assignments.parquet"
    mouse_assignments_path = results_root / "tables" / "mouse_cohort_assignments.parquet"
    inventory_path = results_root / "manifests" / "session_inventory.parquet"
    qc_path = results_root / "tables" / "session_qc.parquet"

    def authenticate() -> tuple[str, ...]:
        return ALLOCATE_COHORTS.dg.cohorts.select_authenticated_discovery_session_sources(
            session_assignments_path=assignments_path,
            cohort_manifest_path=manifest_path,
            session_inventory_path=inventory_path,
            session_qc_path=qc_path,
            repository_root=isolated_root,
        )

    baseline_sources = authenticate()
    assert len(baseline_sources) == 12
    baseline_assignments = pl.read_parquet(assignments_path)
    baseline_mouse_assignments = pl.read_parquet(mouse_assignments_path)
    baseline_assignment_bytes = assignments_path.read_bytes()
    baseline_mouse_assignment_bytes = mouse_assignments_path.read_bytes()
    baseline_manifest_bytes = manifest_path.read_bytes()

    assignments_path.write_bytes(baseline_assignment_bytes + b"changed")
    with pytest.raises(RuntimeError, match="content differs from its recorded hash"):
        authenticate()

    def publish_changed_assignments(
        changed: pl.DataFrame,
        *,
        changed_mice: pl.DataFrame | None = None,
        update_identity_and_counts: bool,
    ) -> None:
        assignments_path.write_bytes(baseline_assignment_bytes)
        mouse_assignments_path.write_bytes(baseline_mouse_assignment_bytes)
        changed.write_parquet(assignments_path)
        manifest = json.loads(baseline_manifest_bytes)
        assignment_record = _artifact_record(assignments_path)
        manifest["outputs"]["session_assignments_parquet"] = assignment_record
        manifest["assignment_hashes"]["session_assignments_parquet_sha256"] = assignment_record[
            "sha256"
        ]
        if changed_mice is not None:
            changed_mice.write_parquet(mouse_assignments_path)
            mouse_record = _artifact_record(mouse_assignments_path)
            manifest["outputs"]["mouse_assignments_parquet"] = mouse_record
            manifest["assignment_hashes"]["mouse_assignments_parquet_sha256"] = mouse_record[
                "sha256"
            ]
        if update_identity_and_counts:
            mice = changed.select("subject_id", "cohort_assignment").unique()
            for cohort in ("discovery", "confirmation", "excluded"):
                identifiers = (
                    mice.filter(pl.col("cohort_assignment") == cohort)
                    .get_column("subject_id")
                    .sort()
                    .to_list()
                )
                manifest["assignment_identity"][f"{cohort}_mouse_ids"] = identifiers
                manifest["cohort_counts"][f"{cohort}_mice"] = len(identifiers)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    flag_flip = baseline_assignments.with_columns(
        (~pl.col("discovery_analysis_included")).alias("discovery_analysis_included")
    )
    publish_changed_assignments(flag_flip, update_identity_and_counts=False)
    with pytest.raises(RuntimeError, match="recomputed exact D05 split"):
        authenticate()

    discovery_mouse = baseline_assignments.filter(pl.col("cohort_assignment") == "discovery").item(
        0, "subject_id"
    )
    confirmation_mouse = baseline_assignments.filter(
        pl.col("cohort_assignment") == "confirmation"
    ).item(0, "subject_id")
    swapped = _reassign_fixture_sessions(
        baseline_assignments,
        {discovery_mouse: "confirmation", confirmation_mouse: "discovery"},
    )
    swapped_mice = _reassign_fixture_mice(
        baseline_mouse_assignments,
        {discovery_mouse: "confirmation", confirmation_mouse: "discovery"},
    )
    publish_changed_assignments(
        swapped,
        changed_mice=swapped_mice,
        update_identity_and_counts=True,
    )
    with pytest.raises(RuntimeError, match="recomputed exact D05 split"):
        authenticate()

    wrong_counts = _reassign_fixture_sessions(
        baseline_assignments,
        {discovery_mouse: "confirmation"},
    )
    wrong_count_mice = _reassign_fixture_mice(
        baseline_mouse_assignments,
        {discovery_mouse: "confirmation"},
    )
    publish_changed_assignments(
        wrong_counts,
        changed_mice=wrong_count_mice,
        update_identity_and_counts=True,
    )
    with pytest.raises(RuntimeError, match="recomputed exact D05 split"):
        authenticate()


def _reassign_fixture_sessions(
    assignments: pl.DataFrame,
    replacements: dict[str, str],
) -> pl.DataFrame:
    cohort = pl.col("subject_id").replace_strict(
        replacements,
        default=pl.col("cohort_assignment"),
        return_dtype=pl.String,
    )
    return assignments.with_columns(cohort.alias("cohort_assignment")).with_columns(
        (pl.col("cohort_assignment") == "discovery").alias("discovery_mouse_included"),
        ((pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible")).alias(
            "discovery_analysis_included"
        ),
        pl.when(pl.col("cohort_assignment") == "confirmation")
        .then(pl.lit("sealed_until_one_time_confirmation_gate"))
        .when(pl.col("cohort_assignment") == "discovery")
        .then(pl.lit("discovery_activated_confirmation_sealed"))
        .otherwise(pl.lit("not_applicable"))
        .alias("confirmation_access_policy"),
        pl.when((pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible"))
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit("not_selected_for_discovery_analysis"))
        .alias("analysis_activation_block_reason"),
    )


def _reassign_fixture_mice(
    assignments: pl.DataFrame,
    replacements: dict[str, str],
) -> pl.DataFrame:
    cohort = pl.col("subject_id").replace_strict(
        replacements,
        default=pl.col("cohort_assignment"),
        return_dtype=pl.String,
    )
    return assignments.with_columns(cohort.alias("cohort_assignment")).with_columns(
        (pl.col("cohort_assignment") == "discovery").alias("discovery_mouse_included"),
        pl.when(pl.col("cohort_assignment") == "confirmation")
        .then(pl.lit("sealed_until_one_time_confirmation_gate"))
        .when(pl.col("cohort_assignment") == "discovery")
        .then(pl.lit("discovery_activated_confirmation_sealed"))
        .otherwise(pl.lit("not_applicable"))
        .alias("confirmation_access_policy"),
    )
