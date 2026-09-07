"""Confirmation requires a frozen, unused gate in addition to D01-D14."""

import copy
import hashlib
import json
import pathlib

import polars as pl
import pytest

import dg.confirmation
import dg.gates


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _record(path: pathlib.Path, root: pathlib.Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _assignment_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for index in range(18):
        subject_id = f"fixture_mouse_{index + 1:02d}"
        cohort = "discovery" if index < 12 else "confirmation"
        rows.append(
            {
                "subject_id": subject_id,
                "cohort_assignment": cohort,
                "allocation_status": dg.confirmation.D05_FINAL_ASSIGNMENT_STATUS,
                "split_seed": dg.confirmation.D05_SPLIT_SEED,
                "confirmation_access_policy": (
                    "discovery_activated_confirmation_sealed"
                    if cohort == "discovery"
                    else "sealed_until_one_time_confirmation_gate"
                ),
                "mouse_behavior_eligible": True,
                "ready_for_neural_discovery": True,
                "discovery_mouse_included": cohort == "discovery",
                "confirmation_mouse_included": False,
                "neural_activity_used_for_allocation": False,
            }
        )
    mice = pl.DataFrame(rows, infer_schema_length=None)
    sessions = mice.with_columns(
        pl.concat_str(pl.lit("fixture_session_"), pl.col("subject_id")).alias("_nwb_path"),
        pl.lit(True).alias("session_behavior_eligible"),
        (pl.col("cohort_assignment") == "discovery").alias("discovery_analysis_included"),
        pl.lit(False).alias("confirmation_analysis_included"),
    ).select(
        "_nwb_path",
        "subject_id",
        "cohort_assignment",
        "allocation_status",
        "split_seed",
        "confirmation_access_policy",
        "session_behavior_eligible",
        "mouse_behavior_eligible",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "discovery_analysis_included",
        "confirmation_analysis_included",
        "neural_activity_used_for_allocation",
    )
    return mice, sessions


def _valid_gate(tmp_path: pathlib.Path) -> tuple[dict, dict]:
    lock = dg.gates.read_analysis_lock("config/analysis_lock.yaml")
    m0 = {
        "run_status": "complete",
        "authoritative": True,
        "milestone_0_status": "complete",
    }
    m0_path = tmp_path / "m0_audit_manifest.json"
    _write_json(m0_path, m0)
    m0_record = _record(m0_path, tmp_path)
    behavior = {
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "pending_behavior_decisions": [],
        "behavior_trials_checkpoint": {"output_matches_trusted_checkpoint": True},
        "input_audit_run_manifest": m0_record,
        "input_audit_run": m0,
    }
    behavior_path = tmp_path / "behavior_qc_manifest.json"
    _write_json(behavior_path, behavior)
    behavior_record = _record(behavior_path, tmp_path)

    mice, sessions = _assignment_frames()
    mouse_path = tmp_path / "mouse_assignments.parquet"
    session_path = tmp_path / "session_assignments.parquet"
    mice.write_parquet(mouse_path)
    sessions.write_parquet(session_path)
    mouse_record = _record(mouse_path, tmp_path)
    session_record = _record(session_path, tmp_path)
    identities = {
        cohort: (
            mice.filter(pl.col("cohort_assignment") == cohort)
            .get_column("subject_id")
            .sort()
            .to_list()
        )
        for cohort in ("discovery", "confirmation", "excluded")
    }
    cohort = {
        "schema_version": dg.confirmation.D05_COHORT_MANIFEST_SCHEMA_VERSION,
        "analysis_id": dg.confirmation.D05_ANALYSIS_ID,
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "allocation_status": "final",
        "assignment_table_status": dg.confirmation.D05_FINAL_ASSIGNMENT_STATUS,
        "ready_for_neural_discovery": True,
        "confirmation_holdout_accessed": False,
        "neural_activity_used_for_allocation": False,
        "split_seed": dg.confirmation.D05_SPLIT_SEED,
        "requested_discovery_mice": dg.confirmation.D05_DISCOVERY_MICE,
        "requested_confirmation_mice": dg.confirmation.D05_CONFIRMATION_MICE,
        "inputs": {
            "m0_audit_run": m0_record,
            "behavior_input_m0_manifest": m0_record,
            "behavior_analysis_run": behavior_record,
        },
        "parent_hashes": {
            "m0_audit_run_sha256": m0_record["sha256"],
            "behavior_analysis_run_sha256": behavior_record["sha256"],
        },
        "outputs": {
            "mouse_assignments_parquet": mouse_record,
            "session_assignments_parquet": session_record,
        },
        "assignment_hashes": {
            "mouse_assignments_parquet_sha256": mouse_record["sha256"],
            "session_assignments_parquet_sha256": session_record["sha256"],
        },
        "assignment_identity": {
            f"{name}_mouse_ids": identifiers for name, identifiers in identities.items()
        },
        "cohort_counts": {
            "inventory_mice": 18,
            "eligible_mice": 18,
            "excluded_mice": 0,
            "discovery_mice": 12,
            "confirmation_mice": 6,
            "inventory_sessions": 18,
            "threshold_selected_sessions": 18,
            "discovery_threshold_selected_sessions": 12,
            "confirmation_threshold_selected_sessions": 6,
        },
    }
    cohort_path = tmp_path / "cohort_allocation_manifest.json"
    _write_json(cohort_path, cohort)

    values = {
        "discovery_specification": {"frozen": True},
        "analysis_code_archive": {"frozen": True},
        "software_environment": {"frozen": True},
        "region_pair_viability": {
            "status": "pass",
            "eligible_confirmation_mice": 6,
            "simultaneous_region_pair_confirmation_mice": 5,
        },
    }
    records = {
        "cohort_allocation_manifest": _record(cohort_path, tmp_path),
        "m0_audit_manifest": m0_record,
        "behavior_qc_manifest": behavior_record,
    }
    for artifact_id, value in values.items():
        path = tmp_path / f"{artifact_id}.json"
        _write_json(path, value)
        records[artifact_id] = _record(path, tmp_path)
    gate = {
        "schema_version": 1,
        "status": "approved_for_single_run",
        "consumed_at": None,
        "confirmation_run_id": "confirmation-001",
        "approval_record": {
            "approved_by": "reviewer",
            "approved_at": "2026-09-07T05:00:00Z",
        },
        "analysis_lock_sha256": dg.confirmation._canonical_json_sha256(lock),
        "viability": {
            "status": "pass",
            "eligible_confirmation_mice": 6,
            "simultaneous_region_pair_confirmation_mice": 5,
        },
        "frozen_artifacts": records,
    }
    return lock, gate


def _artifact_path(
    gate: dict,
    artifact_id: str,
    root: pathlib.Path,
) -> pathlib.Path:
    return root / gate["frozen_artifacts"][artifact_id]["path"]


def _rewrite_cohort_manifest(
    gate: dict,
    cohort: dict,
    root: pathlib.Path,
) -> None:
    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", root)
    _write_json(cohort_path, cohort)
    gate["frozen_artifacts"]["cohort_allocation_manifest"] = _record(cohort_path, root)


def test_approved_decisions_do_not_open_confirmation_without_gate(
    tmp_path: pathlib.Path,
) -> None:
    lock = dg.gates.read_analysis_lock("config/analysis_lock.yaml")

    with pytest.raises(RuntimeError, match="does not exist"):
        dg.confirmation.require_confirmation_gate(
            lock,
            tmp_path / "missing.json",
            repository_root=tmp_path,
        )


def test_complete_unused_content_addressed_gate_opens(tmp_path: pathlib.Path) -> None:
    lock, gate = _valid_gate(tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    observed = dg.confirmation.require_confirmation_gate(
        lock,
        path,
        repository_root=tmp_path,
    )

    assert observed["confirmation_run_id"] == "confirmation-001"


def test_confirmation_claim_is_atomic_and_one_time(tmp_path: pathlib.Path) -> None:
    lock, gate = _valid_gate(tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    claimed = dg.confirmation.claim_confirmation_gate(
        lock,
        path,
        repository_root=tmp_path,
        claimed_by="confirmation runner",
    )

    assert claimed["status"] == "consumed"
    assert claimed["consumed_at"] is not None
    assert (tmp_path / "confirmation_gate.json.claimed").is_file()
    with pytest.raises(RuntimeError, match="claim marker"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)
    with pytest.raises(RuntimeError, match="claim marker"):
        dg.confirmation.claim_confirmation_gate(
            lock,
            path,
            repository_root=tmp_path,
            claimed_by="second runner",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda gate: gate.update(consumed_at="2026-09-07T05:01:00Z"), "already consumed"),
        (lambda gate: gate["viability"].update(eligible_confirmation_mice=5), "D14"),
        (lambda gate: gate.update(analysis_lock_sha256="0" * 64), "analysis lock"),
    ],
)
def test_confirmation_gate_fails_closed_on_state_or_viability_change(
    tmp_path: pathlib.Path,
    mutation,
    message: str,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    mutation(gate)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match=message):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_gate_rejects_changed_frozen_artifact(tmp_path: pathlib.Path) -> None:
    lock, gate = _valid_gate(tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)
    frozen_path = tmp_path / gate["frozen_artifacts"]["discovery_specification"]["path"]
    _write_json(frozen_path, {"frozen": False})

    with pytest.raises(RuntimeError, match="content has changed"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_rejects_noncomplete_or_nonauthoritative_d05(
    tmp_path: pathlib.Path,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["run_status"] = "failed"
    cohort["authoritative"] = False
    _rewrite_cohort_manifest(gate, cohort, tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="non-authoritative final D05"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


@pytest.mark.parametrize(
    ("artifact_id", "parent_label"),
    [
        ("m0_audit_manifest", "M0"),
        ("behavior_qc_manifest", "behavior"),
    ],
)
def test_confirmation_rejects_d05_bound_to_a_different_parent_path(
    tmp_path: pathlib.Path,
    artifact_id: str,
    parent_label: str,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    original = _artifact_path(gate, artifact_id, tmp_path)
    replacement = tmp_path / f"unrelated_{artifact_id}.json"
    replacement.write_bytes(original.read_bytes())
    gate["frozen_artifacts"][artifact_id] = _record(replacement, tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match=rf"D05 {parent_label} parent.*exactly"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_rejects_noncomplete_m0_even_when_parent_chain_is_rebound(
    tmp_path: pathlib.Path,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    m0_path = _artifact_path(gate, "m0_audit_manifest", tmp_path)
    m0 = json.loads(m0_path.read_text(encoding="utf-8"))
    m0["milestone_0_status"] = "partial"
    _write_json(m0_path, m0)
    m0_record = _record(m0_path, tmp_path)

    behavior_path = _artifact_path(gate, "behavior_qc_manifest", tmp_path)
    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    behavior["input_audit_run_manifest"] = m0_record
    behavior["input_audit_run"] = m0
    _write_json(behavior_path, behavior)
    behavior_record = _record(behavior_path, tmp_path)

    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["inputs"]["m0_audit_run"] = m0_record
    cohort["inputs"]["behavior_input_m0_manifest"] = m0_record
    cohort["inputs"]["behavior_analysis_run"] = behavior_record
    cohort["parent_hashes"]["m0_audit_run_sha256"] = m0_record["sha256"]
    cohort["parent_hashes"]["behavior_analysis_run_sha256"] = behavior_record["sha256"]
    _rewrite_cohort_manifest(gate, cohort, tmp_path)
    gate["frozen_artifacts"]["m0_audit_manifest"] = m0_record
    gate["frozen_artifacts"]["behavior_qc_manifest"] = behavior_record
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="non-authoritative M0"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


@pytest.mark.parametrize(
    "output_name",
    ["mouse_assignments_parquet", "session_assignments_parquet"],
)
def test_confirmation_rejects_changed_assignment_bytes(
    tmp_path: pathlib.Path,
    output_name: str,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    assignment_path = tmp_path / cohort["outputs"][output_name]["path"]
    assignment_path.write_bytes(assignment_path.read_bytes() + b"changed")
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="content differs from its recorded hash"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_rejects_self_consistent_assignment_that_activates_holdout(
    tmp_path: pathlib.Path,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    assignment_path = tmp_path / cohort["outputs"]["session_assignments_parquet"]["path"]
    assignments = pl.read_parquet(assignment_path).with_columns(
        pl.when(pl.col("cohort_assignment") == "confirmation")
        .then(pl.lit(True))
        .otherwise(pl.col("confirmation_analysis_included"))
        .alias("confirmation_analysis_included")
    )
    assignments.write_parquet(assignment_path)
    assignment_record = _record(assignment_path, tmp_path)
    cohort["outputs"]["session_assignments_parquet"] = assignment_record
    cohort["assignment_hashes"]["session_assignments_parquet_sha256"] = assignment_record["sha256"]
    _rewrite_cohort_manifest(gate, cohort, tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="activate the confirmation holdout"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_rejects_manifest_count_that_disagrees_with_assignments(
    tmp_path: pathlib.Path,
) -> None:
    lock, gate = _valid_gate(tmp_path)
    cohort_path = _artifact_path(gate, "cohort_allocation_manifest", tmp_path)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["cohort_counts"]["confirmation_threshold_selected_sessions"] = 5
    _rewrite_cohort_manifest(gate, cohort, tmp_path)
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="counts disagree"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)


def test_confirmation_gate_rejects_unapproved_decision(tmp_path: pathlib.Path) -> None:
    lock, gate = _valid_gate(tmp_path)
    lock = copy.deepcopy(lock)
    lock["overall_status"] = "proposed"
    lock.pop("approval_record")
    lock["decisions"]["D01"]["status"] = "proposed"
    path = tmp_path / "confirmation_gate.json"
    _write_json(path, gate)

    with pytest.raises(RuntimeError, match="decision approvals is locked"):
        dg.confirmation.require_confirmation_gate(lock, path, repository_root=tmp_path)
