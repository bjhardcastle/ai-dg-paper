"""Fail-closed validation for a one-time held-out confirmation run.

Approval of the high-level analysis decisions is necessary but deliberately
insufficient.  A confirmation run also requires a separately reviewed,
content-addressed gate that freezes every discovery-dependent choice and has
not previously been consumed.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import pathlib
import re
import tempfile
from typing import Any

import polars as pl

import dg.gates

CONFIRMATION_GATE_SCHEMA_VERSION = 1
READY_STATUS = "approved_for_single_run"
REQUIRED_FROZEN_ARTIFACTS = (
    "discovery_specification",
    "analysis_code_archive",
    "software_environment",
    "cohort_allocation_manifest",
    "m0_audit_manifest",
    "behavior_qc_manifest",
    "region_pair_viability",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
D05_COHORT_MANIFEST_SCHEMA_VERSION = 3
D05_ANALYSIS_ID = "d05_mouse_grouped_cohort_allocation"
D05_FINAL_ASSIGNMENT_STATUS = "final_coarse_regional_coverage_balanced"
D05_SPLIT_SEED = 1051
D05_DISCOVERY_MICE = 12
D05_CONFIRMATION_MICE = 6


def require_confirmation_gate(
    analysis_lock: dict[str, Any],
    gate_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a verified unused confirmation gate or raise ``RuntimeError``.

    The gate is intentionally a second artifact rather than another decision
    flag in ``analysis_lock.yaml``.  It binds the final discovery freeze and
    viability assessment by hash, and therefore cannot exist legitimately
    until discovery has finished.
    """

    dg.gates.require_approved_decisions(
        analysis_lock,
        dg.gates.CONFIRMATION_GATE_DECISIONS,
        stage="confirmation decision approvals",
    )
    path = pathlib.Path(gate_path).resolve()
    root = pathlib.Path(repository_root).resolve()
    if _claim_marker_path(path).exists():
        raise RuntimeError("confirmation is locked: a one-time claim marker already exists")
    try:
        gate_bytes = path.read_bytes()
        gate = json.loads(gate_bytes)
    except FileNotFoundError as error:
        raise RuntimeError(
            "confirmation is locked: the one-time confirmation gate does not exist"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"confirmation gate cannot be read: {error}") from error
    if not isinstance(gate, dict):
        raise RuntimeError("confirmation gate must be a JSON object")
    _validate_gate_fields(gate, analysis_lock=analysis_lock)
    _validate_frozen_artifacts(gate["frozen_artifacts"], repository_root=root)
    _validate_bound_manifests(
        gate["frozen_artifacts"],
        gate_viability=gate["viability"],
        repository_root=root,
    )
    return gate


def claim_confirmation_gate(
    analysis_lock: dict[str, Any],
    gate_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    claimed_by: str,
) -> dict[str, Any]:
    """Atomically consume the gate before any confirmation source is opened.

    The exclusive adjacent claim marker is the concurrency guard.  It is never
    removed automatically: a crash after claiming is still a consumed attempt,
    as required by the one-time confirmation policy.
    """

    if not isinstance(claimed_by, str) or not claimed_by.strip():
        raise ValueError("claimed_by must be a non-empty string")
    path = pathlib.Path(gate_path).resolve()
    gate = require_confirmation_gate(
        analysis_lock,
        path,
        repository_root=repository_root,
    )
    ready_hash = _sha256(path)
    claimed_at = datetime.datetime.now(datetime.UTC).isoformat()
    marker = _claim_marker_path(path)
    marker_payload = (
        json.dumps(
            {
                "confirmation_run_id": gate["confirmation_run_id"],
                "ready_gate_sha256": ready_hash,
                "claimed_at": claimed_at,
                "claimed_by": claimed_by.strip(),
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(
            "confirmation is locked: the one-time gate was already claimed"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(marker_payload)
            stream.flush()
            os.fsync(stream.fileno())
        if _sha256(path) != ready_hash:
            raise RuntimeError("confirmation gate changed while its one-time claim was created")
        consumed = {
            **gate,
            "status": "consumed",
            "consumed_at": claimed_at,
            "consumed_by": claimed_by.strip(),
            "ready_gate_sha256": ready_hash,
        }
        _atomic_write_json(path, consumed)
    except BaseException:
        # The marker deliberately remains: an ambiguous/crashed claim must be
        # reviewed rather than silently granting another holdout attempt.
        raise
    return consumed


def confirmation_gate_sha256(path: str | os.PathLike[str]) -> str:
    """Return the content hash a runner must claim atomically before access."""

    return _sha256(pathlib.Path(path).resolve())


def _validate_gate_fields(
    gate: dict[str, Any],
    *,
    analysis_lock: dict[str, Any],
) -> None:
    if gate.get("schema_version") != CONFIRMATION_GATE_SCHEMA_VERSION:
        raise RuntimeError("confirmation gate has an unsupported schema version")
    if gate.get("status") != READY_STATUS:
        raise RuntimeError("confirmation is locked: one-time gate is not ready")
    if gate.get("consumed_at") is not None:
        raise RuntimeError("confirmation is locked: the one-time gate was already consumed")
    run_id = gate.get("confirmation_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise RuntimeError("confirmation gate lacks a non-empty confirmation_run_id")

    approval = gate.get("approval_record")
    if not isinstance(approval, dict) or not approval.get("approved_by"):
        raise RuntimeError("confirmation gate lacks reviewer approval metadata")
    _validate_timestamp(approval.get("approved_at"), label="confirmation approval")

    expected_lock_hash = gate.get("analysis_lock_sha256")
    observed_lock_hash = _canonical_json_sha256(analysis_lock)
    if expected_lock_hash != observed_lock_hash:
        raise RuntimeError("confirmation gate is not bound to the current analysis lock")

    viability = gate.get("viability")
    if not isinstance(viability, dict) or viability.get("status") != "pass":
        raise RuntimeError("confirmation viability has not passed")
    confirmation_mice = viability.get("eligible_confirmation_mice")
    pair_mice = viability.get("simultaneous_region_pair_confirmation_mice")
    if isinstance(confirmation_mice, bool) or not isinstance(confirmation_mice, int):
        raise RuntimeError("confirmation viability mouse count must be an integer")
    if isinstance(pair_mice, bool) or not isinstance(pair_mice, int):
        raise RuntimeError("region-pair viability mouse count must be an integer")
    if confirmation_mice < 6 or pair_mice < 5:
        raise RuntimeError("confirmation viability is below the approved D14 minima")

    artifacts = gate.get("frozen_artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("confirmation gate lacks a frozen_artifacts mapping")
    missing = set(REQUIRED_FROZEN_ARTIFACTS).difference(artifacts)
    unexpected = set(artifacts).difference(REQUIRED_FROZEN_ARTIFACTS)
    if missing or unexpected:
        raise RuntimeError(
            "confirmation frozen artifacts differ from the required set: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _validate_frozen_artifacts(
    records: dict[str, Any],
    *,
    repository_root: pathlib.Path,
) -> None:
    for artifact_id in REQUIRED_FROZEN_ARTIFACTS:
        _validated_artifact_payload(
            records[artifact_id],
            repository_root=repository_root,
            label=f"confirmation artifact {artifact_id!r}",
            changed_message=f"confirmation artifact {artifact_id!r} content has changed",
        )


def _validate_bound_manifests(
    records: dict[str, Any],
    *,
    gate_viability: dict[str, Any],
    repository_root: pathlib.Path,
) -> None:
    cohort = _read_json_artifact(records["cohort_allocation_manifest"], repository_root)
    expected_cohort_fields = {
        "schema_version": D05_COHORT_MANIFEST_SCHEMA_VERSION,
        "analysis_id": D05_ANALYSIS_ID,
        "analysis_status": "pass",
        "run_status": "complete",
        "authoritative": True,
        "allocation_status": "final",
        "assignment_table_status": D05_FINAL_ASSIGNMENT_STATUS,
        "ready_for_neural_discovery": True,
        "confirmation_holdout_accessed": False,
        "neural_activity_used_for_allocation": False,
        "split_seed": D05_SPLIT_SEED,
        "requested_discovery_mice": D05_DISCOVERY_MICE,
        "requested_confirmation_mice": D05_CONFIRMATION_MICE,
    }
    for field, expected in expected_cohort_fields.items():
        if cohort.get(field) != expected:
            raise RuntimeError(
                "confirmation gate is bound to a non-authoritative final D05 allocation"
            )

    cohort_inputs = cohort.get("inputs")
    if not isinstance(cohort_inputs, dict):
        raise RuntimeError("frozen D05 manifest lacks its input provenance chain")
    required_parent_inputs = {
        "m0_audit_run",
        "behavior_analysis_run",
        "behavior_input_m0_manifest",
    }
    if missing_inputs := sorted(required_parent_inputs.difference(cohort_inputs)):
        raise RuntimeError(f"frozen D05 manifest lacks required parent inputs: {missing_inputs}")
    _require_exact_record_binding(
        cohort_inputs["m0_audit_run"],
        records["m0_audit_manifest"],
        repository_root=repository_root,
        label="D05 M0 parent",
    )
    _require_exact_record_binding(
        cohort_inputs["behavior_input_m0_manifest"],
        records["m0_audit_manifest"],
        repository_root=repository_root,
        label="D05 behavior-to-M0 parent",
    )
    _require_exact_record_binding(
        cohort_inputs["behavior_analysis_run"],
        records["behavior_qc_manifest"],
        repository_root=repository_root,
        label="D05 behavior parent",
    )
    parent_hashes = cohort.get("parent_hashes")
    if not isinstance(parent_hashes, dict) or (
        parent_hashes.get("m0_audit_run_sha256") != cohort_inputs["m0_audit_run"].get("sha256")
        or parent_hashes.get("behavior_analysis_run_sha256")
        != cohort_inputs["behavior_analysis_run"].get("sha256")
    ):
        raise RuntimeError("frozen D05 exact parent hashes disagree with its input records")

    m0 = _read_json_artifact(records["m0_audit_manifest"], repository_root)
    if (
        m0.get("run_status") != "complete"
        or m0.get("authoritative") is not True
        or m0.get("milestone_0_status") != "complete"
    ):
        raise RuntimeError("confirmation gate is bound to a non-authoritative M0 audit")

    behavior = _read_json_artifact(records["behavior_qc_manifest"], repository_root)
    if (
        behavior.get("analysis_status") != "pass"
        or behavior.get("run_status") != "complete"
        or behavior.get("authoritative") is not True
        or behavior.get("pending_behavior_decisions") != []
    ):
        raise RuntimeError(
            "confirmation gate is bound to behavior QC that is not a completed authoritative pass"
        )
    checkpoint = behavior.get("behavior_trials_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("output_matches_trusted_checkpoint") is not True
    ):
        raise RuntimeError("confirmation behavior QC is not bound to trusted canonical trials")
    behavior_m0_record = behavior.get("input_audit_run_manifest")
    _require_exact_record_binding(
        behavior_m0_record,
        records["m0_audit_manifest"],
        repository_root=repository_root,
        label="behavior M0 parent",
    )
    if behavior.get("input_audit_run") != m0:
        raise RuntimeError("frozen behavior QC does not embed the frozen M0 audit")

    _validate_d05_assignments(cohort, repository_root=repository_root)

    viability = _read_json_artifact(records["region_pair_viability"], repository_root)
    if (
        viability.get("status") != "pass"
        or viability.get("eligible_confirmation_mice")
        != gate_viability["eligible_confirmation_mice"]
        or viability.get("simultaneous_region_pair_confirmation_mice")
        != gate_viability["simultaneous_region_pair_confirmation_mice"]
    ):
        raise RuntimeError("frozen region-pair viability artifact does not match the gate")


def _require_exact_record_binding(
    parent_record: Any,
    frozen_record: Any,
    *,
    repository_root: pathlib.Path,
    label: str,
) -> None:
    """Require two provenance records to identify the same exact repository file."""

    parent_path, _ = _validated_artifact_payload(
        parent_record,
        repository_root=repository_root,
        label=label,
    )
    frozen_path, _ = _validated_artifact_payload(
        frozen_record,
        repository_root=repository_root,
        label=f"frozen {label}",
    )
    if (
        parent_path != frozen_path
        or parent_record.get("sha256") != frozen_record.get("sha256")
        or parent_record.get("size_bytes") != frozen_record.get("size_bytes")
    ):
        raise RuntimeError(f"{label} is not exactly bound to the gate-frozen artifact")


def _validate_d05_assignments(
    cohort: dict[str, Any],
    *,
    repository_root: pathlib.Path,
) -> None:
    """Authenticate the sealed D05 mouse and session assignment artifacts."""

    outputs = cohort.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("frozen D05 manifest lacks its output provenance chain")
    required = {"mouse_assignments_parquet", "session_assignments_parquet"}
    if missing := sorted(required.difference(outputs)):
        raise RuntimeError(f"frozen D05 manifest lacks assignment outputs: {missing}")

    assignment_hashes = cohort.get("assignment_hashes")
    expected_assignment_hashes = {
        "mouse_assignments_parquet_sha256": outputs["mouse_assignments_parquet"].get("sha256"),
        "session_assignments_parquet_sha256": outputs["session_assignments_parquet"].get("sha256"),
    }
    if assignment_hashes != expected_assignment_hashes:
        raise RuntimeError("frozen D05 assignment hashes disagree with its output records")

    mouse_path, mouse_payload = _validated_artifact_payload(
        outputs["mouse_assignments_parquet"],
        repository_root=repository_root,
        label="D05 mouse-assignment Parquet",
    )
    session_path, session_payload = _validated_artifact_payload(
        outputs["session_assignments_parquet"],
        repository_root=repository_root,
        label="D05 session-assignment Parquet",
    )
    if (
        mouse_path == session_path
        or mouse_path.suffix.lower() != ".parquet"
        or session_path.suffix.lower() != ".parquet"
    ):
        raise RuntimeError("frozen D05 assignment outputs are not distinct Parquet files")
    try:
        mice = pl.read_parquet(io.BytesIO(mouse_payload))
        sessions = pl.read_parquet(io.BytesIO(session_payload))
    except (OSError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"could not read frozen D05 assignment Parquet: {error}") from error

    _validate_assignment_content(mice, sessions, cohort=cohort)


def _validate_assignment_content(
    mice: pl.DataFrame,
    sessions: pl.DataFrame,
    *,
    cohort: dict[str, Any],
) -> None:
    mouse_required = {
        "subject_id",
        "cohort_assignment",
        "allocation_status",
        "split_seed",
        "confirmation_access_policy",
        "mouse_behavior_eligible",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "neural_activity_used_for_allocation",
    }
    session_required = mouse_required | {
        "_nwb_path",
        "session_behavior_eligible",
        "discovery_analysis_included",
        "confirmation_analysis_included",
    }
    if missing := sorted(mouse_required.difference(mice.columns)):
        raise RuntimeError(f"D05 mouse assignments lack required columns: {missing}")
    if missing := sorted(session_required.difference(sessions.columns)):
        raise RuntimeError(f"D05 session assignments lack required columns: {missing}")
    if mice.is_empty() or sessions.is_empty():
        raise RuntimeError("D05 assignment Parquets must not be empty")

    _validate_string_key(mice, "subject_id", label="D05 mouse assignments")
    _validate_string_key(sessions, "_nwb_path", label="D05 session assignments")
    _validate_nonempty_strings(sessions, "subject_id", label="D05 session assignments")
    for frame, label in ((mice, "mouse"), (sessions, "session")):
        _validate_boolean_columns(
            frame,
            (
                "mouse_behavior_eligible",
                "ready_for_neural_discovery",
                "discovery_mouse_included",
                "confirmation_mouse_included",
                "neural_activity_used_for_allocation",
            ),
            label=f"D05 {label} assignments",
        )
        if (
            set(frame.get_column("allocation_status").drop_nulls().unique().to_list())
            != {D05_FINAL_ASSIGNMENT_STATUS}
            or frame.get_column("allocation_status").null_count()
        ):
            raise RuntimeError(f"D05 {label} assignments are not uniformly final")
        if (
            set(frame.get_column("split_seed").drop_nulls().unique().to_list()) != {D05_SPLIT_SEED}
            or frame.get_column("split_seed").null_count()
        ):
            raise RuntimeError(f"D05 {label} assignments use an unexpected split seed")
        if not frame.get_column("ready_for_neural_discovery").all():
            raise RuntimeError(f"D05 {label} assignment readiness is not uniformly true")
        if frame.get_column("confirmation_mouse_included").any():
            raise RuntimeError(f"D05 {label} assignments activate confirmation mice")
        if frame.get_column("neural_activity_used_for_allocation").any():
            raise RuntimeError(f"D05 {label} assignments report outcome-informed allocation")

    _validate_boolean_columns(
        sessions,
        (
            "session_behavior_eligible",
            "discovery_analysis_included",
            "confirmation_analysis_included",
        ),
        label="D05 session assignments",
    )
    allowed_cohorts = {"discovery", "confirmation", "excluded"}
    mouse_cohorts = set(mice.get_column("cohort_assignment").drop_nulls().unique().to_list())
    session_cohorts = set(sessions.get_column("cohort_assignment").drop_nulls().unique().to_list())
    if (
        mice.get_column("cohort_assignment").null_count()
        or sessions.get_column("cohort_assignment").null_count()
        or not {"discovery", "confirmation"}.issubset(mouse_cohorts)
        or not mouse_cohorts.issubset(allowed_cohorts)
        or not session_cohorts.issubset(allowed_cohorts)
    ):
        raise RuntimeError("D05 assignment Parquets contain invalid cohort labels")

    mouse_rows = {
        row["subject_id"]: row
        for row in mice.select(
            "subject_id",
            "cohort_assignment",
            "mouse_behavior_eligible",
            "discovery_mouse_included",
            "confirmation_mouse_included",
            "confirmation_access_policy",
        ).iter_rows(named=True)
    }
    if set(sessions.get_column("subject_id").to_list()) != set(mouse_rows):
        raise RuntimeError("D05 mouse and session assignment identities differ")
    expected_policy = {
        "discovery": "discovery_activated_confirmation_sealed",
        "confirmation": "sealed_until_one_time_confirmation_gate",
        "excluded": "not_applicable",
    }
    for row in mice.iter_rows(named=True):
        expected_eligible = row["cohort_assignment"] != "excluded"
        if (
            row["mouse_behavior_eligible"] is not expected_eligible
            or row["discovery_mouse_included"] is not (row["cohort_assignment"] == "discovery")
            or row["confirmation_access_policy"] != expected_policy[row["cohort_assignment"]]
        ):
            raise RuntimeError("D05 mouse assignment flags disagree with sealed cohort status")
    for row in sessions.iter_rows(named=True):
        mouse = mouse_rows.get(row["subject_id"])
        if mouse is None or any(
            row[field] != mouse[field]
            for field in (
                "cohort_assignment",
                "mouse_behavior_eligible",
                "discovery_mouse_included",
                "confirmation_mouse_included",
                "confirmation_access_policy",
            )
        ):
            raise RuntimeError("D05 session assignments disagree with mouse assignments")
        expected_discovery = (
            row["cohort_assignment"] == "discovery" and row["session_behavior_eligible"]
        )
        if row["discovery_analysis_included"] is not expected_discovery:
            raise RuntimeError("D05 discovery activation flags disagree with final allocation")
        if row["confirmation_analysis_included"]:
            raise RuntimeError("D05 session assignments activate the confirmation holdout")

    cohort_ids = {
        name: sorted(
            mice.filter(pl.col("cohort_assignment") == name).get_column("subject_id").to_list()
        )
        for name in ("discovery", "confirmation", "excluded")
    }
    if (
        len(cohort_ids["discovery"]) != D05_DISCOVERY_MICE
        or len(cohort_ids["confirmation"]) != D05_CONFIRMATION_MICE
    ):
        raise RuntimeError("D05 assignment mouse counts differ from the locked 12/6 split")
    identity = cohort.get("assignment_identity")
    if not isinstance(identity, dict) or any(
        identity.get(f"{name}_mouse_ids") != identifiers for name, identifiers in cohort_ids.items()
    ):
        raise RuntimeError("D05 assignment identities disagree with the frozen manifest")

    expected_counts = {
        "inventory_mice": mice.height,
        "eligible_mice": len(cohort_ids["discovery"]) + len(cohort_ids["confirmation"]),
        "excluded_mice": len(cohort_ids["excluded"]),
        "discovery_mice": len(cohort_ids["discovery"]),
        "confirmation_mice": len(cohort_ids["confirmation"]),
        "inventory_sessions": sessions.height,
        "threshold_selected_sessions": sessions.filter(pl.col("session_behavior_eligible")).height,
        "discovery_threshold_selected_sessions": sessions.filter(
            (pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible")
        ).height,
        "confirmation_threshold_selected_sessions": sessions.filter(
            (pl.col("cohort_assignment") == "confirmation") & pl.col("session_behavior_eligible")
        ).height,
    }
    counts = cohort.get("cohort_counts")
    if not isinstance(counts, dict) or any(
        isinstance(counts.get(name), bool) or counts.get(name) != expected
        for name, expected in expected_counts.items()
    ):
        raise RuntimeError("D05 assignment counts disagree with the frozen manifest")


def _validate_string_key(frame: pl.DataFrame, column: str, *, label: str) -> None:
    _validate_nonempty_strings(frame, column, label=label)
    if frame.get_column(column).n_unique() != frame.height:
        raise RuntimeError(f"{label} {column} is not unique")


def _validate_nonempty_strings(frame: pl.DataFrame, column: str, *, label: str) -> None:
    values = frame.get_column(column)
    if values.dtype != pl.String or values.null_count():
        raise RuntimeError(f"{label} {column} must contain non-null strings")
    if frame.filter(pl.col(column).str.strip_chars() == "").height:
        raise RuntimeError(f"{label} {column} contains empty values")
    if frame.filter(pl.col(column) != pl.col(column).str.strip_chars()).height:
        raise RuntimeError(f"{label} {column} contains non-canonical whitespace")


def _validate_boolean_columns(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    for column in columns:
        values = frame.get_column(column)
        if values.dtype != pl.Boolean or values.null_count():
            raise RuntimeError(f"{label} {column} must contain non-null booleans")


def _validated_artifact_payload(
    record: Any,
    *,
    repository_root: pathlib.Path,
    label: str,
    changed_message: str | None = None,
) -> tuple[pathlib.Path, bytes]:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} is not a mapping")
    path_value = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"{label} lacks a path")
    if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(expected_hash):
        raise RuntimeError(f"{label} lacks a SHA-256 hash")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise RuntimeError(f"{label} lacks a non-negative integer size")
    unresolved = pathlib.Path(path_value)
    path = (
        unresolved.resolve()
        if unresolved.is_absolute()
        else (repository_root / unresolved).resolve()
    )
    if not path.is_relative_to(repository_root):
        raise RuntimeError(f"{label} escapes the repository")
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise RuntimeError(f"{label} does not exist or cannot be read: {error}") from error
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was read")
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_hash:
        raise RuntimeError(changed_message or f"{label} content differs from its recorded hash")
    return path, payload


def _read_json_artifact(
    record: dict[str, Any],
    repository_root: pathlib.Path,
) -> dict[str, Any]:
    path, payload = _validated_artifact_payload(
        record,
        repository_root=repository_root,
        label="frozen JSON artifact",
    )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"could not read frozen JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"frozen JSON artifact {path} must contain an object")
    return value


def _validate_timestamp(value: Any, *, label: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} timestamp is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"{label} timestamp must include a timezone")


def _canonical_json_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _claim_marker_path(gate_path: pathlib.Path) -> pathlib.Path:
    return gate_path.with_name(f"{gate_path.name}.claimed")


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
