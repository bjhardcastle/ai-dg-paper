"""Human-review gates for discovery and one-time confirmation analyses."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
from typing import Any

REQUIRED_DECISION_IDS = tuple(f"D{index:02d}" for index in range(1, 16))
AUDIT_GATE_DECISIONS = ("D01", "D02")
BEHAVIOR_QC_GATE_DECISIONS = (
    "D03",
    "D04",
    "D05",
    "D06",
    "D12",
    "D13",
    "D14",
)
DISCOVERY_GATE_DECISIONS = ("D07", "D08")
ANATOMY_NETWORK_GATE_DECISIONS = ("D09", "D10", "D11")
CONFIRMATION_GATE_DECISIONS = tuple(f"D{index:02d}" for index in range(1, 15))


def read_analysis_lock(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the repository's JSON-compatible YAML analysis lock."""

    lock_path = pathlib.Path(path)
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read analysis lock {lock_path}: {error}") from error
    validate_analysis_lock(value)
    return value


def validate_analysis_lock(lock: dict[str, Any]) -> None:
    """Validate decision coverage and approval metadata."""

    if lock.get("schema_version") != 1:
        raise ValueError("analysis lock schema_version must equal 1")
    overall_status = lock.get("overall_status")
    if overall_status not in {"proposed", "approved"}:
        raise ValueError("analysis lock overall_status must be proposed or approved")
    decisions = lock.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("analysis lock must contain a decisions mapping")
    missing = set(REQUIRED_DECISION_IDS).difference(decisions)
    unexpected = set(decisions).difference(REQUIRED_DECISION_IDS)
    if missing or unexpected:
        raise ValueError(
            f"analysis lock decision IDs differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for decision_id, decision in decisions.items():
        if not isinstance(decision, dict):
            raise ValueError(f"{decision_id} must be a mapping")
        status = decision.get("status")
        if status not in {"proposed", "approved", "rejected", "superseded"}:
            raise ValueError(f"{decision_id} has invalid status {status!r}")
        if status == "approved" and not (
            decision.get("approved_by") and decision.get("approved_at")
        ):
            raise ValueError(f"{decision_id} is approved without reviewer and timestamp")
        if status == "approved":
            _validate_approval_timestamp(decision["approved_at"], label=decision_id)

    all_approved = all(decision.get("status") == "approved" for decision in decisions.values())
    if overall_status == "approved" and not all_approved:
        raise ValueError("analysis lock cannot be approved while a decision is unapproved")
    if all_approved and overall_status != "approved":
        raise ValueError("analysis lock overall_status must be approved when all decisions are")
    if overall_status == "approved":
        approval_record = lock.get("approval_record")
        if not isinstance(approval_record, dict):
            raise ValueError("approved analysis lock requires an approval_record")
        approved_by = approval_record.get("approved_by")
        approved_at = approval_record.get("approved_at")
        if not approved_by or not approved_at:
            raise ValueError("analysis lock approval_record requires reviewer and timestamp")
        _validate_approval_timestamp(approved_at, label="approval_record")
        for decision_id, decision in decisions.items():
            if decision["approved_by"] != approved_by or decision["approved_at"] != approved_at:
                raise ValueError(
                    f"{decision_id} approval metadata differs from the overall approval record"
                )


def _validate_approval_timestamp(value: Any, *, label: str) -> None:
    """Require an ISO-8601 approval time with an explicit UTC offset."""

    if not isinstance(value, str):
        raise ValueError(f"{label} approved_at must be an ISO-8601 string")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} approved_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} approved_at must include a timezone")


def unapproved_decisions(
    lock: dict[str, Any],
    decision_ids: tuple[str, ...],
) -> list[str]:
    """Return requested decisions that lack complete human approval."""

    validate_analysis_lock(lock)
    return [
        decision_id
        for decision_id in decision_ids
        if lock["decisions"][decision_id].get("status") != "approved"
    ]


def require_approved_decisions(
    lock: dict[str, Any],
    decision_ids: tuple[str, ...],
    *,
    stage: str,
) -> None:
    """Stop a gated analysis stage unless all required decisions are approved."""

    pending = unapproved_decisions(lock, decision_ids)
    if pending:
        joined = ", ".join(pending)
        raise RuntimeError(f"{stage} is locked pending human approval of: {joined}")
