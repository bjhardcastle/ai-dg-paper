"""Tests for explicit human-review gates."""

import copy
import pathlib

import pytest

import dg.gates

LOCK_PATH = pathlib.Path("config/analysis_lock.yaml")


def test_repository_analysis_decisions_are_complete_and_approved() -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)

    assert set(lock["decisions"]) == set(dg.gates.REQUIRED_DECISION_IDS)
    assert lock["overall_status"] == "approved"
    assert lock["approval_record"]["approved_at"] == "2026-09-07T03:15:53Z"
    assert {decision["status"] for decision in lock["decisions"].values()} == {"approved"}
    assert {decision["approved_at"] for decision in lock["decisions"].values()} == {
        lock["approval_record"]["approved_at"]
    }
    assert {decision["approved_by"] for decision in lock["decisions"].values()} == {
        lock["approval_record"]["approved_by"]
    }
    assert dg.gates.unapproved_decisions(lock, dg.gates.CONFIRMATION_GATE_DECISIONS) == []
    # This proves only that the high-level decisions are approved. The
    # content-addressed one-time confirmation gate is tested separately.


def test_confirmation_gate_fails_closed() -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)
    incomplete = copy.deepcopy(lock)
    incomplete["overall_status"] = "proposed"
    incomplete.pop("approval_record")
    incomplete["decisions"]["D01"]["status"] = "proposed"

    with pytest.raises(RuntimeError, match="confirmation is locked"):
        dg.gates.require_approved_decisions(
            incomplete,
            dg.gates.CONFIRMATION_GATE_DECISIONS,
            stage="confirmation",
        )


def test_approved_decisions_require_reviewer_metadata() -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)
    invalid = copy.deepcopy(lock)
    invalid["decisions"]["D01"].pop("approved_by")

    with pytest.raises(ValueError, match="without reviewer"):
        dg.gates.validate_analysis_lock(invalid)


def test_decision_approval_check_passes_after_complete_approvals() -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)

    dg.gates.require_approved_decisions(
        lock,
        dg.gates.CONFIRMATION_GATE_DECISIONS,
        stage="confirmation decision approvals",
    )


def test_overall_approval_must_match_decisions_and_metadata() -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)
    invalid_status = copy.deepcopy(lock)
    invalid_status["decisions"]["D01"]["status"] = "proposed"
    with pytest.raises(ValueError, match="decision is unapproved"):
        dg.gates.validate_analysis_lock(invalid_status)

    invalid_metadata = copy.deepcopy(lock)
    invalid_metadata["decisions"]["D01"]["approved_at"] = "2026-09-07T03:16:00Z"
    with pytest.raises(ValueError, match="differs from the overall approval record"):
        dg.gates.validate_analysis_lock(invalid_metadata)


@pytest.mark.parametrize("timestamp", ["not-a-time", "2026-09-07T03:15:53"])
def test_approval_timestamp_must_be_iso8601_and_timezone_aware(timestamp: str) -> None:
    lock = dg.gates.read_analysis_lock(LOCK_PATH)
    invalid = copy.deepcopy(lock)
    invalid["approval_record"]["approved_at"] = timestamp
    invalid["decisions"]["D01"]["approved_at"] = timestamp

    with pytest.raises(ValueError, match="approved_at"):
        dg.gates.validate_analysis_lock(invalid)
