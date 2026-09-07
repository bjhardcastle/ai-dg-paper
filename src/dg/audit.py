"""Immutable DANDI inventory, NWB schema, and task-parameter audit utilities.

The functions in this module deliberately separate remote reads from pure
normalization.  The latter can therefore be tested without contacting DANDI,
while manuscript runs retain the exact published asset metadata used to
resolve every NWB source.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
import urllib.request
from typing import Any

import lazynwb
import polars as pl

import dg.data

ASSET_MANIFEST_URL = (
    "https://dandiarchive.s3.amazonaws.com/dandisets/"
    f"{dg.data.DANDISET_ID}/{dg.data.DANDISET_VERSION}/assets.jsonld"
)
COMPANION_TRIALS_SHA256 = "a29a3edd22b1f0838ddc1794139ba61d075e03c2d3fa5ba7d14e7caf7f9ebf7b"
COMPANION_SESSION_METADATA_SHA256 = (
    "2c5ba529375b1c1407c4c5f5cb40a428fd5d00fb17027039cd0b4f32574abb5f"
)

_SESSION_PATH_PATTERN = re.compile(
    r"^sub-(?P<subject>[^/]+)/sub-(?P=subject)_ses-(?P<session>\d{8}T\d{6})\.nwb$"
)
_PROBE_LFP_PATH_PATTERN = re.compile(r"_probe-[^/]+_ecephys\.nwb$")

REQUIRED_TABLE_COLUMNS = {
    dg.data.TRIALS_PATH: (
        "id",
        "start_time",
        "stop_time",
        "change_time",
        "initial_image_name",
        "change_image_name",
        "is_change",
        "go",
        "catch",
        "aborted",
        "auto_rewarded",
        "hit",
        "miss",
        "false_alarm",
        "correct_reject",
        "lick_times",
    ),
    dg.data.UNITS_PATH: (
        "id",
        "peak_channel_id",
        "spike_times",
        "isi_violations",
        "amplitude_cutoff",
        "quality",
    ),
    dg.data.ELECTRODES_PATH: ("id", "location", "x", "y", "z", "probe_id"),
}

AUDITED_OPTIONAL_TRIAL_COLUMNS = (
    "is_sham_change",
    "no_reward_epoch",
    "omitted_reward",
)

EXPECTED_RESPONSE_WINDOW_SEC = (0.15, 0.75)

TASK_PARAMETERS_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "subject_id": pl.String,
    "_nwb_path": pl.String,
    "task_parameters_path": pl.String,
    "response_window_sec": pl.String,
    "response_window_start_seconds": pl.Float64,
    "response_window_stop_seconds": pl.Float64,
    "response_window_matches_expected": pl.Boolean,
    "audit_error": pl.String,
}

NWB_ROOT_IDENTIFIER_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "_nwb_path": pl.String,
    "expected_subject_id": pl.String,
    "nwb_subject_id": pl.String,
    "subject_id_agrees": pl.Boolean,
    "expected_ecephys_session_id": pl.Int64,
    "nwb_ecephys_session_id": pl.Int64,
    "ecephys_session_id_valid": pl.Boolean,
    "ecephys_session_id_agrees": pl.Boolean,
    "session_identifier_conflict": pl.Boolean,
    "root_identifier_audit_pass": pl.Boolean,
    "audit_error": pl.String,
}


def fetch_asset_manifest(
    *,
    url: str = ASSET_MANIFEST_URL,
    retrieved_at: datetime.datetime | None = None,
    timeout_seconds: float = 120.0,
) -> pl.DataFrame:
    """Read and normalize the immutable version-level DANDI asset manifest."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = urllib.request.Request(url, headers={"User-Agent": "dg-manuscript-audit/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        records = json.load(response)
    if not isinstance(records, list):
        raise ValueError("DANDI assets.jsonld must contain a JSON list")
    return normalize_asset_manifest(records, retrieved_at=retrieved_at, manifest_url=url)


def normalize_asset_manifest(
    records: collections.abc.Iterable[collections.abc.Mapping[str, Any]],
    *,
    retrieved_at: datetime.datetime | None = None,
    manifest_url: str = ASSET_MANIFEST_URL,
) -> pl.DataFrame:
    """Convert DANDI JSON-LD asset records to a stable, tidy manifest."""

    timestamp = retrieved_at or datetime.datetime.now(datetime.UTC)
    if timestamp.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware")
    retrieval_time = timestamp.astimezone(datetime.UTC).isoformat()

    rows: list[dict[str, Any]] = []
    for record in records:
        path = record.get("path")
        identifier = record.get("identifier")
        if not isinstance(path, str) or not path:
            raise ValueError("every DANDI asset must have a non-empty path")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"asset {path!r} is missing its identifier")

        session_match = _SESSION_PATH_PATTERN.fullmatch(path)
        content_urls = record.get("contentUrl") or []
        s3_url = next(
            (
                value
                for value in content_urls
                if isinstance(value, str) and "dandiarchive.s3.amazonaws.com" in value
            ),
            None,
        )
        digest = record.get("digest") or {}
        participant = _first_schema_record(record.get("wasAttributedTo"), "Participant")
        acquisition = _first_schema_record(record.get("wasGeneratedBy"), "Session")
        rows.append(
            {
                "dandiset_id": dg.data.DANDISET_ID,
                "dandiset_version": dg.data.DANDISET_VERSION,
                "dandiset_doi": dg.data.DANDISET_DOI,
                "manifest_url": manifest_url,
                "retrieved_at_utc": retrieval_time,
                "asset_id": identifier.removeprefix("dandiasset:"),
                "path": path,
                "content_size_bytes": record.get("contentSize"),
                "dandi_etag": digest.get("dandi:dandi-etag"),
                "sha256": digest.get("dandi:sha2-256"),
                "encoding_format": record.get("encodingFormat"),
                "date_modified": record.get("dateModified"),
                "date_published": record.get("datePublished"),
                "s3_url": s3_url,
                "asset_kind": (
                    "session_nwb"
                    if session_match is not None
                    else "probe_lfp_nwb"
                    if _PROBE_LFP_PATH_PATTERN.search(path)
                    else "other"
                ),
                "subject_id": (
                    session_match.group("subject")
                    if session_match is not None
                    else _participant_identifier(participant)
                ),
                "session_start_time": (
                    acquisition.get("startDate") if acquisition is not None else None
                ),
                "recording_day": (
                    acquisition.get("description") if acquisition is not None else None
                ),
                "sex": _nested_name(participant, "sex"),
                "genotype": participant.get("genotype") if participant is not None else None,
                "species": _nested_name(participant, "species"),
                "age": _nested_value(participant, "age"),
                "n_probes": _count_equipment(acquisition),
            }
        )

    manifest = pl.DataFrame(rows, infer_schema_length=None)
    if manifest.is_empty():
        return manifest
    duplicates = manifest.filter(
        pl.col("asset_id").is_duplicated() | pl.col("path").is_duplicated()
    )
    if duplicates.height:
        raise ValueError("DANDI manifest contains duplicate asset IDs or paths")
    invalid_size = manifest.filter(
        pl.col("content_size_bytes").is_null() | (pl.col("content_size_bytes") < 0)
    )
    if invalid_size.height:
        raise ValueError("DANDI manifest contains missing or negative asset sizes")
    return manifest.sort("path")


def cache_companion_trials(
    destination: str | os.PathLike[str],
    *,
    source_url: str = dg.data.COMPANION_TRIALS_URL,
    expected_sha256: str = COMPANION_TRIALS_SHA256,
    timeout_seconds: float = 180.0,
) -> pl.DataFrame:
    """Cache and verify the commit-pinned companion trial table.

    The returned one-row table is suitable for the provenance manifest. An
    existing cache is reused only after its complete SHA-256 digest matches.
    """

    return _cache_verified_file(
        destination,
        source_url=source_url,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
    )


def cache_companion_session_metadata(
    destination: str | os.PathLike[str],
    *,
    source_url: str = dg.data.COMPANION_SESSION_METADATA_URL,
    expected_sha256: str = COMPANION_SESSION_METADATA_SHA256,
    timeout_seconds: float = 120.0,
) -> pl.DataFrame:
    """Cache and verify the pinned 99-session reconciliation table."""

    return _cache_verified_file(
        destination,
        source_url=source_url,
        expected_sha256=expected_sha256,
        timeout_seconds=timeout_seconds,
    )


def _cache_verified_file(
    destination: str | os.PathLike[str],
    *,
    source_url: str,
    expected_sha256: str,
    timeout_seconds: float,
) -> pl.DataFrame:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    output = pathlib.Path(destination).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.is_file():
        observed_sha256, content_size = _hash_file(output)
        if observed_sha256 == expected_sha256:
            output.chmod(0o644)
            return _companion_provenance(
                output,
                source_url,
                observed_sha256,
                content_size,
                cache_status="verified_existing",
            )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = pathlib.Path(temporary_name)
    digest = hashlib.sha256()
    content_size = 0
    try:
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "dg-manuscript-audit/0.1"},
        )
        with (
            os.fdopen(descriptor, "wb") as target,
            urllib.request.urlopen(request, timeout=timeout_seconds) as response,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                content_size += len(chunk)
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValueError(
                "companion table SHA-256 mismatch: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return _companion_provenance(
        output,
        source_url,
        observed_sha256,
        content_size,
        cache_status="downloaded_and_verified",
    )


def summarize_asset_manifest(manifest: pl.DataFrame) -> pl.DataFrame:
    """Summarize immutable asset counts and sizes by asset kind."""

    required = {"asset_id", "path", "asset_kind", "subject_id", "content_size_bytes"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return (
        manifest.group_by("asset_kind")
        .agg(
            pl.len().alias("n_assets"),
            pl.col("content_size_bytes").sum().alias("content_size_bytes"),
            pl.col("subject_id").drop_nulls().n_unique().alias("n_subjects"),
        )
        .sort("asset_kind")
    )


def get_session_sources(manifest: pl.DataFrame) -> list[str]:
    """Return traceable S3 URLs for session NWBs in manifest path order."""

    required = {"asset_kind", "path", "s3_url"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    sessions = manifest.filter(pl.col("asset_kind") == "session_nwb").sort("path")
    if sessions.get_column("s3_url").null_count():
        raise ValueError("one or more session NWBs have no public S3 URL")
    return sessions.get_column("s3_url").to_list()


def audit_nwb_schemas(
    session_manifest: pl.DataFrame,
    *,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Audit required NWB tables/columns in every session without loading arrays.

    Results are long-form: one row for every required or audited optional field
    in every session.  Internal-path discovery reads HDF5 metadata only.  It
    does not materialize spike arrays or other table columns.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    required = {"asset_id", "path", "s3_url", "subject_id"}
    missing = required.difference(session_manifest.columns)
    if missing:
        raise ValueError(f"session_manifest is missing columns: {sorted(missing)}")
    if session_manifest.get_column("asset_id").n_unique() != session_manifest.height:
        raise ValueError("session_manifest must have unique asset IDs")
    if session_manifest.get_column("s3_url").null_count():
        raise ValueError("session_manifest contains a null S3 URL")

    rows = session_manifest.select(*sorted(required)).to_dicts()
    lazynwb.config.anon = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        audited = list(executor.map(_audit_one_nwb_schema, rows))
    flat_rows = [row for group in audited for row in group]
    return pl.DataFrame(flat_rows, infer_schema_length=None).sort(
        "path", "table_path", "column_name"
    )


def normalize_response_window_sec(value: Any) -> tuple[float, float]:
    """Return a finite, increasing two-value task response window.

    Published task-parameter attributes encode the two values as strings in a
    list. JSON-array strings and other non-string iterables cover the observed
    attribute container encodings without weakening value validation.
    """

    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError("response_window_sec must be a two-value sequence") from error
    if isinstance(candidate, (str, bytes, collections.abc.Mapping)):
        raise ValueError("response_window_sec must be a two-value sequence")
    try:
        values = list(candidate)
    except TypeError as error:
        raise ValueError("response_window_sec must be a two-value sequence") from error
    if len(values) != 2:
        raise ValueError("response_window_sec must contain exactly two values")
    if any(isinstance(item, bool) for item in values):
        raise ValueError("response_window_sec values must be numeric, not boolean")
    try:
        start_seconds, stop_seconds = (float(item) for item in values)
    except (TypeError, ValueError) as error:
        raise ValueError("response_window_sec values must be numeric") from error
    if not math.isfinite(start_seconds) or not math.isfinite(stop_seconds):
        raise ValueError("response_window_sec values must be finite")
    if stop_seconds <= start_seconds:
        raise ValueError("response_window_sec stop must be greater than start")
    return start_seconds, stop_seconds


def audit_task_parameters(
    session_manifest: pl.DataFrame,
    *,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Read and normalize response-window attributes for every session NWB.

    Reads run concurrently, but failures remain scoped to their session so the
    complete audit table is written before required validation fails.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    required = {"asset_id", "path", "s3_url", "subject_id"}
    missing = required.difference(session_manifest.columns)
    if missing:
        raise ValueError(f"session_manifest is missing columns: {sorted(missing)}")
    if session_manifest.get_column("asset_id").n_unique() != session_manifest.height:
        raise ValueError("session_manifest must have unique asset IDs")
    if session_manifest.get_column("s3_url").null_count():
        raise ValueError("session_manifest contains a null S3 URL")

    rows = session_manifest.select(*sorted(required)).to_dicts()
    lazynwb.config.anon = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        audited = list(executor.map(_audit_one_task_parameters, rows))
    return pl.DataFrame(
        audited,
        schema=TASK_PARAMETERS_AUDIT_SCHEMA,
        orient="row",
    ).sort("path")


def audit_nwb_root_identifiers(
    session_inventory: pl.DataFrame,
    *,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Reconcile root NWB subject/session identifiers one source at a time.

    Per-source reads run concurrently because a single batched metadata read can
    obscure which remote file failed.  Errors remain scoped to an output row so
    the complete reconciliation is inspectable before the hard gate is applied.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    required = {"asset_id", "path", "_nwb_path", "subject_id", "ecephys_session_id"}
    missing = required.difference(session_inventory.columns)
    if missing:
        raise ValueError(f"session_inventory is missing columns: {sorted(missing)}")
    if session_inventory.is_empty():
        raise ValueError("session_inventory contains no sessions")
    if session_inventory.get_column("_nwb_path").n_unique() != session_inventory.height:
        raise ValueError("session_inventory must contain one row per NWB source")

    rows = session_inventory.select(*sorted(required)).to_dicts()
    lazynwb.config.anon = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        audited = list(executor.map(_audit_one_nwb_root_identifier, rows))
    return pl.DataFrame(
        audited,
        schema=NWB_ROOT_IDENTIFIER_AUDIT_SCHEMA,
        orient="row",
    ).sort("path")


def read_nwb_root_identity(source: str) -> dict[str, Any]:
    """Read only the three scalar NWB datasets needed for identity reconciliation."""

    lazynwb.config.anon = True
    nwb = lazynwb.LazyNWB(source)
    return {
        "identifier": nwb.identifier,
        "session_id": nwb.session_id,
        "subject_id": nwb.subject.subject_id,
    }


def validate_nwb_root_identifier_audit(
    root_identifier_audit: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Fail unless root identifiers pass for every frozen session source."""

    required = {"_nwb_path", "root_identifier_audit_pass", "audit_error"}
    missing = required.difference(root_identifier_audit.columns)
    if missing:
        raise ValueError(f"root_identifier_audit is missing columns: {sorted(missing)}")
    expected_sources = set(session_inventory.get_column("_nwb_path").to_list())
    observed_sources = set(root_identifier_audit.get_column("_nwb_path").to_list())
    if (
        root_identifier_audit.height != len(expected_sources)
        or root_identifier_audit.get_column("_nwb_path").n_unique() != root_identifier_audit.height
        or observed_sources != expected_sources
    ):
        raise ValueError("root identifier audit does not exactly cover the session inventory")
    failures = root_identifier_audit.filter(~pl.col("root_identifier_audit_pass").fill_null(False))
    if failures.height:
        raise ValueError(f"required NWB-root identifier reconciliation failed:\n{failures.head(5)}")


def validate_task_parameters_audit(
    task_parameters_audit: pl.DataFrame,
    *,
    expected_response_window_sec: tuple[float, float] = EXPECTED_RESPONSE_WINDOW_SEC,
) -> None:
    """Fail unless every audited session has the expected response window."""

    expected_start, expected_stop = normalize_response_window_sec(expected_response_window_sec)
    required = {
        "asset_id",
        "path",
        "response_window_start_seconds",
        "response_window_stop_seconds",
        "audit_error",
    }
    missing = required.difference(task_parameters_audit.columns)
    if missing:
        raise ValueError(f"task_parameters_audit is missing columns: {sorted(missing)}")
    if task_parameters_audit.is_empty():
        raise ValueError("task_parameters_audit contains no session rows")
    if task_parameters_audit.get_column("asset_id").n_unique() != task_parameters_audit.height:
        raise ValueError("task_parameters_audit must have one row per asset ID")

    failures: list[dict[str, Any]] = []
    for row in task_parameters_audit.select(*sorted(required)).iter_rows(named=True):
        start_seconds = row["response_window_start_seconds"]
        stop_seconds = row["response_window_stop_seconds"]
        matches_expected = (
            row["audit_error"] is None
            and start_seconds is not None
            and stop_seconds is not None
            and start_seconds == expected_start
            and stop_seconds == expected_stop
        )
        if not matches_expected:
            failures.append(row)
    if failures:
        example = pl.DataFrame(failures, infer_schema_length=None).select(
            "path",
            "response_window_start_seconds",
            "response_window_stop_seconds",
            "audit_error",
        )
        raise ValueError(
            "required task-parameter audit failed; expected response_window_sec "
            f"{list(expected_response_window_sec)} in every session:\n{example.head(5)}"
        )


def summarize_schema_audit(schema_audit: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-session field availability without obscuring failures."""

    required = {"table_path", "column_name", "required", "present", "audit_error"}
    missing = required.difference(schema_audit.columns)
    if missing:
        raise ValueError(f"schema_audit is missing columns: {sorted(missing)}")
    return (
        schema_audit.group_by("table_path", "column_name", "required")
        .agg(
            pl.len().alias("n_sessions_audited"),
            pl.col("present").fill_null(False).sum().alias("n_sessions_present"),
            pl.col("audit_error").is_not_null().sum().alias("n_sessions_failed_audit"),
        )
        .with_columns(
            (pl.col("n_sessions_audited") - pl.col("n_sessions_present")).alias(
                "n_sessions_missing"
            )
        )
        .sort("table_path", "column_name")
    )


def _audit_one_task_parameters(asset: dict[str, Any]) -> dict[str, Any]:
    common = {
        "asset_id": asset["asset_id"],
        "path": asset["path"],
        "subject_id": asset["subject_id"],
        "_nwb_path": asset["s3_url"],
        "task_parameters_path": dg.data.TASK_PARAMETERS_PATH,
    }
    response_window: Any = None
    try:
        task_parameters = dg.data.get_task_parameters(asset["s3_url"])
        if not isinstance(task_parameters, collections.abc.Mapping):
            raise ValueError("task parameters must be a mapping")
        if "response_window_sec" not in task_parameters:
            raise ValueError("task parameters are missing response_window_sec")
        response_window = task_parameters["response_window_sec"]
        start_seconds, stop_seconds = normalize_response_window_sec(response_window)
        audit_error = None
    except Exception as error:  # pragma: no cover - remote failures covered via mocking
        start_seconds = None
        stop_seconds = None
        audit_error = f"{type(error).__name__}: {error}"

    return {
        **common,
        "response_window_sec": _serialize_attribute(response_window),
        "response_window_start_seconds": start_seconds,
        "response_window_stop_seconds": stop_seconds,
        "response_window_matches_expected": (
            audit_error is None
            and start_seconds == EXPECTED_RESPONSE_WINDOW_SEC[0]
            and stop_seconds == EXPECTED_RESPONSE_WINDOW_SEC[1]
        ),
        "audit_error": audit_error,
    }


def _audit_one_nwb_root_identifier(asset: dict[str, Any]) -> dict[str, Any]:
    expected_subject_id = str(asset["subject_id"]).removeprefix("sub-")
    expected_session_id = int(asset["ecephys_session_id"])
    nwb_subject_id: str | None = None
    nwb_session_id: int | None = None
    session_id_valid = False
    session_identifier_conflict = False
    try:
        identity = read_nwb_root_identity(asset["_nwb_path"])
        raw_subject_id = identity.get("subject_id")
        if raw_subject_id is not None:
            nwb_subject_id = str(raw_subject_id).removeprefix("sub-")
        raw_identifier = identity.get("identifier")
        if isinstance(raw_identifier, bool):
            raise ValueError("NWB identifier must be an integer-like scalar, not boolean")
        try:
            nwb_session_id = int(raw_identifier)
        except (TypeError, ValueError) as error:
            raise ValueError("NWB identifier is not an integer-like scalar") from error
        session_id_valid = True
        raw_session_id = identity.get("session_id")
        if raw_session_id is not None:
            try:
                session_identifier_conflict = int(raw_session_id) != nwb_session_id
            except (TypeError, ValueError):
                session_identifier_conflict = True
        audit_error = None
    except Exception as error:  # pragma: no cover - remote failures covered via mocking
        audit_error = f"{type(error).__name__}: {error}"

    subject_agrees = nwb_subject_id == expected_subject_id
    session_agrees = nwb_session_id == expected_session_id
    passed = (
        audit_error is None
        and subject_agrees
        and session_id_valid
        and session_agrees
        and not session_identifier_conflict
    )
    return {
        "asset_id": asset["asset_id"],
        "path": asset["path"],
        "_nwb_path": asset["_nwb_path"],
        "expected_subject_id": expected_subject_id,
        "nwb_subject_id": nwb_subject_id,
        "subject_id_agrees": subject_agrees,
        "expected_ecephys_session_id": expected_session_id,
        "nwb_ecephys_session_id": nwb_session_id,
        "ecephys_session_id_valid": session_id_valid,
        "ecephys_session_id_agrees": session_agrees,
        "session_identifier_conflict": session_identifier_conflict,
        "root_identifier_audit_pass": passed,
        "audit_error": audit_error,
    }


def _serialize_attribute(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _audit_one_nwb_schema(asset: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "asset_id": asset["asset_id"],
        "path": asset["path"],
        "subject_id": asset["subject_id"],
        "_nwb_path": asset["s3_url"],
    }
    internal_path_checks = [
        (table_path, column, True)
        for table_path, columns in REQUIRED_TABLE_COLUMNS.items()
        if table_path != dg.data.ELECTRODES_PATH
        for column in columns
    ]
    internal_path_checks.extend(
        (dg.data.TRIALS_PATH, column, False) for column in AUDITED_OPTIONAL_TRIAL_COLUMNS
    )
    try:
        paths = set(lazynwb.get_internal_paths(asset["s3_url"], include_table_columns=True))
        image_candidates = _image_presentation_candidates(paths)
        internal_path_error = None
    except Exception as error:  # pragma: no cover - exercised only by remote I/O failures
        paths = set()
        image_candidates = None
        internal_path_error = f"{type(error).__name__}: {error}"

    try:
        electrode_schema = lazynwb.get_table_schema(
            asset["s3_url"],
            dg.data.ELECTRODES_PATH,
            exclude_array_columns=True,
            exclude_internal_columns=True,
            raise_on_missing=True,
        )
        electrode_columns = set(electrode_schema.names())
        electrode_error = None
    except Exception as error:  # pragma: no cover - exercised only by remote I/O failures
        electrode_columns = set()
        electrode_error = f"{type(error).__name__}: {error}"

    output = []
    for table_path, column, is_required in internal_path_checks:
        output.append(
            {
                **common,
                "table_path": table_path,
                "column_name": column,
                "required": is_required,
                "present": (None if internal_path_error else f"{table_path}/{column}" in paths),
                "n_image_presentation_candidates": (
                    None if image_candidates is None else len(image_candidates)
                ),
                "image_presentation_paths": (
                    None if image_candidates is None else ";".join(image_candidates)
                ),
                "audit_error": internal_path_error,
            }
        )
    for column in REQUIRED_TABLE_COLUMNS[dg.data.ELECTRODES_PATH]:
        output.append(
            {
                **common,
                "table_path": dg.data.ELECTRODES_PATH,
                "column_name": column,
                "required": True,
                "present": None if electrode_error else column in electrode_columns,
                "n_image_presentation_candidates": (
                    None if image_candidates is None else len(image_candidates)
                ),
                "image_presentation_paths": (
                    None if image_candidates is None else ";".join(image_candidates)
                ),
                "audit_error": electrode_error,
            }
        )
    return output


def _image_presentation_candidates(paths: set[str]) -> list[str]:
    parents: dict[str, set[str]] = {}
    for path in paths:
        if not path.startswith("/intervals/") or "/" not in path[1:]:
            continue
        parent, column = path.rsplit("/", maxsplit=1)
        parents.setdefault(parent, set()).add(column)
    required = {"image_name", "is_change", "active"}
    excluded = {"flash", "gabor", "spontaneous"}
    return sorted(
        parent
        for parent, columns in parents.items()
        if parent.endswith("_presentations")
        and required.issubset(columns)
        and not any(term in parent.lower() for term in excluded)
    )


def _first_schema_record(value: Any, schema_key: str) -> collections.abc.Mapping[str, Any] | None:
    if not isinstance(value, list):
        return None
    return next(
        (
            item
            for item in value
            if isinstance(item, collections.abc.Mapping) and item.get("schemaKey") == schema_key
        ),
        None,
    )


def _participant_identifier(
    participant: collections.abc.Mapping[str, Any] | None,
) -> str | None:
    if participant is None:
        return None
    identifier = participant.get("identifier")
    return str(identifier) if identifier is not None else None


def _nested_name(record: collections.abc.Mapping[str, Any] | None, key: str) -> str | None:
    if record is None or not isinstance(record.get(key), collections.abc.Mapping):
        return None
    value = record[key].get("name")
    return str(value) if value is not None else None


def _nested_value(record: collections.abc.Mapping[str, Any] | None, key: str) -> str | None:
    if record is None or not isinstance(record.get(key), collections.abc.Mapping):
        return None
    value = record[key].get("value")
    return str(value) if value is not None else None


def _count_equipment(session: collections.abc.Mapping[str, Any] | None) -> int | None:
    if session is None or not isinstance(session.get("used"), list):
        return None
    return sum(
        isinstance(item, collections.abc.Mapping) and item.get("schemaKey") == "Equipment"
        for item in session["used"]
    )


def _hash_file(path: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    content_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            content_size += len(chunk)
    return digest.hexdigest(), content_size


def _companion_provenance(
    path: pathlib.Path,
    source_url: str,
    sha256: str,
    content_size: int,
    *,
    cache_status: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_url": [source_url],
            "repository_commit": [dg.data.COMPANION_REPOSITORY_COMMIT],
            "sha256": [sha256],
            "content_size_bytes": [content_size],
            "cached_path": [str(path)],
            "cache_status": [cache_status],
        }
    )
